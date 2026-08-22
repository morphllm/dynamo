# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send SGLang scheduler batches to independent Dynamo response streams.

SGLang updates all request states in one ``_handle_batch_output`` call, then
wakes one Python task for each request. Under large decode batches, those tasks
serialize token conversion and response delivery under the GIL.

This adapter keeps SGLang's batch callback as the owner of request state,
timing, tracing, metrics, and final cleanup. It bypasses per-request task
wakeups only for successful intermediate token outputs. The outputs cross
Python to Rust in one call, but remain separate ordered response frames on
their original request streams.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class _NullEvent:
    """Receive SGLang's notification without waking the request task."""

    def set(self) -> None:
        return None


class _ContextWithRequestId:
    """Use Dynamo's request ID as SGLang RID when no trace ID exists."""

    __slots__ = ("_context", "trace_id")

    def __init__(self, context: Any, rid: str) -> None:
        self._context = context
        self.trace_id = rid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)


@dataclass(slots=True)
class _FanoutEntry:
    handler: Any
    context: Any
    response_sender: Any
    return_tokens_as_token_ids: bool
    armed: bool = False


class _FanoutRegistry:
    """Event-loop-local mapping from SGLang request IDs to Rust sinks."""

    def __init__(self) -> None:
        self._entries: dict[str, _FanoutEntry] = {}

    def add(self, rid: str, entry: _FanoutEntry) -> bool:
        if rid in self._entries:
            return False
        self._entries[rid] = entry
        return True

    def get(self, rid: str) -> _FanoutEntry | None:
        return self._entries.get(rid)

    def remove(self, rid: str, entry: _FanoutEntry) -> None:
        if self._entries.get(rid) is entry:
            del self._entries[rid]


_REGISTRY = _FanoutRegistry()


def _eligible(handler: Any, request: dict[str, Any], context: Any) -> bool:
    """Select requests that use the supported Dynamo token-stream path."""

    from dynamo.sglang.engine_generate import native_generate_payload

    if context is None or handler.use_sglang_tokenizer:
        return False
    if (request.get("sampling_options") or {}).get("n") not in (None, 1):
        return False
    if getattr(getattr(handler.config, "dynamo_args", None), "enable_rl", False):
        # RL metadata upload is asynchronous and stays on the normal path.
        return False
    return native_generate_payload(request) is None


def _convert_intermediate(
    entry: _FanoutEntry, res: dict[str, Any]
) -> dict[str, Any] | None:
    """Apply Dynamo's normal non-final token conversion in the batch task."""

    context = entry.context
    meta_info = res.get("meta_info", {})
    finish_reason = meta_info["finish_reason"]
    if finish_reason:
        raise ValueError("batch fanout received a final response")

    output_ids = res.get("output_ids", [])
    if not output_ids:
        return None

    out: dict[str, Any] = {
        "index": res.get("index") or 0,
        "token_ids": output_ids,
    }
    log_probs, top_logprobs = entry.handler._extract_logprobs(
        meta_info,
        num_output_tokens_in_chunk=len(output_ids),
        return_tokens_as_token_ids=entry.return_tokens_as_token_ids,
    )
    if log_probs is not None:
        out["log_probs"] = log_probs
    if top_logprobs is not None:
        out["top_logprobs"] = top_logprobs

    engine_data: dict[str, Any] = dict(res.get("engine_data") or {})
    routed_experts = meta_info.get("routed_experts")
    if routed_experts is not None:
        engine_data["routed_experts"] = routed_experts
    if engine_data:
        out["engine_data"] = engine_data

    if context.is_stopped():
        return None
    return out


def _mark_response_sent(state: Any, raw_output: dict[str, Any]) -> None:
    """Preserve SGLang's first response-sent timestamp semantics."""

    if state.time_stats.response_sent_to_client_time:
        return
    state.time_stats.set_response_sent_to_client_time()
    raw_output["meta_info"]["response_sent_to_client_ts"] = (
        state.time_stats.get_response_sent_to_client_realtime()
    )


def _restore_to_request_task(
    state: Any, raw_output: dict[str, Any], event: Any
) -> None:
    """Return one output to the unchanged per-request path."""

    state.out_list.append(raw_output)
    event.set()


def _install_tokenizer_batch_hook() -> None:
    """Install the SGLang v0.5 batch-output adapter once per process."""

    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    original = TokenizerManager._handle_batch_output
    if getattr(original, "_dynamo_batch_fanout", False):
        return

    @functools.wraps(original)
    async def handle_batch_output(self: Any, recv_obj: Any) -> None:
        # Take this snapshot before SGLang deletes finished request states.
        intercepted: dict[str, tuple[Any, Any]] = {}
        use_normal_batch = False
        for rid in recv_obj.rids:
            entry = _REGISTRY.get(rid)
            state = self.rid_to_state.get(rid)
            if entry is None or state is None:
                use_normal_batch = True
                continue
            if not entry.armed:
                # The existing request task must consume its first output. This
                # gives the existing cancellation monitor the SGLang request ID.
                entry.armed = True
                use_normal_batch = True
                continue
            event = state.event
            state.event = _NullEvent()
            intercepted[rid] = (state, event)

        # Keep mixed scheduler batches fully on the normal path. Changing the
        # notification behavior for part of a batch can affect other requests.
        if use_normal_batch or len(intercepted) != len(recv_obj.rids):
            for state, event in intercepted.values():
                state.event = event
            await original(self, recv_obj)
            return

        # A value above this batch size prevents the original method's
        # cooperative sleep. No intercepted Event is set, so no request task
        # can run before this callback regains control.
        old_batch_notify_size = self.server_args.batch_notify_size
        self.server_args.batch_notify_size = len(recv_obj.rids) + 1
        try:
            await original(self, recv_obj)
        finally:
            self.server_args.batch_notify_size = old_batch_notify_size
            for state, event in intercepted.values():
                state.event = event

        candidates: list[
            tuple[str, _FanoutEntry, Any, dict[str, Any], Any, Any, dict[str, Any]]
        ] = []
        for rid, (state, event) in intercepted.items():
            raw_outputs = state.out_list
            state.out_list = []

            # Final outputs, aborts, request logging, metrics export, cleanup,
            # and typed errors remain on the normal task path.
            if state.finished:
                for raw_output in raw_outputs:
                    _restore_to_request_task(state, raw_output, event)
                continue

            entry = _REGISTRY.get(rid)
            if entry is None:
                for raw_output in raw_outputs:
                    _restore_to_request_task(state, raw_output, event)
                continue

            for raw_output in raw_outputs:
                try:
                    _mark_response_sent(state, raw_output)
                    response = _convert_intermediate(entry, raw_output)
                except Exception:
                    logger.exception("batch fanout conversion failed for rid=%s", rid)
                    _REGISTRY.remove(rid, entry)
                    _restore_to_request_task(state, raw_output, event)
                    continue
                if response is not None:
                    candidates.append(
                        (
                            rid,
                            entry,
                            state,
                            raw_output,
                            event,
                            entry.response_sender,
                            response,
                        )
                    )

        if not candidates:
            return

        items = [(sender, response) for *_, sender, response in candidates]
        sender_type = type(items[0][0])
        try:
            failed = set(sender_type.send_batch(items))
        except Exception:
            logger.exception("native batch fanout rejected a scheduler batch")
            failed = set(range(len(candidates)))

        for index in failed:
            rid, entry, state, raw_output, event, _, _ = candidates[index]
            # Keep a failed request on the normal path. Otherwise a later batch
            # could take a restored output from the already-woken request task.
            _REGISTRY.remove(rid, entry)
            _restore_to_request_task(state, raw_output, event)

    handle_batch_output._dynamo_batch_fanout = True
    TokenizerManager._handle_batch_output = handle_batch_output


async def _drive_batch_fanout_stream(
    stream: AsyncGenerator[Any, None],
    entry: _FanoutEntry,
    rid: str,
) -> AsyncGenerator[Any, None]:
    """Keep request setup and finalization on the existing Dynamo stream."""

    try:
        async for response in stream:
            # Intermediate outputs use the batch hook. This stream still owns
            # the first output, final outputs, and all fallback cases.
            entry.response_sender.send(response)
        entry.response_sender.close()
    finally:
        _REGISTRY.remove(rid, entry)
    if False:  # pragma: no cover - required async-generator shape for Rust
        yield


def native_batch_fanout_capable(func: Any) -> Any:
    """Use native batch fanout for supported SGLang token requests.

    The parameter name is the opt-in used by the Rust push-egress engine. This
    decorator must stay outermost, and deleting ``__wrapped__`` makes
    ``inspect.signature`` report ``response_sender``.
    """

    # This import remains in the existing TRT-LLM module until the shared push
    # adapter gets a backend-neutral home.
    from dynamo.trtllm.request_handlers.push_egress import drive_push_egress_stream

    @functools.wraps(func)
    def dispatch(
        self: Any,
        request: dict[str, Any],
        context: Any = None,
        response_sender: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        if response_sender is None:
            return func(self, request, context, **kwargs)
        if not _eligible(self, request, context):
            return drive_push_egress_stream(
                func(self, request, context, **kwargs), response_sender
            )

        _install_tokenizer_batch_hook()
        rid = str(context.trace_id or context.id())
        effective_context = context
        if context.trace_id is None:
            effective_context = _ContextWithRequestId(context, rid)
        stream = func(self, request, effective_context, **kwargs)
        output_options = request.get("output_options") or {}
        entry = _FanoutEntry(
            handler=self,
            context=effective_context,
            response_sender=response_sender,
            return_tokens_as_token_ids=bool(
                output_options.get("return_tokens_as_token_ids")
            ),
        )
        if not _REGISTRY.add(rid, entry):
            logger.error("duplicate SGLang request ID; using normal push path: %s", rid)
            return drive_push_egress_stream(stream, response_sender)
        return _drive_batch_fanout_stream(stream, entry, rid)

    del dispatch.__wrapped__
    return dispatch
