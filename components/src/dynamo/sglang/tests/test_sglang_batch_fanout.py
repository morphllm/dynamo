# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SGLang scheduler batch fanout."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

_COMPONENTS_SRC = Path(__file__).resolve().parents[3]


def _load_batch_fanout():
    path = (
        _COMPONENTS_SRC
        / "dynamo"
        / "sglang"
        / "request_handlers"
        / "llm"
        / "batch_fanout.py"
    )
    spec = importlib.util.spec_from_file_location("test_sglang_batch_fanout_impl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fanout = _load_batch_fanout()


class FakeEvent:
    def __init__(self) -> None:
        self.set_count = 0

    def set(self) -> None:
        self.set_count += 1


class FakeTimeStats:
    def __init__(self) -> None:
        self.response_sent_to_client_time = 0.0

    def set_response_sent_to_client_time(self) -> None:
        self.response_sent_to_client_time = 1.0

    def get_response_sent_to_client_realtime(self) -> float:
        return 123.0


class FakeState:
    def __init__(self) -> None:
        self.event = FakeEvent()
        self.finished = False
        self.out_list = []
        self.time_stats = FakeTimeStats()


class FakeContext:
    trace_id = "trace"

    def __init__(self, stopped: bool = False) -> None:
        self.stopped = stopped

    def is_stopped(self) -> bool:
        return self.stopped


class FakeHandler:
    @staticmethod
    def _extract_logprobs(*args, **kwargs):
        return None, None


class FakeSender:
    batches: ClassVar[list[list[tuple[object, dict]]]] = []
    failed: ClassVar[list[int]] = []

    @staticmethod
    def send_batch(items):
        FakeSender.batches.append(items)
        return FakeSender.failed


class FakeServerArgs:
    batch_notify_size = 1


class FakeRecv:
    def __init__(self, rids, finished_reasons=None) -> None:
        self.rids = rids
        self.finished_reasons = finished_reasons or [None] * len(rids)


class FakeTokenizerManager:
    def __init__(self, rids) -> None:
        self.rid_to_state = {rid: FakeState() for rid in rids}
        self.server_args = FakeServerArgs()

    async def _handle_batch_output(self, recv_obj) -> None:
        for rid, finish_reason in zip(recv_obj.rids, recv_obj.finished_reasons):
            state = self.rid_to_state[rid]
            state.finished = finish_reason is not None
            state.out_list.append(
                {
                    "output_ids": [7],
                    "meta_info": {"id": rid, "finish_reason": finish_reason},
                }
            )
            state.event.set()


def _install_fake_tokenizer_hook() -> None:
    tokenizer_module = types.ModuleType("sglang.srt.managers.tokenizer_manager")
    tokenizer_module.TokenizerManager = FakeTokenizerManager
    fake_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.tokenizer_manager": tokenizer_module,
    }
    with patch.dict(sys.modules, fake_modules):
        fanout._install_tokenizer_batch_hook()


_install_fake_tokenizer_hook()


@pytest.fixture(autouse=True)
def reset_fanout_state():
    fanout._REGISTRY = fanout._FanoutRegistry()
    FakeSender.batches = []
    FakeSender.failed = []


def _register(rid: str, *, stopped: bool = False, armed: bool = True) -> None:
    assert fanout._REGISTRY.add(
        rid,
        fanout._FanoutEntry(
            handler=FakeHandler(),
            context=FakeContext(stopped),
            response_sender=FakeSender(),
            return_tokens_as_token_ids=False,
            armed=armed,
        ),
    )


@pytest.mark.asyncio
async def test_first_output_stays_on_request_task_to_arm_cancellation():
    _register("arming", armed=False)
    manager = FakeTokenizerManager(["arming"])
    event = manager.rid_to_state["arming"].event

    await manager._handle_batch_output(FakeRecv(["arming"]))

    assert FakeSender.batches == []
    assert event.set_count == 1
    assert len(manager.rid_to_state["arming"].out_list) == 1

    # Model the request task consuming its first output.
    manager.rid_to_state["arming"].out_list = []
    await manager._handle_batch_output(FakeRecv(["arming"]))

    assert len(FakeSender.batches) == 1
    assert event.set_count == 1


@pytest.mark.asyncio
async def test_one_native_call_keeps_one_response_per_request():
    _register("a")
    _register("b")
    manager = FakeTokenizerManager(["a", "b"])
    events = {rid: state.event for rid, state in manager.rid_to_state.items()}

    await manager._handle_batch_output(FakeRecv(["a", "b"]))

    assert len(FakeSender.batches) == 1
    assert len(FakeSender.batches[0]) == 2
    assert [response["token_ids"] for _, response in FakeSender.batches[0]] == [
        [7],
        [7],
    ]
    assert [events["a"].set_count, events["b"].set_count] == [0, 0]
    assert manager.rid_to_state["a"].out_list == []
    assert manager.rid_to_state["b"].out_list == []


@pytest.mark.asyncio
async def test_final_output_stays_on_normal_request_path():
    _register("final")
    manager = FakeTokenizerManager(["final"])
    event = manager.rid_to_state["final"].event

    await manager._handle_batch_output(FakeRecv(["final"], [{"type": "length"}]))

    assert FakeSender.batches == []
    assert event.set_count == 1
    assert len(manager.rid_to_state["final"].out_list) == 1


@pytest.mark.asyncio
async def test_failed_native_item_stays_on_normal_request_path():
    _register("failed")
    FakeSender.failed = [0]
    manager = FakeTokenizerManager(["failed"])
    event = manager.rid_to_state["failed"].event

    await manager._handle_batch_output(FakeRecv(["failed"]))

    assert len(FakeSender.batches) == 1
    assert event.set_count == 1
    assert len(manager.rid_to_state["failed"].out_list) == 1

    FakeSender.failed = []
    await manager._handle_batch_output(FakeRecv(["failed"]))

    assert len(FakeSender.batches) == 1
    assert event.set_count == 2


@pytest.mark.asyncio
async def test_mixed_batch_stays_fully_on_normal_path():
    _register("eligible")
    manager = FakeTokenizerManager(["eligible", "normal"])
    events = {rid: state.event for rid, state in manager.rid_to_state.items()}

    await manager._handle_batch_output(FakeRecv(["eligible", "normal"]))

    assert FakeSender.batches == []
    assert events["eligible"].set_count == 1
    assert events["normal"].set_count == 1


@pytest.mark.asyncio
async def test_stopped_request_is_not_sent_or_woken():
    _register("stopped", stopped=True)
    manager = FakeTokenizerManager(["stopped"])
    event = manager.rid_to_state["stopped"].event

    await manager._handle_batch_output(FakeRecv(["stopped"]))

    assert FakeSender.batches == []
    assert event.set_count == 0
    assert manager.rid_to_state["stopped"].out_list == []
