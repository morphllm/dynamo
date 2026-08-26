# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path
from types import ModuleType

import pytest


class _Metric:
    def labels(self, *args):
        return self

    def inc(self):
        pass

    def observe(self, value):
        pass


fake_prometheus = ModuleType("prometheus_client")
fake_prometheus.Counter = lambda *args, **kwargs: _Metric()
fake_prometheus.Histogram = lambda *args, **kwargs: _Metric()
sys.modules.setdefault("prometheus_client", fake_prometheus)

from dynamo.frontend.global_routing import (  # noqa: E402
    MultiDcConfig,
    MultiDcRouter,
    PoolRoute,
    select_pool,
)
from dynamo.frontend import global_routing as router_module  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def fact(dc, prefix, used, *, capacity=100, ready=True, readiness_age=1, load_age=1, suffix=""):
    return {
        "pool_id": f"pool-{dc}{suffix}",
        "indexer_domain": {},
        "dc_id": dc,
        "availability": "available",
        "prefix_depth_blocks": prefix,
        "readiness": [{"canonical_model_id": "model", "state": 2 if ready else 1}],
        "readiness_age_ms": readiness_age,
        "kv_used_blocks": used,
        "total_kv_blocks": capacity,
        "kv_observed_ranks": 1,
        "kv_expected_ranks": 1,
        "load_age_ms": load_age,
    }


def select(pools, **overrides):
    args = {
        "local_dc": 1,
        "block_size": 256,
        "stable_tie_key": 42,
        "readiness_max_age_ms": 5,
        "load_max_age_ms": 5,
        "gateways": {"2": "https://dc2", "3": "https://dc3"},
    }
    args.update(overrides)
    return select_pool({"model": "model", "query_blocks": 10, "pools": pools}, **args)


def test_exact_prefix_precedes_authoritative_kv_occupancy():
    assert select([fact(1, 9, 0), fact(2, 10, 99)]).target_dc == 2
    assert select([fact(1, 10, 60), fact(2, 9, 0)]).target_dc == 1


def test_equal_prefix_uses_normalized_kv_occupancy():
    assert select([fact(1, 10, 60), fact(2, 10, 100, capacity=200)]).target_dc == 2


def test_stale_missing_and_unready_facts_fail_closed():
    missing = fact(1, 10, 0)
    missing["kv_used_blocks"] = None
    assert select([missing, fact(2, 10, 0, load_age=6)]) is None
    assert (
        select([fact(1, 10, 0, readiness_age=6), fact(2, 10, 0, ready=False)]) is None
    )


def test_local_wins_equal_work_and_remote_requires_gateway():
    assert select([fact(2, 10, 50), fact(1, 10, 50)]).target_dc == 1
    with pytest.raises(ValueError, match="no gateway"):
        select([fact(2, 10, 0)], gateways={})


class Response:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        pass

    async def aiter_raw(self):
        for chunk in self.chunks:
            yield chunk
        await asyncio.Future()

    async def aclose(self):
        self.closed = True


class Client:
    def __init__(self, response):
        self.response = response
        self.sends = 0

    def build_request(self, *args, **kwargs):
        return object()

    async def send(self, request, *, stream):
        self.sends += 1
        return self.response


def config():
    return MultiDcConfig(
        policy="enforce",
        facts_url="https://facts",
        local_dc=1,
        block_size=256,
        gateways={"2": "https://dc2"},
        secret=b"s" * 32,
    )


@pytest.mark.asyncio
async def test_caller_cancellation_closes_unbounded_remote_stream():
    response = Response([])
    router = MultiDcRouter(config(), client=Client(response))
    stream = router.stream_remote(
        PoolRoute("pool-2", 2, "https://dc2", 0, 10), {"stream": True}, "r"
    )
    task = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.closed


@pytest.mark.asyncio
async def test_committed_stream_is_not_retried():
    response = Response([b'data: {"ok":true}\n\n'])
    client = Client(response)
    router = MultiDcRouter(config(), client=client)
    stream = router.stream_remote(
        PoolRoute("pool-2", 2, "https://dc2", 0, 10), {"stream": True}, "r"
    )
    assert await stream.__anext__() == {"data": {"ok": True}}
    await stream.aclose()
    assert client.sends == 1
    assert response.closed


def test_recursion_marker_is_authenticated():
    router = MultiDcRouter(config(), client=object())
    body = {"model": "model", "chat_template_args": {}}
    issued_ms = router_module.time.time_ns() // 1_000_000
    marker = router._marker(body, 2, issued_ms, "nonce")
    assert marker["source_dc"] == 1
    assert marker["target_dc"] == 2
    assert len(marker["signature"]) == 64

    remote_config = config()
    remote_config = MultiDcConfig(**{**remote_config.__dict__, "local_dc": 2})
    remote = MultiDcRouter(remote_config, client=object())
    body["chat_template_args"]["_dynamo_global_router"] = marker
    assert remote.consume_private_marker(body)
    assert body["chat_template_args"] == {}


def test_enforce_configuration_requires_mtls(tmp_path):
    secret = tmp_path / "secret"
    secret.write_bytes(b"s" * 32)
    with pytest.raises(ValueError, match="certificate"):
        MultiDcConfig.from_dict(
            {
                "policy": "enforce",
                "facts_url": "https://facts",
                "local_dc_id": 1,
                "block_size": 256,
                "gateways": {"2": "https://dc2"},
                "hmac_secret_file": str(secret),
            }
        )


def test_vllm_hook_routes_only_after_native_tokenization():
    source = (Path(__file__).parents[1] / "vllm_processor.py").read_text()
    marker = source.index("consume_private_marker(request)")
    preprocessing = source.index("pre = await preprocess_chat_request(")
    native_tokens = source.index("tokens = pre.prompt_token_ids")
    decision = source.index("remote_route = await self.global_router.decide(")

    assert marker < preprocessing < native_tokens < decision
    assert 'request["openai_request"]' not in source
