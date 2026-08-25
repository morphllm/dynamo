# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact CKF based global selection at the tokenizing frontend seam."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator

import httpx
from prometheus_client import Counter, Histogram

DECISIONS = Counter(
    "dynamo_global_router_multi_dc_decisions_total",
    "Aggregated multi data center routing decisions.",
    ("policy", "outcome", "target_dc"),
)
FAILURES = Counter(
    "dynamo_global_router_multi_dc_failures_total",
    "Aggregated multi data center routing failures.",
    ("stage", "reason"),
)
FACTS_SECONDS = Histogram(
    "dynamo_global_router_multi_dc_facts_seconds",
    "Time spent querying global CKF facts.",
)
REMOTE_SECONDS = Histogram(
    "dynamo_global_router_multi_dc_remote_seconds",
    "Remote request lifetime including the streamed body.",
    ("target_dc",),
)

_MARKER = "_dynamo_global_router"
_READY = 2
_U64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class MultiDcConfig:
    policy: str
    facts_url: str
    local_dc: int
    block_size: int
    gateways: dict[str, str]
    secret: bytes
    readiness_max_age_ms: int = 45_000
    load_max_age_ms: int = 15_000
    marker_max_age_ms: int = 30_000
    connect_timeout_s: float = 2.0
    facts_timeout_s: float = 1.0
    cert: tuple[str, str] | None = None
    verify: str | bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiDcConfig":
        policy = str(data.get("policy", "shadow"))
        if policy not in {"shadow", "enforce"}:
            raise ValueError("multi_dc.policy must be shadow or enforce")
        secret_file = str(data.get("hmac_secret_file", ""))
        secret = open(secret_file, "rb").read().strip() if secret_file else b""
        if len(secret) < 32:
            raise ValueError("multi_dc HMAC secret must contain at least 32 bytes")
        cert_file = data.get("client_cert_file")
        key_file = data.get("client_key_file")
        ca_file = data.get("ca_file")
        cert = (str(cert_file), str(key_file)) if cert_file and key_file else None
        gateways = {
            str(key): str(value) for key, value in data.get("gateways", {}).items()
        }
        if policy == "enforce":
            if cert is None or not ca_file:
                raise ValueError(
                    "multi_dc enforce requires client certificate, key, and CA"
                )
            if any(
                not endpoint.startswith("https://") for endpoint in gateways.values()
            ):
                raise ValueError("multi_dc enforce requires HTTPS gateways")
        return cls(
            policy=policy,
            facts_url=str(data["facts_url"]).rstrip("/"),
            local_dc=_u64(data["local_dc_id"]),
            block_size=int(data["block_size"]),
            gateways=gateways,
            secret=secret,
            readiness_max_age_ms=int(data.get("readiness_max_age_ms", 45_000)),
            load_max_age_ms=int(data.get("load_max_age_ms", 15_000)),
            marker_max_age_ms=int(data.get("marker_max_age_ms", 30_000)),
            connect_timeout_s=float(data.get("connect_timeout_seconds", 2)),
            facts_timeout_s=float(data.get("facts_timeout_seconds", 1)),
            cert=cert,
            verify=str(ca_file) if ca_file else True,
        )

    @classmethod
    def from_env(cls) -> "MultiDcConfig | None":
        policy = os.getenv("DYN_GLOBAL_ROUTER_POLICY", "off").lower()
        if policy == "off":
            return None
        return cls.from_dict(
            {
                "policy": policy,
                "facts_url": os.environ["DYN_GLOBAL_ROUTER_FACTS_URL"],
                "local_dc_id": os.environ["DYN_GLOBAL_ROUTER_LOCAL_DC_ID"],
                "block_size": os.environ["DYN_GLOBAL_ROUTER_BLOCK_SIZE"],
                "gateways": json.loads(
                    os.environ.get("DYN_GLOBAL_ROUTER_GATEWAYS_JSON", "{}")
                ),
                "hmac_secret_file": os.environ.get(
                    "DYN_GLOBAL_ROUTER_HMAC_SECRET_FILE", ""
                ),
                "client_cert_file": os.environ.get(
                    "DYN_GLOBAL_ROUTER_CLIENT_CERT_FILE"
                ),
                "client_key_file": os.environ.get("DYN_GLOBAL_ROUTER_CLIENT_KEY_FILE"),
                "ca_file": os.environ.get("DYN_GLOBAL_ROUTER_CA_FILE"),
                "readiness_max_age_ms": os.environ.get(
                    "DYN_GLOBAL_ROUTER_READINESS_MAX_AGE_MS", 45_000
                ),
                "load_max_age_ms": os.environ.get(
                    "DYN_GLOBAL_ROUTER_LOAD_MAX_AGE_MS", 15_000
                ),
                "marker_max_age_ms": os.environ.get(
                    "DYN_GLOBAL_ROUTER_MARKER_MAX_AGE_MS", 30_000
                ),
                "connect_timeout_seconds": os.environ.get(
                    "DYN_GLOBAL_ROUTER_CONNECT_TIMEOUT_SECONDS", 2
                ),
                "facts_timeout_seconds": os.environ.get(
                    "DYN_GLOBAL_ROUTER_FACTS_TIMEOUT_SECONDS", 1
                ),
            }
        )


@dataclass(frozen=True)
class PoolRoute:
    pool_id: str
    target_dc: int
    gateway: str
    total_prefill_tokens: int
    prefix_blocks: int


def select_pool(
    payload: dict[str, Any],
    *,
    local_dc: int,
    block_size: int,
    stable_tie_key: int,
    readiness_max_age_ms: int,
    load_max_age_ms: int,
    gateways: dict[str, str],
) -> PoolRoute | None:
    """Select the fresh ready pool with the least expected prefill work."""

    query_blocks = _u64(payload["query_blocks"])
    candidates: list[tuple[tuple[Any, ...], PoolRoute]] = []
    for pool in payload["pools"]:
        if pool.get("availability") != "available":
            continue
        readiness = [
            item
            for item in pool.get("readiness") or []
            if item.get("canonical_model_id") == payload.get("model")
        ]
        readiness_age = pool.get("readiness_age_ms")
        if (
            not readiness
            or readiness_age is None
            or _u64(readiness_age) > readiness_max_age_ms
            or not any(item.get("state") == _READY for item in readiness)
        ):
            continue
        active = pool.get("active_prefill_tokens")
        load_age = pool.get("load_age_ms")
        prefix = pool.get("prefix_depth_blocks")
        if (
            active is None
            or load_age is None
            or _u64(load_age) > load_max_age_ms
            or prefix is None
        ):
            continue
        prefix = _u64(prefix)
        if prefix > query_blocks:
            continue
        total = _checked_add(
            _checked_mul(query_blocks - prefix, block_size), _u64(active)
        )
        dc = _u64(pool["dc_id"])
        pool_id = str(pool["pool_id"])
        route = PoolRoute(
            pool_id, dc, gateways.get(pool_id, gateways.get(str(dc), "")), total, prefix
        )
        candidates.append(
            (
                (total, dc != local_dc, _stable_rank(stable_tie_key, pool), pool_id),
                route,
            )
        )
    if not candidates:
        return None
    route = min(candidates, key=lambda item: item[0])[1]
    if route.target_dc != local_dc and not route.gateway:
        raise ValueError(f"no gateway configured for selected pool {route.pool_id}")
    return route


class MultiDcRouter:
    def __init__(self, config: MultiDcConfig, client: httpx.AsyncClient | None = None):
        if config.block_size <= 0:
            raise ValueError("multi_dc block_size must be positive")
        self.config = config
        self.client = client or httpx.AsyncClient(
            http2=True,
            cert=config.cert,
            verify=config.verify,
            timeout=httpx.Timeout(
                connect=config.connect_timeout_s,
                read=None,
                write=config.connect_timeout_s,
                pool=config.connect_timeout_s,
            ),
        )

    @classmethod
    def from_env(cls, block_size: int) -> "MultiDcRouter | None":
        config = MultiDcConfig.from_env()
        if config is None:
            return None
        if config.block_size != block_size:
            raise ValueError(
                "DYN_GLOBAL_ROUTER_BLOCK_SIZE must match the frontend native block size"
            )
        return cls(config)

    def consume_private_marker(self, request: dict[str, Any]) -> bool:
        args = request.get("chat_template_args")
        marker = args.pop(_MARKER, None) if isinstance(args, dict) else None
        if marker is None:
            return False
        if not isinstance(marker, dict):
            raise ValueError("invalid global router marker")
        try:
            issued_ms = int(marker["issued_ms"])
            source_dc = int(marker["source_dc"])
            target_dc = int(marker["target_dc"])
            nonce = str(marker["nonce"])
            signature = str(marker["signature"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid global router marker") from error
        now_ms = time.time_ns() // 1_000_000
        if (
            target_dc != self.config.local_dc
            or not 0 <= now_ms - issued_ms <= self.config.marker_max_age_ms
        ):
            raise ValueError("expired or misdirected global router marker")
        expected = self._marker(request, target_dc, issued_ms, nonce, source_dc)[
            "signature"
        ]
        if not hmac.compare_digest(signature, expected):
            raise ValueError("unauthenticated global router marker")
        return True

    async def decide(
        self, model: str, token_ids: list[int], request_id: str, bypass: bool = False
    ) -> PoolRoute | None:
        if bypass or len(token_ids) < self.config.block_size:
            return None
        started = time.monotonic()
        try:
            response = await self.client.post(
                f"{self.config.facts_url}/v1/prefix-matches/tokens",
                json={
                    "model": model,
                    "role": "aggregated",
                    "token_ids": token_ids,
                    "block_size": self.config.block_size,
                },
                timeout=self.config.facts_timeout_s,
            )
            response.raise_for_status()
            route = select_pool(
                response.json(),
                local_dc=self.config.local_dc,
                block_size=self.config.block_size,
                stable_tie_key=int.from_bytes(
                    hashlib.sha256(request_id.encode()).digest()[:8], "big"
                ),
                readiness_max_age_ms=self.config.readiness_max_age_ms,
                load_max_age_ms=self.config.load_max_age_ms,
                gateways=self.config.gateways,
            )
        except Exception as error:
            FAILURES.labels("facts", type(error).__name__).inc()
            return None
        finally:
            FACTS_SECONDS.observe(time.monotonic() - started)
        outcome = (
            "remote"
            if route is not None and route.target_dc != self.config.local_dc
            else "local"
        )
        target = str(route.target_dc if route is not None else self.config.local_dc)
        DECISIONS.labels(self.config.policy, outcome, target).inc()
        return None if self.config.policy == "shadow" or outcome == "local" else route

    async def stream_remote(
        self, route: PoolRoute, openai_request: dict[str, Any], request_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        body = json.loads(json.dumps(openai_request))
        issued_ms = time.time_ns() // 1_000_000
        nonce = f"{request_id}:{secrets.token_hex(8)}"
        args = body.setdefault("chat_template_args", {})
        if not isinstance(args, dict):
            raise ValueError("chat_template_args must be an object")
        args[_MARKER] = self._marker(body, route.target_dc, issued_ms, nonce)
        response = None
        started = time.monotonic()
        try:
            request = self.client.build_request(
                "POST", f"{route.gateway.rstrip('/')}/v1/chat/completions", json=body
            )
            response = await self.client.send(request, stream=True)
            response.raise_for_status()
            async for item in _decode_sse(response.aiter_raw()):
                yield item
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            FAILURES.labels("remote", type(error).__name__).inc()
            raise
        finally:
            if response is not None:
                await response.aclose()
            REMOTE_SECONDS.labels(str(route.target_dc)).observe(
                time.monotonic() - started
            )

    def _marker(
        self,
        body: dict[str, Any],
        target_dc: int,
        issued_ms: int,
        nonce: str,
        source_dc: int | None = None,
    ) -> dict[str, Any]:
        unsigned = json.loads(json.dumps(body))
        unsigned.get("chat_template_args", {}).pop(_MARKER, None)
        digest = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_dc = self.config.local_dc if source_dc is None else source_dc
        message = (
            f"1\n{source_dc}\n{target_dc}\n{issued_ms}\n{nonce}\n{digest}".encode()
        )
        return {
            "version": 1,
            "source_dc": source_dc,
            "target_dc": target_dc,
            "issued_ms": issued_ms,
            "nonce": nonce,
            "signature": hmac.new(
                self.config.secret, message, hashlib.sha256
            ).hexdigest(),
        }


async def _decode_sse(
    chunks: AsyncIterator[bytes],
) -> AsyncGenerator[dict[str, Any], None]:
    buffered = b""
    async for chunk in chunks:
        buffered += chunk
        while b"\n\n" in buffered:
            frame, buffered = buffered.split(b"\n\n", 1)
            for line in frame.splitlines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                if data:
                    yield {"data": json.loads(data)}


def _stable_rank(key: int, pool: dict[str, Any]) -> int:
    value = f"{key}:{pool['pool_id']}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _u64(value: Any) -> int:
    parsed = int(value)
    if not 0 <= parsed <= _U64_MAX:
        raise ValueError("value is outside u64")
    return parsed


def _checked_add(left: int, right: int) -> int:
    return _u64(left + right)


def _checked_mul(left: int, right: int) -> int:
    return _u64(left * right)
