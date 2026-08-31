"""Minimal EPP gRPC ext-proc client.

Sends token IDs to EPP and reads back the chosen endpoint
(x-gateway-destination-endpoint) plus any sidecar headers.
Hand-rolled protobuf — no generated stubs needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

import grpc.aio

logger = logging.getLogger(__name__)

_EXT_PROC_METHOD = "/envoy.service.ext_proc.v3.ExternalProcessor/Process"
DESTINATION_HEADER = "x-gateway-destination-endpoint"


# ---------------------------------------------------------------------------
# Minimal protobuf encoder
# ---------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    buf = []
    while n > 0x7F:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n & 0x7F)
    return bytes(buf)


def _lv(field: int, data: bytes) -> bytes:
    tag = _varint((field << 3) | 2)
    return tag + _varint(len(data)) + data


def _bool_field(field: int, value: bool) -> bytes:
    return _varint((field << 3) | 0) + bytes([1 if value else 0])


def _encode_header_map(headers: list[tuple[str, bytes]]) -> bytes:
    out = b""
    for k, v in headers:
        hv = _lv(1, k.encode()) + _lv(3, v)
        out += _lv(1, hv)
    return out


def _encode_request_headers(headers: list[tuple[str, bytes]]) -> bytes:
    http_headers = _lv(1, _encode_header_map(headers)) + _bool_field(3, False)
    return _lv(2, http_headers)


def _encode_request_body(body: bytes) -> bytes:
    http_body = _lv(1, body) + _bool_field(2, True)
    return _lv(4, http_body)


def _encode_response_headers(headers: list[tuple[str, bytes]], end_of_stream: bool = False) -> bytes:
    """ProcessingRequest.response_headers (field 3), an HttpHeaders message."""
    http_headers = _lv(1, _encode_header_map(headers)) + _bool_field(3, end_of_stream)
    return _lv(3, http_headers)


def _encode_response_body(body: bytes, end_of_stream: bool) -> bytes:
    """ProcessingRequest.response_body (field 5), an HttpBody message."""
    http_body = _lv(1, body) + _bool_field(2, end_of_stream)
    return _lv(5, http_body)


# ---------------------------------------------------------------------------
# Minimal protobuf decoder
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _decode_fields(data: bytes) -> dict[int, list[bytes]]:
    fields: dict[int, list[bytes]] = {}
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        wire_type = tag & 0x7
        field_number = tag >> 3
        if wire_type == 0:
            _, pos = _decode_varint(data, pos)
        elif wire_type == 2:
            length, pos = _decode_varint(data, pos)
            fields.setdefault(field_number, []).append(data[pos: pos + length])
            pos += length
        elif wire_type == 5:
            pos += 4
        elif wire_type == 1:
            pos += 8
        else:
            break
    return fields


def _extract_headers(response_bytes: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    top = _decode_fields(response_bytes)
    for hr_bytes in top.get(1, []):
        for cr_bytes in _decode_fields(hr_bytes).get(1, []):
            for hm_bytes in _decode_fields(cr_bytes).get(2, []):
                for hvo_bytes in _decode_fields(hm_bytes).get(1, []):
                    for hv_bytes in _decode_fields(hvo_bytes).get(1, []):
                        hv = _decode_fields(hv_bytes)
                        key = (hv.get(1, [b""])[0]).decode(errors="ignore")
                        val = (hv.get(3, [b""])[0]).decode(errors="ignore")
                        if key:
                            result[key] = val
    return result


# ---------------------------------------------------------------------------
# EPP gRPC client
# ---------------------------------------------------------------------------

class _RequestStream:
    """A single open ext_proc Process stream for one request, kept open past the
    routing decision so a response phase can be sent later via ``complete()``."""

    __slots__ = ("call", "send_q", "endpoint", "sidecar")

    def __init__(self, call, send_q: "asyncio.Queue", endpoint: Optional[str], sidecar: dict[str, str]):
        self.call = call
        self.send_q = send_q
        self.endpoint = endpoint
        self.sidecar = sidecar


class RoutingResult:
    """Outcome of routing a request through EPP via ``EPPGrpcClient.route()``.

    ``complete()`` is unconditionally safe to call once generation finishes: it
    is a no-op when the request was routed without completion tracking, so
    callers never need to know whether ``pick()`` or ``begin()``/``complete()``
    was used underneath.
    """

    __slots__ = ("endpoint", "sidecar_headers", "_complete_fn")

    def __init__(self, endpoint: Optional[str], sidecar_headers: dict[str, str], complete_fn) -> None:
        self.endpoint = endpoint
        self.sidecar_headers = sidecar_headers
        self._complete_fn = complete_fn

    async def complete(self, output_tokens: int = 0) -> None:
        await self._complete_fn(output_tokens)


class EPPGrpcClient:
    """Thin gRPC client for EPP's ext-proc endpoint.

    Creates a persistent channel per instance. Each AgentLoopWorker
    should create its own instance (lazy, after unpickling).
    """

    def __init__(self, grpc_addr: str) -> None:
        self._addr = grpc_addr
        self._channel = grpc.aio.insecure_channel(grpc_addr)
        self._method = self._channel.stream_stream(
            _EXT_PROC_METHOD,
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )

    async def route(
        self,
        model: str,
        prompt_ids: list[int],
        request_id: str,
        *,
        track_completion: bool = False,
        cache_salt: Optional[str] = None,
    ) -> RoutingResult:
        """Route a request through EPP and return the decision plus a completion hook.

        Chooses between the fire-and-forget (``pick()``) and tracked-completion
        (``begin()``/``complete()``) gRPC patterns based on ``track_completion``,
        so callers never need to know either protocol exists. Call
        ``result.complete(ntok)`` unconditionally when generation finishes; it
        does the real report to EPP in tracked mode and nothing otherwise.

        ``cache_salt``: forwarded to EPP, which folds it into the prefix-hash seed
        (see EPP's own ``prefixhash`` package). EPP's block hashing only ever looks
        at token IDs -- never multimodal content -- so requests whose TEXT tokens
        are identical (e.g. a fixed-template multimodal prompt where only the image
        differs) hash identically and are treated as one prefix-sharing group
        regardless of image content. Pass a value that is identical within a real
        GRPO group and distinct across groups (e.g. ``sample_index``) to restore
        correct group-boundary separation for that case; omit it for workloads
        where distinct text already implies distinct groups (the common case).
        """
        if track_completion:
            stream = await self.begin(model, prompt_ids, request_id, cache_salt=cache_salt)

            async def _complete(output_tokens: int = 0) -> None:
                await self.complete(stream, output_tokens)

            return RoutingResult(stream.endpoint, stream.sidecar, _complete)

        endpoint, sidecar_headers = await self.pick(model, prompt_ids, cache_salt=cache_salt)

        async def _noop(output_tokens: int = 0) -> None:
            return None

        return RoutingResult(endpoint, sidecar_headers, _noop)

    async def pick(
        self, model: str, prompt_ids: list[int], *, cache_salt: Optional[str] = None
    ) -> tuple[Optional[str], dict[str, str]]:
        """Ask EPP which endpoint to route this request to.

        Returns:
            (endpoint, sidecar_headers) where endpoint is ``host:port`` or None.
            sidecar_headers contains all EPP-set headers except the destination header.
        """
        payload = {"model": model, "token_ids": prompt_ids}
        if cache_salt:
            payload["cache_salt"] = cache_salt
        body = json.dumps(payload).encode()
        req_headers = _encode_request_headers([
            (":method", b"POST"),
            (":path", b"/inference/v1/generate"),
            ("content-type", b"application/json"),
            ("content-length", str(len(body)).encode()),
        ])
        req_body = _encode_request_body(body)

        async def _iter():
            yield req_headers
            yield req_body

        async for response_bytes in self._method(_iter()):
            headers = _extract_headers(response_bytes)
            endpoint = headers.get(DESTINATION_HEADER)
            if endpoint:
                sidecar_headers = {k: v for k, v in headers.items() if k != DESTINATION_HEADER}
                return endpoint, sidecar_headers

        return None, {}

    async def begin(
        self, model: str, prompt_ids: list[int], request_id: str, *, cache_salt: Optional[str] = None
    ) -> _RequestStream:
        """Like ``pick()``, but leaves the stream's send side open after the
        routing decision so ``complete()`` can later report the response phase.

        Use this (with ``complete()``) instead of ``pick()`` when EPP's in-flight
        counter needs to reflect the real generation window (e.g. an
        active-request-scorer or a per-endpoint concurrency cap) rather than the
        pick-time snapshot ``pick()`` produces.
        """
        payload = {"model": model, "token_ids": prompt_ids}
        if cache_salt:
            payload["cache_salt"] = cache_salt
        body = json.dumps(payload).encode()
        req_headers = _encode_request_headers([
            (":method", b"POST"),
            (":path", b"/inference/v1/generate"),
            ("content-type", b"application/json"),
            ("content-length", str(len(body)).encode()),
            # Stable key EPP uses to correlate the completion signal (PluginState).
            ("x-request-id", request_id.encode()),
        ])
        req_body = _encode_request_body(body)

        send_q: asyncio.Queue = asyncio.Queue()

        async def _req_iter():
            while True:
                item = await send_q.get()
                if item is None:  # sentinel -> half-close the send side
                    return
                yield item

        call = self._method(_req_iter())
        await send_q.put(req_headers)
        await send_q.put(req_body)

        # Read responses until we see the routing decision, then STOP (leave the
        # stream open). Use call.read() so we can resume reading in complete().
        endpoint: Optional[str] = None
        sidecar: dict[str, str] = {}
        while True:
            msg = await call.read()
            if msg is grpc.aio.EOF:
                break
            headers = _extract_headers(msg)
            ep = headers.get(DESTINATION_HEADER)
            if ep:
                endpoint = ep
                sidecar = {k: v for k, v in headers.items() if k != DESTINATION_HEADER}
                break
        return _RequestStream(call, send_q, endpoint, sidecar)

    async def complete(self, stream: _RequestStream, output_tokens: int = 0) -> None:
        """Send the response phase so EPP fires ResponseBodyProcessor -> decrement.

        Best-effort: any error here (or a dropped stream) still triggers EPP's
        completion defer on the server side, so the in-flight count is released.
        """
        try:
            usage = json.dumps({"usage": {"completion_tokens": int(output_tokens)}}).encode()
            await stream.send_q.put(_encode_response_headers([
                (":status", b"200"),
                ("content-type", b"application/json"),
            ]))
            await stream.send_q.put(_encode_response_body(usage, True))  # EndOfStream
            await stream.send_q.put(None)  # half-close send side -> server sees end
            # Drain remaining server messages so the RPC finishes cleanly.
            while True:
                msg = await stream.call.read()
                if msg is grpc.aio.EOF:
                    break
        except Exception:  # noqa: BLE001 - completion is best-effort; janitor backstops
            try:
                stream.call.cancel()
            except Exception:  # noqa: BLE001
                pass

    async def close(self) -> None:
        await self._channel.close()
