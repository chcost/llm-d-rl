#!/usr/bin/env python3
"""
Registration shim: the only part of sglang_router that EPP + Envoy cannot
replace on their own.

Envoy (port 8081) is the single address slime points --sglang-router-ip/port at.
Envoy handles /generate directly via EPP ext_proc → ORIGINAL_DST → sglang engine,
with no extra hop. It routes all /workers* and legacy registration paths here
(port 3001, internal only).

This shim's only job is to:
  1. Maintain the in-memory worker registry.
  2. Atomically rewrite /tmp/epp-endpoints.yaml on every change so EPP's
     file-discovery plugin (watchFile: true) stays in sync.

Endpoints exposed (called by Envoy, never by slime directly):
  POST   /workers              register engine   (sglang_router ≥0.3.0)
  GET    /workers              list engines      (used by slime abort())
  DELETE /workers/{id}         deregister engine
  POST   /add_worker?url=…     register engine   (sglang_router ≤0.2.1 compat)
  POST   /remove_worker?url=…  deregister engine (sglang_router ≤0.2.1 compat)
  GET    /list_workers         list engines      (sglang_router ≤0.2.1 compat)
"""

import argparse
import logging
import os
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

_workers: dict[str, dict[str, Any]] = {}   # worker_id → {id, url, worker_type}
_endpoints_file: Path = Path("/tmp/epp-endpoints.yaml")


def _write_endpoints() -> None:
    """Atomically rewrite the EPP file-discovery endpoints YAML."""
    if not _workers:
        content = "endpoints: []\n"
    else:
        lines = ["endpoints:"]
        for i, w in enumerate(_workers.values()):
            parsed = urllib.parse.urlparse(w["url"])
            lines.append(f"  - name: sglang-{i}")
            lines.append(f"    address: {parsed.hostname}")
            lines.append(f"    port: '{parsed.port}'")
            lines.append(f"    rankIndex: {i}")
            lines.append(f"    labels:")
            lines.append(f"      llm-d.ai/engine-type: sglang")
        content = "\n".join(lines) + "\n"

    tmp = Path(str(_endpoints_file) + ".tmp")
    tmp.write_text(content)
    tmp.replace(_endpoints_file)
    logger.info("endpoints: %d worker(s) → %s", len(_workers), _endpoints_file)


# ---------------------------------------------------------------------------
# sglang_router ≥0.3.0 API
# ---------------------------------------------------------------------------

@app.post("/workers")
async def add_worker(request: Request):
    body = await request.json()
    url = body["url"]
    existing = next((w for w in _workers.values() if w["url"] == url), None)
    if existing:
        logger.info("already registered %s → %s", existing["id"], url)
        return {"id": existing["id"]}
    wid = uuid.uuid4().hex[:8]
    _workers[wid] = {"id": wid, "url": url, "worker_type": body.get("worker_type", "regular")}
    _write_endpoints()
    logger.info("registered %s → %s", wid, url)
    return {"id": wid}


@app.get("/workers")
def list_workers():
    return {"workers": list(_workers.values())}


@app.delete("/workers/{worker_id}")
def remove_worker(worker_id: str):
    removed = _workers.pop(worker_id, None)
    if removed:
        _write_endpoints()
        logger.info("deregistered %s → %s", worker_id, removed["url"])
        return {}
    raise HTTPException(status_code=404, detail={"status": "not_found"})


# ---------------------------------------------------------------------------
# sglang_router ≤0.2.1 compat
# ---------------------------------------------------------------------------

@app.post("/add_worker")
async def add_worker_legacy(request: Request):
    url = request.query_params.get("url", "")
    wid = uuid.uuid4().hex[:8]
    _workers[wid] = {"id": wid, "url": url, "worker_type": "regular"}
    _write_endpoints()
    logger.info("registered (legacy) %s → %s", wid, url)
    return {"id": wid}


@app.post("/remove_worker")
async def remove_worker_legacy(request: Request):
    url = request.query_params.get("url", "")
    to_remove = [wid for wid, w in _workers.items() if w["url"] == url]
    for wid in to_remove:
        del _workers[wid]
    if to_remove:
        _write_endpoints()
    return {}


@app.get("/list_workers")
def list_workers_legacy():
    return {"urls": [w["url"] for w in _workers.values()]}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Slime EPP registration shim")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: localhost — Envoy proxies to us)")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--endpoints-file", default="/tmp/epp-endpoints.yaml")
    args = parser.parse_args()

    global _endpoints_file
    _endpoints_file = Path(args.endpoints_file)
    _endpoints_file.parent.mkdir(parents=True, exist_ok=True)

    # Write empty file so EPP starts cleanly without a missing-file error
    _write_endpoints()
    logger.info("shim listening on %s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
