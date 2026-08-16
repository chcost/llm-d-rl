"""EPP registration shim.

FastAPI server that accepts vllm-router / sglang-router worker register/deregister
calls (POST/DELETE /workers) and writes the YAML file EPP's file-discovery
plugin watches. EPP does not speak that HTTP API; verl writes the file from
the trainer instead.

Start with:
    llm-d-registration-shim --engine-type vllm               # URL-keyed (vllm-router / vime)
    llm-d-registration-shim --engine-type sglang --id-field id  # UUID-keyed (sglang-router / slime)

Endpoints:
    POST   /workers            register engine   body: {url, worker_type?}
    GET    /workers            list engines
    DELETE /workers/{ref:path} deregister engine  ref = id_field value returned by POST
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from llm_d_rl_common.endpoints import write_rollout_endpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RegistrationShim:
    """Framework-agnostic EPP registration shim.
    Accepts a custom engine_type and id_field per integration.

    id_field="url"  → URL-keyed protocol (vllm-router / vime)
    id_field="id"   → UUID-keyed protocol (sglang-router / slime)
    """

    def __init__(self, engine_type: str, id_field: str = "url") -> None:
        self._engine_type = engine_type
        self._id_field = id_field
        self._endpoints_file: Path = Path("/tmp/epp-endpoints.yaml")
        self._workers: dict[str, dict] = {}  # key -> entry dict
        self.app: FastAPI = self._make_app()

    def _make_key(self, url: str) -> str:
        return url if self._id_field == "url" else uuid.uuid4().hex[:8]

    def _add_response(self, key: str, url: str) -> dict:
        return {self._id_field: key}

    def _entry(self, key: str, url: str, worker_type: str) -> dict:
        d = {"url": url, "worker_type": worker_type}
        d[self._id_field] = key  # no-op when id_field=="url" (same value already set)
        return d

    def _urls(self) -> list[str]:
        return [w["url"] for w in self._workers.values()]

    def _write_endpoints(self) -> None:
        write_rollout_endpoints(
            str(self._endpoints_file),
            self._urls(),
            engine_type=self._engine_type,
        )
        logger.info("endpoints: %d worker(s) -> %s", len(self._workers), self._endpoints_file)

    def _register(self, url: str, worker_type: str) -> dict:
        """Register a worker; idempotent on duplicate URL."""
        for key, w in self._workers.items():
            if w["url"] == url:
                logger.info("already registered url=%s", url)
                return self._add_response(key, url)
        key = self._make_key(url)
        self._workers[key] = self._entry(key, url, worker_type)
        self._write_endpoints()
        logger.info("registered key=%s url=%s", key, url)
        return self._add_response(key, url)

    def _deregister(self, ref: str) -> bool:
        """Remove by key. Returns True if found."""
        # vllm-router sends DELETE /workers/http%3A/host%3Aport — FastAPI decodes
        # %3A→: but leaves / as-is, yielding http:/host:port (single slash).
        # Normalize to http://host:port so it matches the stored key.
        if ref.startswith("http:/") and not ref.startswith("http://"):
            ref = "http://" + ref[6:]
        elif ref.startswith("https:/") and not ref.startswith("https://"):
            ref = "https://" + ref[7:]
        removed = self._workers.pop(ref, None)
        if removed:
            self._write_endpoints()
            logger.info("deregistered %s", ref)
        return removed is not None

    def _make_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/workers")
        def list_workers():
            return {"workers": list(self._workers.values())}

        @app.post("/workers")
        async def add_worker(request: Request):
            body = await request.json()
            return self._register(body["url"], body.get("worker_type", "regular"))

        # `:path` matches plain UUIDs (sglang) and percent-encoded URLs (vllm) alike.
        @app.delete("/workers/{ref:path}")
        def remove_worker(ref: str):
            if self._deregister(ref):
                return {}
            raise HTTPException(status_code=404, detail={"status": "not_found"})

        return app

    def run(self, host: str, port: int, endpoints_file: str) -> None:
        self._endpoints_file = Path(endpoints_file)
        self._endpoints_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_endpoints()
        logger.info("shim listening on %s:%d (engine=%s id_field=%s)",
                    host, port, self._engine_type, self._id_field)
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(description="EPP registration shim")
    parser.add_argument("--engine-type", required=True,
                        help="Engine type written to endpoints file (e.g. vllm, sglang)")
    parser.add_argument("--id-field", default="url",
                        help="Response field for the worker handle: 'url' for URL-keyed "
                             "(vllm-router), 'id' for UUID-keyed (sglang-router)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: localhost — Envoy proxies to us)")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--endpoints-file", default="/tmp/epp-endpoints.yaml")
    args = parser.parse_args()

    RegistrationShim(engine_type=args.engine_type, id_field=args.id_field).run(
        host=args.host,
        port=args.port,
        endpoints_file=args.endpoints_file,
    )
