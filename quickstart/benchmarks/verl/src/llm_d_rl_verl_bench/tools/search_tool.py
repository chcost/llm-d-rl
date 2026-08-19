"""SearchTool: a verl BaseTool that queries a Search-R1 retrieval server.

Wired into verl's multi-turn ToolAgentLoop for agentic Search-R1 RL. Each tool
call POSTs the model's ``query_list`` to the retrieval server's ``/retrieve``
endpoint and returns the top-k passages as the tool-response text, formatted the
same way Search-R1 does (``Doc i(Title: T) body``). This only supplies the tool
observation each turn - routing and reqlog are unaffected, so the same
native/EPP comparison tooling applies per turn.

Registered from a tool_config.yaml via the dotted class_name
``llm_d_rl_verl_bench.tools.search_tool.SearchTool`` with

    config: {retrieval_service_url: "http://<svc>:8000/retrieve", topk: 3, type: native}

and a tool_schema whose function ``name`` is ``search`` (must match the dataset's
``extra_info.tools_kwargs["search"]`` so the per-sample create_kwargs are routed here).
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

logger = logging.getLogger(__name__)


def _doc_to_text(doc: Any) -> str:
    """Render one retrieved wiki-18 record as ``Title) body``.

    The Search-R1 retrieval server returns, per doc, either
      - {"title": T, "text": body, "contents": "T\\nbody"}  (BM25 with raw docs), or
      - {"contents": "T\\nbody"}                            (bare contents), or
      - {"document": <one of the above>, "score": s}        (return_scores=True wrapper).
    Prefer explicit title/text; otherwise split the title off the first line of contents.
    """
    if isinstance(doc, dict) and "document" in doc:
        doc = doc["document"]
    if not isinstance(doc, dict):
        return str(doc)
    title = doc.get("title")
    text = doc.get("text")
    if text is None:
        contents = doc.get("contents") or ""
        first, _, rest = contents.partition("\n")
        if title is None:
            title = first.strip().strip('"')
        text = rest
    return f"{title}) {text}".strip()


def _passages_to_text(per_query: list) -> str:
    """Format one query's ranked docs as Search-R1's numbered reference block."""
    lines = []
    for i, item in enumerate(per_query or []):
        lines.append(f"Doc {i + 1}(Title: {_doc_to_text(item)}")
    return "\n".join(lines)


class SearchTool(BaseTool):
    """Query a Search-R1 retrieval server and return passages as the tool response."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config, tool_schema)
        self._url: str = config["retrieval_service_url"]
        self._topk: int = int(config.get("topk", 3))
        self._timeout: float = float(config.get("timeout_s", 30))
        self._session: aiohttp.ClientSession | None = None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Default schema (function name ``search``) used when the YAML omits tool_schema."""
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="search",
                description=(
                    "Search a Wikipedia corpus and return the top-k passages for each query. "
                    "Use it whenever you lack the knowledge to answer."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query_list": OpenAIFunctionPropertySchema(
                            type="array",
                            description="One or more fully-formed search queries.",
                        )
                    },
                    required=["query_list"],
                ),
            ),
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def create(self, instance_id: str | None = None, create_kwargs: dict | None = None, **kwargs):
        # create_kwargs carries {ground_truth, question, data_source} per sample. The reward
        # reads ground_truth from the dataset row directly, so retrieval needs none of it;
        # accept and ignore it (kept for parity / future per-sample retrieval tuning).
        return (instance_id or uuid4().hex), ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        queries = parameters.get("query_list") or parameters.get("query") or []
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            return ToolResponse(text="No search query provided."), 0.0, {"num_queries": 0}

        # return_scores=True: the Search-R1 server's /retrieve unpacks (results, scores)
        # unconditionally, so return_scores=False raises server-side ("not enough values to
        # unpack"). We ignore the scores; _doc_to_text unwraps the {"document","score"} form.
        payload = {"queries": queries, "topk": self._topk, "return_scores": True}
        try:
            session = await self._get_session()
            async with session.post(self._url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:  # noqa: BLE001 - never crash the rollout on a retrieval hiccup
            logger.warning("SearchTool retrieval failed (%s): %s", self._url, e)
            return ToolResponse(text=f"Search error: {e}"), 0.0, {"num_queries": len(queries), "error": 1}

        results = data.get("result", []) if isinstance(data, dict) else []
        blocks = [_passages_to_text(per_query) for per_query in results]
        text = "\n".join(b for b in blocks if b) or "No results found."
        return ToolResponse(text=text), 0.0, {"num_queries": len(queries)}

    async def release(self, instance_id: str, **kwargs) -> None:
        # Session is shared across trajectories; closed at process exit, not per call.
        pass
