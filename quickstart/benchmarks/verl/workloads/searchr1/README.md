# searchr1 workload

> Stub - to be populated later with the test description and a summary of main results.

## Test description

TODO: what this workload measures and why (prompt/decode profile, reward), and the config rationale.

## Results summary

TODO: headline native-vs-EPP numbers for this workload. Raw runs are kept out of the repo (not committed).

## Files

- `task.env` - verl overrides sourced by `run_test.sh --task searchr1`.
- Data builder: `make_searchr1.py`.
- `tool_config.yaml` - registers the `search` tool (resolves to `llm_d_rl_verl_integration.tools.search_tool.SearchTool`).
- `retriever/` - the BM25 retrieval side-service (Dockerfile, server, k8s manifest).
