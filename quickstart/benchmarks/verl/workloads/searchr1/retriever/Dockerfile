# CPU-only Search-R1 retrieval server image (BM25 path) for the agentic searchr1 task.
#
# Serves deploy/retriever/retrieval_server.py (vendored from Search-R1
# @ e23b87911693e5b174fc4d2d6bfd21f99137b5fc) as a FastAPI POST /retrieve on :8000.
# BM25 (pyserini/Lucene) needs a JRE; the server also imports faiss/torch/transformers
# at module scope, so those are installed even though the BM25 path doesn't use a GPU.
#
# Build + push (mirrors the other IMG_* images -> ghcr.io/ezrasilvera).
# Context is the verl repo root ("." - the COPY path below is repo-root-relative), run from there:
#   docker build -f benchmarks/workloads/searchr1/retriever/Dockerfile -t ghcr.io/ezrasilvera/searchr1-retriever:bm25 .
#   docker push ghcr.io/ezrasilvera/searchr1-retriever:bm25
# Then set IMG_RETRIEVER in deploy/deploy.env to that ref.

FROM python:3.11-slim

# JRE for pyserini/Lucene (BM25) + tools for the HF index download.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# CPU torch first (avoid pulling the CUDA wheels), then the retrieval-server deps.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        faiss-cpu \
        transformers \
        datasets \
        numpy \
        tqdm \
        uvicorn \
        fastapi \
        pydantic \
        pyserini \
        "huggingface_hub[cli]"

WORKDIR /app
COPY ./benchmarks/workloads/searchr1/retriever/retrieval_server.py /app/retrieval_server.py

EXPOSE 8000
# Launch args are supplied by the k8s manifest (index/corpus paths on the mounted volume).
CMD ["python", "/app/retrieval_server.py", "--help"]
