# enterprise-multi-agent-rag

`enterprise-multi-agent-rag` is a portfolio project for a production-oriented,
multi-agent Retrieval-Augmented Generation platform. The target platform will
coordinate specialized agents with LangGraph, retrieve enterprise knowledge,
support multiple model providers, expose a FastAPI API and MCP server, and run
in containerized AWS infrastructure.

The current foundation establishes package boundaries, configuration, logging,
a health endpoint, test infrastructure, local document loading, and document
chunking on Python 3.11.

## Architecture

```text
Clients
  |
  v
FastAPI API ---- MCP server
  |
  v
LangGraph workflow
  |
  +-- Agents
  +-- Tools
  +-- Retrieval ---- Document ingestion / FAISS
  +-- LLM providers ---- OpenAI / Anthropic / Amazon Bedrock
  |
  v
Evaluation and observability
```

The source package is organized by responsibility:

- `api`: HTTP application and route definitions
- `ingestion`: document loading and indexing boundary
- `retrieval`: retrieval boundary
- `llm/providers`: model-provider integrations
- `agents`: LangGraph agent definitions
- `graph`: graph workflow assembly
- `evaluation`: quality and regression evaluation
- `tools`: agent tools
- `mcp`: MCP server boundary
- `core`: configuration and logging

## Local setup

Create and activate a Python 3.11 virtual environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and populate only the credentials needed for your
environment. Never commit `.env`.

Run the API:

```bash
uvicorn enterprise_multi_agent_rag.main:app --reload
```

Check health at `GET http://127.0.0.1:8000/health`.

Run tests:

```bash
pytest
```

## Document ingestion

The ingestion layer validates a local file and converts it to one consistent
`IngestedDocument` model. It supports PDF (`.pdf`), UTF-8 plain text (`.txt`),
and UTF-8 Markdown (`.md`). Document IDs are deterministic SHA-256 hashes of
the original file bytes.

```python
from enterprise_multi_agent_rag.ingestion import DocumentLoader

document = DocumentLoader().load("documents/handbook.pdf")
print(document.document_id, document.metadata["page_count"])
```

Document ingestion does not perform OCR on scanned or image-only PDFs or
preserve PDF layout.

## Document chunking

Chunking divides an `IngestedDocument` into smaller `DocumentChunk` objects
that retain their relationship to the source. Overlap repeats a small amount of
neighboring text so that context near a chunk boundary is not isolated. The
default configuration is 1,000 characters per chunk with 200 characters of
overlap.

```python
from enterprise_multi_agent_rag.chunking import DocumentChunker
from enterprise_multi_agent_rag.ingestion import DocumentLoader

document = DocumentLoader().load("documents/handbook.md")
chunks = DocumentChunker(chunk_size=1000, chunk_overlap=200).chunk(document)
print(chunks[0].chunk_id, chunks[0].metadata["total_chunks"])
```

Chunk content is not summarized or normalized. Character offsets use exact,
forward, overlap-aware substring matching against the original text. An offset
is `None` if an exact location cannot be established safely. This layer does
not provide token-aware splitting, semantic splitting, embeddings, persistence,
vector storage, retrieval, uploads, chains, graphs, or agents.

## Planned technology

LangChain, LangGraph, OpenAI, Anthropic Claude, Amazon Bedrock, FastAPI, FAISS,
MCP, Docker, GitHub Actions, and AWS.
