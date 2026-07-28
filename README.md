# enterprise-multi-agent-rag

`enterprise-multi-agent-rag` is a portfolio project for a production-oriented,
multi-agent Retrieval-Augmented Generation platform. The target platform will
coordinate specialized agents with LangGraph, retrieve enterprise knowledge,
support multiple model providers, expose a FastAPI API and MCP server, and run
in containerized AWS infrastructure.

This initial foundation intentionally contains no RAG, agent, provider, or MCP
behavior. It establishes package boundaries, configuration, logging, a health
endpoint, and test infrastructure on Python 3.11.

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

## Planned technology

LangChain, LangGraph, OpenAI, Anthropic Claude, Amazon Bedrock, FastAPI, FAISS,
MCP, Docker, GitHub Actions, and AWS.

