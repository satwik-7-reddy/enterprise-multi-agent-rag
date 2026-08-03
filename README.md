# enterprise-multi-agent-rag

`enterprise-multi-agent-rag` is a portfolio project for a production-oriented,
multi-agent Retrieval-Augmented Generation platform. The target platform will
coordinate specialized agents with LangGraph, retrieve enterprise knowledge,
support multiple model providers, expose a FastAPI API and MCP server, and run
in containerized AWS infrastructure.

The current foundation establishes package boundaries, configuration, logging,
a health endpoint, test infrastructure, document processing, embedding
generation, FAISS storage, and question retrieval on Python 3.11.

## Architecture

```text
Document ingestion flow:

Document
  ↓
Chunking
  ↓
EmbeddingService
  ↓
EmbeddedChunk[]
  ↓
FAISSVectorStore

Question answering flow:

User question
  ↓
RAGService
  ↓
Retriever
  ↓
PromptBuilder
  ↓
LLMService
  ↓
Generated Answer
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
uvicorn enterprise_multi_agent_rag.main:app --app-dir src --reload
```

Check health at `GET http://127.0.0.1:8000/health`.

### Query API and Swagger UI

Open `http://localhost:8000/docs` after starting the API. Expand `POST /query`,
select **Try it out**, and submit a request such as:

```json
{
  "question": "How many vacation days do employees receive?",
  "k": 5
}
```

The response contains the original question and generated answer:

```json
{
  "question": "How many vacation days do employees receive?",
  "answer": "Employees receive 20 vacation days."
}
```

Run tests:

```bash
pytest
```

## Building the Knowledge Base

`DocumentIndexingService` connects the existing loader, chunker, embedding
service, and FAISS store. After constructing it with those components, index
one document:

```python
indexing_service.index_document("documents/employee_handbook.pdf")
```

Or index several documents into the same store:

```python
indexing_service.index_documents([
    "documents/handbook.pdf",
    "documents/hr_policy.md",
])
```

Both workflows create the searchable knowledge base in `vector_store/`:

```text
vector_store/
    index.faiss
    chunks.json
```

These files must exist before the query API can search indexed documents.

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
is `None` if an exact location cannot be established safely. The chunking layer
does not provide token-aware or semantic splitting and does not alter the
source content.

## Embeddings

Embeddings convert chunk text into numeric vectors for future similarity
search. The embedding layer is provider-independent: `EmbeddingService`
coordinates chunks through a `BaseEmbeddingProvider`, while provider classes
own their native request and response formats. AWS Bedrock is the default,
using `amazon.titan-embed-text-v2:0` with 1,024 normalized dimensions.

Bedrock credentials are not stored or passed by this application. Boto3
discovers them through its standard credential provider chain, including
environment variables, shared AWS configuration, container roles, and EC2
instance roles. Select a provider in `.env`:

```dotenv
EMBEDDING_PROVIDER=bedrock
# or:
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=replace-with-a-local-secret
```

```python
from enterprise_multi_agent_rag.chunking import DocumentChunker
from enterprise_multi_agent_rag.core.config import get_settings
from enterprise_multi_agent_rag.embeddings import (
    EmbeddingService,
    create_embedding_provider,
)
from enterprise_multi_agent_rag.ingestion import DocumentLoader

document = DocumentLoader().load("documents/handbook.md")
chunks = DocumentChunker().chunk(document)
provider = create_embedding_provider(get_settings())
embedded_chunks = EmbeddingService(provider).embed_chunks(chunks)
```

Embeddings from different providers, models, dimensions, or normalization
settings must not be mixed in the same FAISS index. Native Titan V2
requests accept one text input, so Bedrock currently makes one request per
chunk; OpenAI uses one native multi-input request.

## FAISS vector storage

FAISS is a library for efficient similarity search over numeric vectors. This
project uses `faiss.IndexFlatIP`, an exact, non-approximate index that ranks
vectors by inner product. When document and query embeddings are normalized,
inner product is equivalent to cosine similarity. The vector store deliberately
does not normalize vectors itself.

`FAISSVectorStore.add_chunks()` converts embeddings to NumPy `float32` rows and
adds them to FAISS. A parallel in-memory list keeps each FAISS row position
mapped to its complete `EmbeddedChunk`. Searching converts the query to the
same representation, asks FAISS for the top `k` row positions, and returns the
mapped chunks with their scores and one-based ranks.

```python
from enterprise_multi_agent_rag.retrieval import FAISSVectorStore

store = FAISSVectorStore()
store.add_chunks(embedded_chunks)
results = store.search(query_embedding, k=5)

store.save("vector_store")
restored_store = FAISSVectorStore.load("vector_store")
```

Saving writes the native index to `vector_store/index.faiss` and the ordered
chunk mapping to `vector_store/chunks.json`. Loading reconstructs both without
regenerating embeddings and validates their row counts and dimensions. Document
and query vectors must come from the same embedding model with the same
dimension and normalization settings; otherwise their similarity scores are
not meaningful.

## Retriever

`Retriever` accepts a natural-language question, embeds it once with the
configured embedding provider, searches `FAISSVectorStore`, and returns the
top-k `SearchResult` objects in ranking order.

```python
from enterprise_multi_agent_rag.retrieval import Retriever

retriever = Retriever(provider, restored_store)
results = retriever.retrieve("How many vacation days do employees get?", k=5)
```

The retriever coordinates embedding and search only; it does not generate a
final answer, rewrite the query, rerank results, or format citations. Stored
document chunks and questions must use the same embedding model, dimensions,
and normalization configuration.

## Prompt Builder

The user provides only a natural-language question, while the retriever
provides the relevant ranked chunks. `PromptBuilder` combines both into the
final string intended for a future LLM service:

```python
from enterprise_multi_agent_rag.generation import PromptBuilder

prompt = PromptBuilder().build(question, results)
```

Each chunk is numbered and included in retriever order. The prompt instructs
the future model to answer only from the retrieved context and acknowledge when
the context is insufficient. `PromptBuilder` formats text only—it does not call
the retriever, embedding provider, vector store, or LLM.

## LLM Service

`PromptBuilder` creates the final grounded prompt, then `LLMService` sends that
prompt unchanged to the configured model and returns only its generated answer
text. OpenAI chat completions and AWS Bedrock Claude messages are supported.

Select the provider and model in `.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_API_KEY=replace-with-a-local-secret

# Or use Bedrock with credentials discovered by Boto3:
# LLM_PROVIDER=bedrock
# BEDROCK_CHAT_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

```python
from enterprise_multi_agent_rag.core.config import get_settings
from enterprise_multi_agent_rag.generation import create_llm_service

llm_service = create_llm_service(get_settings())
answer = llm_service.generate(prompt)
```

The service performs no retrieval, prompt construction, streaming, memory,
tool calling, or answer formatting. Unit tests use injected mocks and make no
paid OpenAI or AWS requests.

## RAG Service

`RAGService` coordinates retrieval, prompt construction, and answer generation
behind one method. It contains no embedding, search, formatting, or AI provider
logic of its own; each existing component retains one responsibility.

```python
from enterprise_multi_agent_rag.generation import PromptBuilder, RAGService

rag_service = RAGService(
    retriever=retriever,
    prompt_builder=PromptBuilder(),
    llm_service=llm_service,
)
answer = rag_service.answer("How many vacation days are provided?", k=5)
```

Errors from the retriever, prompt builder, or LLM service propagate unchanged
to the caller. No citations, memory, reranking, query rewriting, or answer
post-processing are added by this coordinator.

## Planned technology

LangChain, LangGraph, OpenAI, Anthropic Claude, Amazon Bedrock, FastAPI, FAISS,
MCP, Docker, GitHub Actions, and AWS.
