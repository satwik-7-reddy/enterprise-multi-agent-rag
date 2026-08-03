"""Document indexing coordination for a persistent FAISS knowledge base."""

from dataclasses import dataclass

from enterprise_multi_agent_rag.chunking import DocumentChunker
from enterprise_multi_agent_rag.embeddings import EmbeddingService
from enterprise_multi_agent_rag.ingestion.documents import DocumentLoader
from enterprise_multi_agent_rag.retrieval import FAISSVectorStore


@dataclass(frozen=True)
class DocumentIndexingResult:
    """Statistics for one indexed document."""

    document_id: str
    chunks_created: int
    embeddings_stored: int


class DocumentIndexingService:
    """Coordinate existing components to build a searchable knowledge base."""

    def __init__(
        self,
        document_loader: DocumentLoader,
        document_chunker: DocumentChunker,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.document_loader = document_loader
        self.document_chunker = document_chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_document(
        self,
        document_path: str,
        output_directory: str = "vector_store",
    ) -> DocumentIndexingResult:
        """Index one local document and persist the resulting vector store."""
        return self.index_documents([document_path], output_directory)[0]

    def index_documents(
        self,
        document_paths: list[str],
        output_directory: str = "vector_store",
    ) -> list[DocumentIndexingResult]:
        """Index local documents together and persist the vector store once."""
        if not document_paths:
            raise ValueError("document_paths must contain at least one document")

        all_embeddings = []
        results: list[DocumentIndexingResult] = []
        for document_path in document_paths:
            document = self.document_loader.load(document_path)
            chunks = self.document_chunker.chunk(document)
            embeddings = self.embedding_service.embed_chunks(chunks)
            all_embeddings.extend(embeddings)
            results.append(
                DocumentIndexingResult(
                    document_id=document.document_id,
                    chunks_created=len(chunks),
                    embeddings_stored=len(embeddings),
                )
            )

        self.vector_store.add_chunks(all_embeddings)
        self.vector_store.save(output_directory)
        return results
