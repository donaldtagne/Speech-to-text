from __future__ import annotations

from pathlib import Path

from .config import Settings
from .documents import load_documents, load_serialized_documents, save_documents
from .embeddings import Embedder
from .llm import LocalLLM
from .transcription import transcribe_audio
from .vector_store import InMemoryVectorStore, load_index, save_index


class VoiceEnabledRAG:
    """High-level orchestration for the Whisper + RAG workflow."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedding_model, device=settings.device)
        self.llm = LocalLLM(settings.llm_model, device=settings.device)

    # ------------------- Ingestion -------------------
    def ingest(self) -> None:
        documents = load_documents(self.settings.documents_path)
        embeddings = self.embedder.encode_documents(documents)
        save_index(self.settings.index_path, embeddings)
        metadata_path = self._metadata_path(self.settings.index_path)
        save_documents(metadata_path, documents)

    # ------------------- Querying -------------------
    def _load_store(self) -> InMemoryVectorStore:
        embeddings = load_index(self.settings.index_path)
        metadata_path = self._metadata_path(self.settings.index_path)
        documents = load_serialized_documents(metadata_path)
        return InMemoryVectorStore(embeddings, documents)

    def query_from_audio(self, audio_path: Path) -> dict[str, str]:
        question = transcribe_audio(audio_path, model_size=self.settings.whisper_model_size)
        store = self._load_store()
        query_embedding = self.embedder.encode_query(question)
        results = store.query(query_embedding, top_k=self.settings.max_context_documents)
        context = "\n\n".join(f"[{score:.2f}] {doc.content}" for doc, score in results)
        answer = self.llm.answer(context, question, max_new_tokens=self.settings.max_new_tokens)
        return {
            "question": question,
            "answer": answer,
            "context": context,
        }

    @staticmethod
    def _metadata_path(index_path: Path) -> Path:
        return index_path.with_suffix(".documents.json")
