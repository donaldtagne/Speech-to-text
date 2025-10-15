from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .documents import Document


class InMemoryVectorStore:
    """A lightweight cosine-similarity based vector store for small corpora."""

    def __init__(self, embeddings: NDArray[np.float32], documents: list[Document]):
        if embeddings.shape[0] != len(documents):
            raise ValueError("Number of embeddings must match number of documents.")
        self.embeddings = embeddings.astype(np.float32)
        self.documents = documents
        self._norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self._norms[self._norms == 0] = 1.0

    def as_serializable(self) -> dict[str, NDArray[np.float32]]:
        return {"embeddings": self.embeddings}

    def query(self, query_embedding: NDArray[np.float32], top_k: int = 3) -> list[tuple[Document, float]]:
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            raise ValueError("The query embedding is zero; cannot compute cosine similarity.")
        normalized_query = query_embedding / query_norm
        normalized_docs = self.embeddings / self._norms.squeeze()
        scores = normalized_docs @ normalized_query
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[idx], float(scores[idx])) for idx in top_indices]


def save_index(path, embeddings: NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=embeddings.astype(np.float32))


def load_index(path) -> NDArray[np.float32]:
    if not path.exists():
        raise FileNotFoundError(
            "Vector index file not found. Please run the ingest command before querying."
        )
    data = np.load(path)
    return data["embeddings"].astype(np.float32)
