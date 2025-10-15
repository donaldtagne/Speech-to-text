from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from .documents import Document


class Embedder:
    """Wrapper around SentenceTransformer to generate document embeddings."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, documents: Iterable[Document]) -> NDArray[np.float32]:
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.astype(np.float32)
