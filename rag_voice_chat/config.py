from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator


class Settings(BaseModel):
    """Runtime configuration for the voice-enabled RAG pipeline."""

    documents_path: Path = Field(
        default=Path("data/documents"),
        description="Directory that contains the knowledge base documents.",
    )
    index_path: Path = Field(
        default=Path("artifacts/index.npz"),
        description="Location on disk where document embeddings are stored.",
    )
    whisper_model_size: str = Field(
        default="small",
        description="Whisper model size used for audio transcription.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformer model name used to create embeddings.",
    )
    llm_model: str = Field(
        default="distilgpt2",
        description="Hugging Face text-generation model used for response synthesis.",
    )
    max_context_documents: int = Field(
        default=3,
        ge=1,
        description="How many documents from the retriever should be passed to the LLM.",
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        description="Maximum number of tokens to generate for the answer.",
    )
    device: Optional[str] = Field(
        default=None,
        description="Torch device string. When omitted, the pipeline will infer CPU/GPU automatically.",
    )

    @validator("documents_path", "index_path", pre=True)
    def _expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()


settings = Settings()
