from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Document:
    """Container for knowledge base documents."""

    doc_id: str
    content: str
    source: str

    def to_json(self) -> dict[str, str]:
        return {"doc_id": self.doc_id, "content": self.content, "source": self.source}

    @classmethod
    def from_json(cls, payload: dict[str, str]) -> "Document":
        return cls(doc_id=payload["doc_id"], content=payload["content"], source=payload["source"])


def load_documents(path: Path) -> List[Document]:
    """Read `.txt` and `.md` files from disk and turn them into Document instances."""

    documents: List[Document] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = file_path.read_text(encoding="utf-8")
        doc_id = file_path.relative_to(path).as_posix()
        documents.append(Document(doc_id=doc_id, content=content, source=str(file_path.resolve())))
    if not documents:
        raise ValueError(f"No textual documents were found in {path}.")
    return documents


def save_documents(path: Path, documents: Iterable[Document]) -> None:
    """Persist serialized document metadata alongside the vector index."""

    payload = [doc.to_json() for doc in documents]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_serialized_documents(path: Path) -> List[Document]:
    """Load document metadata previously written by :func:`save_documents`."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Document.from_json(item) for item in payload]
