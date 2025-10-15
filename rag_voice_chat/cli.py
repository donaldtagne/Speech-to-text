from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings, settings
from .pipeline import VoiceEnabledRAG

cli = typer.Typer(help="Voice-enabled RAG assistant powered by Whisper and local Hugging Face models.")


@cli.command()
def ingest(
    documents_path: Path = typer.Option(settings.documents_path, help="Directory containing knowledge base documents."),
    index_path: Path = typer.Option(settings.index_path, help="Location to store the vector index."),
    embedding_model: str = typer.Option(settings.embedding_model, help="Sentence-Transformer model name."),
    device: str | None = typer.Option(settings.device, help="Torch device identifier (cpu, cuda, cuda:0, ...)."),
) -> None:
    """Compute embeddings for the document corpus and persist them to disk."""

    runtime_settings = Settings(
        documents_path=documents_path,
        index_path=index_path,
        embedding_model=embedding_model,
        device=device,
        llm_model=settings.llm_model,
        whisper_model_size=settings.whisper_model_size,
        max_context_documents=settings.max_context_documents,
        max_new_tokens=settings.max_new_tokens,
    )
    pipeline = VoiceEnabledRAG(runtime_settings)
    typer.echo("Building document embeddings…")
    pipeline.ingest()
    typer.secho("Ingestion complete!", fg=typer.colors.GREEN)


@cli.command("voice-query")
def voice_query(
    audio_path: Path = typer.Argument(..., exists=True, readable=True, help="Recorded audio file containing the user's question."),
    index_path: Path = typer.Option(settings.index_path, help="Path to the saved vector index."),
    whisper_model_size: str = typer.Option(settings.whisper_model_size, help="Whisper model size for transcription."),
    embedding_model: str = typer.Option(settings.embedding_model, help="Embedding model used during ingestion."),
    llm_model: str = typer.Option(settings.llm_model, help="Causal language model used to craft answers."),
    max_context_documents: int = typer.Option(settings.max_context_documents, help="Number of retrieved passages to condition the LLM on."),
    max_new_tokens: int = typer.Option(settings.max_new_tokens, help="Maximum tokens to generate for the response."),
    device: str | None = typer.Option(settings.device, help="Torch device identifier (cpu, cuda, cuda:0, ...)."),
) -> None:
    """Transcribe the audio input and answer the question using the RAG pipeline."""

    runtime_settings = Settings(
        documents_path=settings.documents_path,
        index_path=index_path,
        whisper_model_size=whisper_model_size,
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_context_documents=max_context_documents,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    pipeline = VoiceEnabledRAG(runtime_settings)
    typer.echo("Transcribing audio…")
    result = pipeline.query_from_audio(audio_path)

    typer.secho("\nTranscribed question:", fg=typer.colors.CYAN)
    typer.echo(result["question"])

    typer.secho("\nRetrieved context:", fg=typer.colors.MAGENTA)
    typer.echo(result["context"])

    typer.secho("\nAssistant response:", fg=typer.colors.GREEN)
    typer.echo(result["answer"])


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
