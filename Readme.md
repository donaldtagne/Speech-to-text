# Speech to Text Project

Dieses Repository enthält ein referenzierbares Beispiel, wie man ein Retrieval-Augmented-Generation (RAG) System mit Sprachschnittstelle aufbaut. Die Lösung kombiniert folgende Bausteine:

- **OpenAI Whisper** für automatische Spracherkennung.
- **Sentence-Transformers** für dichte Dokument-Embeddings.
- **Hugging-Face Text-Generierungsmodelle** um Antworten aus relevantem Kontext zu erzeugen.

## Voraussetzungen

1. Python 3.10 oder höher.
2. Ein funktionierendes [FFmpeg](https://ffmpeg.org/) Binary im `PATH`, damit Whisper Audio-Dateien dekodieren kann.
3. Installieren Sie die Python-Abhängigkeiten (Torch muss zur lokalen Hardware passen):

   ```bash
   pip install -r requirements.txt
   ```

> **Hinweis:** Für Whisper und viele Transformer-Modelle ist eine GPU empfehlenswert. Auf CPU funktioniert die Pipeline ebenfalls, allerdings deutlich langsamer.

## Projektstruktur

```text
.
├── artifacts/                 # Persistierte Vektoren nach der Ingestion
├── data/
│   └── documents/             # Beispiel-Wissensbasis
├── main.py                    # Einstiegspunkt für die CLI
├── rag_voice_chat/            # Python-Paket mit Pipeline-Code
└── requirements.txt
```

## Nutzung

1. **Dokumente indizieren**

   Legen Sie Ihre Wissensbasis in `data/documents/` (Markdown oder TXT) ab und starten Sie die Ingestion:

   ```bash
   python -m main ingest
   ```

   Optional lassen sich Pfade und Modellnamen über CLI-Optionen anpassen. Der Befehl erzeugt eine komprimierte NumPy-Datei mit Embeddings sowie ein JSON mit Metadaten.

2. **Audiofrage stellen**

   Nehmen Sie eine Frage als Audiodatei (z. B. WAV oder MP3) auf und führen Sie den Sprachdialog-Befehl aus:

   ```bash
   python -m main voice-query pfad/zur/frage.wav
   ```

   Die CLI zeigt die transkribierte Frage, den wiedergegebenen Kontext und die erzeugte Antwort an.

## Konfiguration

Standardwerte wie Modellnamen, maximale Antwortlänge oder Anzahl der Kontextdokumente werden über `rag_voice_chat/config.py` verwaltet. Sie lassen sich über CLI-Flags überschreiben.

## Erweiterungen

- Austausch des lokalen LLM gegen eine gehostete API (z. B. OpenAI, Azure, vLLM).
- Persistierung des Vektorstores in einer dedizierten Datenbank (z. B. Qdrant, Weaviate).
- Aufbau einer Weboberfläche mit Streaming-Antworten und Mikrofon-Upload.

Viel Erfolg beim Aufbau Ihres RAG-Systems! 💡
