# Pipeline Steps

1. The ingestion command indexes Markdown and text files located in `data/documents` using the specified sentence-transformer.
2. Users record a question and provide the audio file to the `voice-query` command.
3. Whisper transcribes the audio into text which is embedded and matched against the vector index.
4. The highest scoring passages are appended to the LLM prompt, enabling grounded answers that cite the retrieved material.
