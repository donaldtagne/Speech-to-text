from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalLLM:
    """Utility wrapper around Hugging Face causal language models."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.generator = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            device=0 if device.startswith("cuda") else -1,
        )

    def answer(self, context: str, question: str, max_new_tokens: int = 256) -> str:
        prompt = (
            "You are an assistant that answers questions based on provided context.\n"
            "If the answer cannot be determined from the context, reply with a polite fallback.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        outputs = self.generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return outputs[0]["generated_text"][len(prompt) :].strip()
