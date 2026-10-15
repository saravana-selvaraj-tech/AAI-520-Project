"""Local Hugging Face generation for the research workflow."""

from __future__ import annotations

from typing import Any


class QwenAnswerGenerator:
    """Generate grounded answers with a local Qwen causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer: Any | None = None,
        model: Any | None = None,
        device: str | None = None,
        max_new_tokens: int = 2000,
    ) -> None:
        import torch

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or self._default_device(torch)
        self._torch = torch

        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            model = model or AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, query: str, context: str, system_context: str = None) -> str:
        if not context.strip():
            return f"No retrieved evidence is available to answer: {query}"
        messages = [
            {
                "role": "system",
                "content": system_context if system_context else "Act as an Financial Research Agent",
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved evidence:\n{context}",
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        prompt_length = inputs["input_ids"].shape[-1]

        with self._torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer_tokens = output_tokens[0][prompt_length:]
        answer = self.tokenizer.decode(
            answer_tokens,
            skip_special_tokens=True,
        ).strip()
        return answer or "The model did not produce an answer from the supplied evidence."

    @staticmethod
    def _default_device(torch: Any) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
