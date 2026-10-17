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
        max_new_tokens: int = 512,
        max_input_chars: int = 6000,
    ) -> None:
        import torch

        self.model_name = model_name
        # Bound generation by default for CPU-only notebooks. Callers can opt
        # into larger outputs explicitly when memory is available.
        self.max_new_tokens = max(32, min(int(max_new_tokens), 1024))
        self.max_input_chars = max(1000, int(max_input_chars))
        self.device = device or self._default_device(torch)
        self._torch = torch

        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            if model is None:
                # Prefer low-memory loading where the optional accelerate
                # dependency is installed; retain compatibility without it.
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        low_cpu_mem_usage=True,
                    )
                except ImportError as exc:
                    if "accelerate" not in str(exc).casefold():
                        raise
                    model = AutoModelForCausalLM.from_pretrained(model_name)

        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, query: str, context: str, system_context: str = None) -> str:
        if not context.strip():
            return f"No retrieved evidence is available to answer: {query}"
        # Limit prompt payload before tokenization; notebook evidence may include
        # long filings and should not create an unbounded attention/KV cache.
        bounded_context = context[: self.max_input_chars]
        messages = [
            {
                "role": "system",
                "content": (system_context or "Act as an Financial Research Agent")[:1000],
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved evidence:\n{bounded_context}",
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

        try:
            with self._torch.inference_mode():
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            answer_tokens = output_tokens[0][prompt_length:]
            answer = self.tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True,
            ).strip()
            return answer or "The model did not produce an answer from the supplied evidence."
        finally:
            # Release large prompt/output tensors promptly between notebook runs.
            del inputs
            if "output_tokens" in locals():
                del output_tokens
            if self.device == "cpu":
                import gc
                gc.collect()

    @staticmethod
    def _default_device(torch: Any) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"