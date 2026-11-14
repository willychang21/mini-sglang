"""Text -> token ids. Chat messages go through the model's chat template first."""

from __future__ import annotations

from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..message import TokenizeMsg


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path)


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def tokenize_one(self, text: Any, chat_template_kwargs: Dict[str, Any] | None = None) -> List[int]:
        if isinstance(text, list) and (not text or isinstance(text[0], int)):
            return list(text)  # already token ids
        if isinstance(text, list):
            text = self.apply_chat_template(text, **(chat_template_kwargs or {}))
        return self.tokenizer.encode(text, add_special_tokens=False)

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[List[int]]:
        return [self.tokenize_one(m.text, m.chat_template_kwargs) for m in msgs]
