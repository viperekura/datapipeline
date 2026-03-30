"""Abstract base class for prompt construction strategies."""
from abc import ABC, abstractmethod
from typing import List

from pipeline.tokenizer import BpeTokenizer


class PromptStrategy(ABC):
    """Abstract base for prompt/response format strategies.

    Strategies operate at the token level: the Processor tokenizes raw
    text (query, response, …) and passes token lists to the Strategy,
    which assembles them with pre-encoded format tokens.
    """

    def __init__(self, tokenizer: BpeTokenizer):
        self.tokenizer = tokenizer

    def _encode_format(self, text: str) -> List[int]:
        """Encode a format string that may contain special tokens."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def assemble_prompt(self, query_tokens: List[int]) -> List[int]:
        """Assemble query tokens into a complete prompt with format tokens.

        The prompt includes all tokens up to (and including) the response
        start marker, e.g. ``<|im_start|>assistant\n``.
        """

    @abstractmethod
    def assemble_response(self, response_tokens: List[int]) -> List[int]:
        """Wrap response tokens with format tokens (suffix, eos, etc)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
