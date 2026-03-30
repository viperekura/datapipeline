"""Alpaca format strategy."""
from typing import List

from pipeline.tokenizer import BpeTokenizer
from pipeline.strategies.base import PromptStrategy
from pipeline.strategies.factory import StrategyFactory


@StrategyFactory.register("alpaca")
class AlpacaStrategy(PromptStrategy):
    """Alpaca format: ``### Instruction: ... \\n\\n### Response: ... <eos>``"""

    def __init__(
        self,
        tokenizer: BpeTokenizer,
        instruction_start: str = "### Instruction:\n",
        response_start: str = "### Response:\n",
        response_suffix: str = "\n<eos>",
    ):
        super().__init__(tokenizer)
        self.instruction_start = instruction_start
        self.response_start = response_start
        self.response_suffix = response_suffix

        self._instruction_start_ids = self._encode_format(instruction_start)
        self._separator_ids = self._encode_format("\n\n")
        self._response_start_ids = self._encode_format(response_start)
        self._response_suffix_ids = self._encode_format(response_suffix)

    @property
    def name(self) -> str:
        return "alpaca"

    def assemble_prompt(self, query_tokens: List[int]) -> List[int]:
        return (
            self._instruction_start_ids
            + query_tokens
            + self._separator_ids
            + self._response_start_ids
        )

    def assemble_response(self, response_tokens: List[int]) -> List[int]:
        return response_tokens + self._response_suffix_ids
