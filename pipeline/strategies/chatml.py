"""ChatML format strategy."""

from typing import List

from pipeline.tokenizer import BpeTokenizer
from pipeline.strategies.base import PromptStrategy
from pipeline.strategies.factory import StrategyFactory


@StrategyFactory.register("chatml")
class ChatMLStrategy(PromptStrategy):
    """ChatML format strategy."""

    def __init__(
        self,
        tokenizer: BpeTokenizer,
        user_start: str = "<｜im▁start｜>user\n",
        user_end: str = "<｜im▁end｜>\n",
        assistant_start: str = "<｜im▁start｜>assistant\n",
        assistant_end: str = "<｜im▁end｜>\n",
    ):
        super().__init__(tokenizer)

        self._user_start_ids = self._encode_format(user_start)
        self._user_end_ids = self._encode_format(user_end)
        self._assistant_start_ids = self._encode_format(assistant_start)
        self._assistant_end_ids = self._encode_format(assistant_end)

    @property
    def name(self) -> str:
        return "chatml"

    def assemble_prompt(self, query_tokens: List[int]) -> List[int]:
        return (
            self._user_start_ids
            + query_tokens
            + self._user_end_ids
            + self._assistant_start_ids
        )

    def assemble_response(self, response_tokens: List[int]) -> List[int]:
        return response_tokens + self._assistant_end_ids
