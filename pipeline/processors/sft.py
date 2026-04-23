"""Supervised fine-tuning data processor."""

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies import PromptStrategy, ChatMLStrategy
from pipeline.processors.base import BaseProcessor, ProcessorSchema, encode_with_mask
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("sft")
class SFTProcessor(BaseProcessor):
    """Supervised fine-tuning data processor.

    Processes query-response pairs into tokenized sequences with loss masks.

    Input schema:
        - query: str - User query/prompt
        - response: str - Assistant response

    Output schema:
        - sequence: int32 tensor - Combined token IDs (query + response)
        - loss_mask: bool tensor - True for response tokens (compute loss)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        strategy: Optional[PromptStrategy] = None,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy or ChatMLStrategy(tokenizer)

    @property
    def schema(self) -> ProcessorSchema:
        return ProcessorSchema(
            input_fields={"query": str, "response": str},
            output_fields={
                "sequence": torch.int32,
                "loss_mask": torch.bool,
            },
        )

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        query_tokens = self.tokenizer.encode(input_dict["query"])
        response_tokens = self.tokenizer.encode(input_dict["response"])

        prompt = self.strategy.assemble_prompt(query_tokens)
        response = self.strategy.assemble_response(response_tokens)

        tokens, loss_mask = encode_with_mask(prompt, response)
        return {"sequence": tokens, "loss_mask": loss_mask}

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask"]
