"""Supervised fine-tuning data processor."""
from typing import Dict, List, Any, Optional

import torch
from torch import Tensor

from pipeline.tokenizer import BpeTokenizer
from pipeline.strategies import PromptStrategy, ChatMLStrategy
from pipeline.processors.base import BaseProcessor, _encode_with_mask
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("sft")
class SFTProcessor(BaseProcessor):
    """Supervised fine-tuning data processor.

    Supports custom prompt strategy via constructor parameter.
    """

    def __init__(
        self,
        tokenizer: BpeTokenizer,
        strategy: Optional[PromptStrategy] = None,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy or ChatMLStrategy(tokenizer)

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        query_tokens = self.tokenizer.encode(input_dict["query"])
        response_tokens = self.tokenizer.encode(input_dict["response"])

        prompt = self.strategy.assemble_prompt(query_tokens)
        response = self.strategy.assemble_response(response_tokens)

        tokens, loss_mask = _encode_with_mask(prompt, response)
        return {"sequence": tokens, "loss_mask": loss_mask}

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask"]
