"""DPO preference learning data processor."""

from typing import Dict, List, Any, Optional

import torch
from torch import Tensor

from pipeline.tokenizer import BpeTokenizer
from pipeline.strategies import PromptStrategy, ChatMLStrategy
from pipeline.processors.base import BaseProcessor, _encode_with_mask
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("dpo")
class DPOProcessor(BaseProcessor):
    """DPO preference learning data processor.

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
        chosen_tokens = self.tokenizer.encode(input_dict["chosen"])
        rejected_tokens = self.tokenizer.encode(input_dict["rejected"])

        prompt = self.strategy.assemble_prompt(query_tokens)

        chosen_t, chosen_m = _encode_with_mask(
            prompt, self.strategy.assemble_response(chosen_tokens)
        )
        rejected_t, rejected_m = _encode_with_mask(
            prompt, self.strategy.assemble_response(rejected_tokens)
        )

        return {
            "chosen": chosen_t,
            "chosen_mask": chosen_m,
            "rejected": rejected_t,
            "rejected_mask": rejected_m,
        }

    @property
    def output_keys(self) -> List[str]:
        return ["chosen", "chosen_mask", "rejected", "rejected_mask"]
