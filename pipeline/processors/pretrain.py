"""Pre-training data processor."""

from typing import Dict, List, Any

import torch
from torch import Tensor

from pipeline.tokenize import AutoTokenizer
from pipeline.processors.base import BaseProcessor
from pipeline.processors.factory import ProcessorFactory


@ProcessorFactory.register("pt")
class PreTrainProcessor(BaseProcessor):
    """Pre-training data processor."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        segment = input_dict["text"]
        tokens = self.tokenizer.encode(f"{segment}<｜end▁of▁sentence｜>")
        return {"sequence": torch.tensor(tokens, dtype=torch.int32)}

    @property
    def output_keys(self) -> List[str]:
        return ["sequence"]
