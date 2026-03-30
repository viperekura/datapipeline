from abc import ABC, abstractmethod
from typing import Dict, List
import torch


class BaseProcessor(ABC):
    """处理器抽象基类"""

    @abstractmethod
    def process(self, input_dict: dict) -> dict:
        pass

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        pass


class PreTrainProcessor(BaseProcessor):
    """预训练数据处理器"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, input_dict: dict) -> dict:
        segment = input_dict["text"]
        tokens = self.tokenizer.encode(f"{segment}<eos>")
        return {'sequence': torch.tensor(tokens, dtype=torch.int32)}

    @property
    def output_keys(self) -> List[str]:
        return ["sequence"]


class SFTProcessor(BaseProcessor):
    """监督微调数据处理器"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, input_dict: dict) -> dict:
        query, response = input_dict["query"], input_dict["response"]
        q = self.tokenizer.encode(
            f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        )
        a = self.tokenizer.encode(f"{response}<|im_end|>\n<eos>")
        tokens = torch.tensor(q + a, dtype=torch.int32)
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        loss_mask[len(q):] = True
        return {"sequence": tokens, "loss_mask": loss_mask}

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask"]


class DPOProcessor(BaseProcessor):
    """DPO 偏好学习数据处理器"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process(self, input_dict: dict) -> dict:
        query = input_dict["query"]
        chosen_response = input_dict["chosen"]
        rejected_response = input_dict["rejected"]

        q = self.tokenizer.encode(
            f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        )

        chosen = self.tokenizer.encode(f"{chosen_response}<|im_end|>\n<eos>")
        chosen_tokens = torch.tensor(q + chosen, dtype=torch.int32)
        chosen_mask = torch.zeros_like(chosen_tokens, dtype=torch.bool)
        chosen_mask[len(q):] = True

        rejected = self.tokenizer.encode(f"{rejected_response}<|im_end|>\n<eos>")
        rejected_tokens = torch.tensor(q + rejected, dtype=torch.int32)
        rejected_mask = torch.zeros_like(rejected_tokens, dtype=torch.bool)
        rejected_mask[len(q):] = True

        return {
            "chosen": chosen_tokens,
            "chosen_mask": chosen_mask,
            "rejected": rejected_tokens,
            "rejected_mask": rejected_mask,
        }

    @property
    def output_keys(self) -> List[str]:
        return ["chosen", "chosen_mask", "rejected", "rejected_mask"]


class ProcessorFactory:
    """处理器工厂"""

    _processors = {
        "pt": PreTrainProcessor,
        "sft": SFTProcessor,
        "dpo": DPOProcessor,
    }

    @classmethod
    def create(cls, processor_type: str, tokenizer) -> BaseProcessor:
        if processor_type not in cls._processors:
            raise ValueError(f"Invalid processor type: {processor_type}")
        return cls._processors[processor_type](tokenizer)

    @classmethod
    def register(cls, processor_type: str, processor_class: type):
        cls._processors[processor_type] = processor_class
