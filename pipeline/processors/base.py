"""Processor base class and shared utilities."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple

import torch
from torch import Tensor


def _encode_with_mask(
    prompt_tokens: List[int],
    response_tokens: List[int],
) -> Tuple[Tensor, Tensor]:
    """Concatenate token lists and build loss mask (prompt=False, response=True)."""
    q_len = len(prompt_tokens)
    combined = torch.tensor(prompt_tokens + response_tokens, dtype=torch.int32)
    mask = torch.zeros(q_len + len(response_tokens), dtype=torch.bool)
    mask[q_len:] = True
    return combined, mask


class BaseProcessor(ABC):
    """Abstract base class for processors."""

    @abstractmethod
    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        pass
