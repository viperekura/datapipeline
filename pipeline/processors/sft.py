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

    Supports two input formats:
      1. messages (recommended):
         ``{"messages": [{"role": "user", "content": "..."},
                         {"role": "assistant", "content": "..."}]}``
         Multi-turn and system prompts are supported.
         The tokenizer's ``apply_chat_template`` is used for rendering.
      2. legacy query/response:
         ``{"query": "...", "response": "..."}``
         Falls back to the configured PromptStrategy (ChatML by default).

    Output schema:
        - sequence: int32 tensor - Combined token IDs (prompt + response)
        - loss_mask: bool tensor - True for response tokens (compute loss)
        - position_ids: int32 tensor - Per-sample position IDs starting from 0
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        strategy: Optional[PromptStrategy] = None,
    ):
        self.tokenizer = tokenizer
        self.strategy = strategy

    @property
    def schema(self) -> ProcessorSchema:
        return ProcessorSchema(
            input_fields={
                "messages": list,
                "query": str,
                "response": str,
            },
            output_fields={
                "sequence": torch.int32,
                "loss_mask": torch.bool,
                "position_ids": torch.int32,
            },
        )

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        if "messages" in input_dict:
            return self._process_messages(input_dict["messages"])
        if "query" in input_dict and "response" in input_dict:
            return self._process_legacy(input_dict)
        raise KeyError(
            "Input must contain 'messages' or 'query'/'response' pair"
        )

    def _process_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Tensor]:
        if not messages:
            raise ValueError("Messages list is empty")
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message must have role 'assistant'")

        last_asst_idx = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )

        prompt_tokens = self.tokenizer.apply_chat_template(
            messages[:last_asst_idx],
            add_generation_prompt=True,
            tokenize=True,
        )

        resp_content = messages[last_asst_idx]["content"]
        im_end = getattr(self.tokenizer, "im_end", "<|im_end|>")
        resp_tokens = self.tokenizer.encode(
            f"{resp_content}{im_end}\n", add_special_tokens=False
        )

        tokens, loss_mask = encode_with_mask(prompt_tokens, resp_tokens)
        position_ids = torch.arange(len(tokens), dtype=torch.int32)
        return {"sequence": tokens, "loss_mask": loss_mask, "position_ids": position_ids}

    def _process_legacy(self, input_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        strategy = self.strategy or ChatMLStrategy(self.tokenizer)

        query_tokens = self.tokenizer.encode(input_dict["query"])
        response_tokens = self.tokenizer.encode(input_dict["response"])

        prompt = strategy.assemble_prompt(query_tokens)
        response = strategy.assemble_response(response_tokens)

        tokens, loss_mask = encode_with_mask(prompt, response)
        position_ids = torch.arange(len(tokens), dtype=torch.int32)
        return {"sequence": tokens, "loss_mask": loss_mask, "position_ids": position_ids}

    @property
    def output_keys(self) -> List[str]:
        return ["sequence", "loss_mask", "position_ids"]
