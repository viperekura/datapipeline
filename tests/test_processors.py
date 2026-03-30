"""Tests for pipeline.processors module."""

import pytest
import torch

from pipeline.processors import (
    BaseProcessor,
    PreTrainProcessor,
    SFTProcessor,
    DPOProcessor,
    ProcessorFactory,
)


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]


class TestBaseProcessor:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseProcessor()


class TestPreTrainProcessor:
    def test_output_keys(self):
        assert PreTrainProcessor(DummyTokenizer()).output_keys == ["sequence"]

    def test_process_returns_tensor(self):
        processor = PreTrainProcessor(DummyTokenizer())
        result = processor.process({"text": "hello world"})
        assert "sequence" in result
        assert isinstance(result["sequence"], torch.Tensor)
        assert result["sequence"].dtype == torch.int32

    def test_process_adds_eos(self):
        result = PreTrainProcessor(DummyTokenizer()).process({"text": "a"})
        assert len(result["sequence"]) > 0


class TestSFTProcessor:
    def test_output_keys(self):
        assert SFTProcessor(DummyTokenizer()).output_keys == ["sequence", "loss_mask"]

    def test_process_returns_both_keys(self):
        result = SFTProcessor(DummyTokenizer()).process({"query": "hello", "response": "world"})
        assert "sequence" in result
        assert "loss_mask" in result
        assert isinstance(result["sequence"], torch.Tensor)
        assert isinstance(result["loss_mask"], torch.Tensor)

    def test_loss_mask_correct_length(self):
        result = SFTProcessor(DummyTokenizer()).process({"query": "hi", "response": "bye"})
        assert len(result["sequence"]) == len(result["loss_mask"])

    def test_loss_mask_is_bool(self):
        result = SFTProcessor(DummyTokenizer()).process({"query": "ab", "response": "cd"})
        assert result["loss_mask"].dtype == torch.bool


class TestDPOProcessor:
    def test_output_keys(self):
        keys = DPOProcessor(DummyTokenizer()).output_keys
        assert keys == ["chosen", "chosen_mask", "rejected", "rejected_mask"]

    def test_process_returns_all_keys(self):
        result = DPOProcessor(DummyTokenizer()).process(
            {"query": "hello", "chosen": "r1", "rejected": "r2"}
        )
        for key in ["chosen", "chosen_mask", "rejected", "rejected_mask"]:
            assert key in result
            assert isinstance(result[key], torch.Tensor)

    def test_masks_match_lengths(self):
        result = DPOProcessor(DummyTokenizer()).process(
            {"query": "test", "chosen": "yes", "rejected": "no"}
        )
        assert len(result["chosen"]) == len(result["chosen_mask"])
        assert len(result["rejected"]) == len(result["rejected_mask"])

    def test_masks_are_bool(self):
        result = DPOProcessor(DummyTokenizer()).process(
            {"query": "test", "chosen": "yes", "rejected": "no"}
        )
        assert result["chosen_mask"].dtype == torch.bool
        assert result["rejected_mask"].dtype == torch.bool


class TestProcessorFactory:
    def test_create_pre_train_processor(self):
        assert isinstance(ProcessorFactory.create("pt", DummyTokenizer()), PreTrainProcessor)

    def test_create_sft_processor(self):
        assert isinstance(ProcessorFactory.create("sft", DummyTokenizer()), SFTProcessor)

    def test_create_dpo_processor(self):
        assert isinstance(ProcessorFactory.create("dpo", DummyTokenizer()), DPOProcessor)

    def test_create_invalid_processor_raises_error(self):
        with pytest.raises(ValueError, match="Unknown processor type"):
            ProcessorFactory.create("invalid", DummyTokenizer())

    def test_register_and_create_custom_processor(self):
        class CustomProcessor(BaseProcessor):
            def __init__(self, tokenizer=None):
                self._tokenizer = tokenizer

            @property
            def output_keys(self):
                return ["custom"]

            def process(self, input_dict):
                return {"custom": torch.tensor([1, 2, 3])}

        ProcessorFactory.register("custom")(CustomProcessor)
        assert isinstance(ProcessorFactory.create("custom", DummyTokenizer()), CustomProcessor)
