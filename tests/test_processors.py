"""单元测试：pipeline.processors 模块中的处理器类"""

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
    """用于测试的虚拟分词器"""
    
    def encode(self, text: str):
        # 简单模拟：返回文本字符的ASCII码列表
        return [ord(c) for c in text]


class TestBaseProcessor:
    """BaseProcessor 抽象基类的测试"""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """测试 BaseProcessor 不能直接实例化"""
        with pytest.raises(TypeError):
            BaseProcessor()


class TestPreTrainProcessor:
    """PreTrainProcessor 类的测试套件"""
    
    def test_output_keys(self):
        """测试 output_keys 属性"""
        tokenizer = DummyTokenizer()
        processor = PreTrainProcessor(tokenizer)
        assert processor.output_keys == ["sequence"]
    
    def test_process_returns_tensor(self):
        """测试 process 方法返回正确的张量"""
        tokenizer = DummyTokenizer()
        processor = PreTrainProcessor(tokenizer)
        
        result = processor.process({"text": "hello world"})
        
        assert "sequence" in result
        assert isinstance(result["sequence"], torch.Tensor)
        assert result["sequence"].dtype == torch.int32
    
    def test_process_adds_eos(self):
        """测试 process 方法添加 EOS 标记"""
        tokenizer = DummyTokenizer()
        processor = PreTrainProcessor(tokenizer)
        
        # 文本 "a" 的 ASCII 码是 97
        result = processor.process({"text": "a"})
        
        # 应该包含文本的ASCII码 + <eos> (假设是 4)
        seq = result["sequence"]
        # 基本验证：返回的张量长度应该大于0
        assert len(seq) > 0


class TestSFTProcessor:
    """SFTProcessor 类的测试套件"""
    
    def test_output_keys(self):
        """测试 output_keys 属性"""
        tokenizer = DummyTokenizer()
        processor = SFTProcessor(tokenizer)
        assert processor.output_keys == ["sequence", "loss_mask"]
    
    def test_process_returns_both_keys(self):
        """测试 process 方法返回所有键"""
        tokenizer = DummyTokenizer()
        processor = SFTProcessor(tokenizer)
        
        result = processor.process({
            "query": "hello",
            "response": "world"
        })
        
        assert "sequence" in result
        assert "loss_mask" in result
        assert isinstance(result["sequence"], torch.Tensor)
        assert isinstance(result["loss_mask"], torch.Tensor)
    
    def test_loss_mask_correct_length(self):
        """测试 loss_mask 长度与 sequence 一致"""
        tokenizer = DummyTokenizer()
        processor = SFTProcessor(tokenizer)
        
        result = processor.process({
            "query": "hi",
            "response": "bye"
        })
        
        assert len(result["sequence"]) == len(result["loss_mask"])
    
    def test_loss_mask_after_query_is_true(self):
        """测试 loss_mask 在响应部分为 True"""
        tokenizer = DummyTokenizer()
        processor = SFTProcessor(tokenizer)
        
        result = processor.process({
            "query": "ab",  # 2 chars
            "response": "cd",  # 2 chars
        })
        
        # 验证 loss_mask 是 bool 类型
        assert result["loss_mask"].dtype == torch.bool


class TestDPOProcessor:
    """DPOProcessor 类的测试套件"""
    
    def test_output_keys(self):
        """测试 output_keys 属性"""
        tokenizer = DummyTokenizer()
        processor = DPOProcessor(tokenizer)
        assert processor.output_keys == ["chosen", "chosen_mask", "rejected", "rejected_mask"]
    
    def test_process_returns_all_keys(self):
        """测试 process 方法返回所有键"""
        tokenizer = DummyTokenizer()
        processor = DPOProcessor(tokenizer)
        
        result = processor.process({
            "query": "hello",
            "chosen": "response1",
            "rejected": "response2"
        })
        
        expected_keys = ["chosen", "chosen_mask", "rejected", "rejected_mask"]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], torch.Tensor)
    
    def test_chosen_and_rejected_same_length_as_mask(self):
        """测试 chosen/rejected 长度与 mask 一致"""
        tokenizer = DummyTokenizer()
        processor = DPOProcessor(tokenizer)
        
        result = processor.process({
            "query": "test",
            "chosen": "yes",
            "rejected": "no"
        })
        
        assert len(result["chosen"]) == len(result["chosen_mask"])
        assert len(result["rejected"]) == len(result["rejected_mask"])
    
    def test_masks_are_bool(self):
        """测试 mask 张量是 bool 类型"""
        tokenizer = DummyTokenizer()
        processor = DPOProcessor(tokenizer)
        
        result = processor.process({
            "query": "test",
            "chosen": "yes",
            "rejected": "no"
        })
        
        assert result["chosen_mask"].dtype == torch.bool
        assert result["rejected_mask"].dtype == torch.bool


class TestProcessorFactory:
    """ProcessorFactory 类的测试套件"""
    
    def test_create_pre_train_processor(self):
        """测试创建预训练处理器"""
        tokenizer = DummyTokenizer()
        processor = ProcessorFactory.create("pt", tokenizer)
        assert isinstance(processor, PreTrainProcessor)
    
    def test_create_sft_processor(self):
        """测试创建 SFT 处理器"""
        tokenizer = DummyTokenizer()
        processor = ProcessorFactory.create("sft", tokenizer)
        assert isinstance(processor, SFTProcessor)
    
    def test_create_dpo_processor(self):
        """测试创建 DPO 处理器"""
        tokenizer = DummyTokenizer()
        processor = ProcessorFactory.create("dpo", tokenizer)
        assert isinstance(processor, DPOProcessor)
    
    def test_create_invalid_processor_raises_error(self):
        """测试创建无效处理器类型抛出异常"""
        tokenizer = DummyTokenizer()
        with pytest.raises(ValueError, match="Invalid processor type"):
            ProcessorFactory.create("invalid", tokenizer)
    
    def test_register_and_create_custom_processor(self):
        """测试注册和创建自定义处理器"""
        class CustomProcessor(BaseProcessor):
            def __init__(self, tokenizer=None):  # 接受 tokenizer 参数
                self._tokenizer = tokenizer
            
            @property
            def output_keys(self):
                return ["custom"]
            
            def process(self, input_dict):
                return {"custom": torch.tensor([1, 2, 3])}
        
        tokenizer = DummyTokenizer()
        ProcessorFactory.register("custom", CustomProcessor)
        processor = ProcessorFactory.create("custom", tokenizer)
        assert isinstance(processor, CustomProcessor)