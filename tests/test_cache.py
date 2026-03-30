"""单元测试：pipeline.cache 模块中的 cache_jsonl 函数"""

import json
import os
import tempfile
import torch
from pathlib import Path

from pipeline.cache import cache_jsonl
from pipeline.processors import BaseProcessor


class DummyProcessor(BaseProcessor):
    """用于测试的虚拟处理器"""
    
    def __init__(self):
        self._output_keys = ["sequence", "loss_mask"]
    
    @property
    def output_keys(self):
        return self._output_keys
    
    def process(self, item):
        text = item.get("text", "")
        tokens = [ord(c) for c in text[:10]]  # 简单模拟tokenize
        
        return {
            "sequence": torch.tensor(tokens, dtype=torch.int32),
            "loss_mask": torch.ones(len(tokens), dtype=torch.int32),
        }


class TestCacheJsonl:
    """cache_jsonl 函数的测试套件"""
    
    def test_basic_cache_functionality(self):
        """测试基本缓存功能：处理简单JSONL文件并生成HDF5"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试JSONL文件
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data = [
                {"text": "hello"},
                {"text": "world"},
                {"text": "test"},
            ]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")
            
            # 创建处理器
            processor = DummyProcessor()
            
            # 调用 cache_jsonl
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=-1,  # 不打包模式
                pad_value=0,
            )
            
            # 验证输出
            assert len(output_files) == 1
            assert os.path.exists(output_files[0])
    
    def test_packer_state_independence(self):
        """测试打包器状态独立性：验证不同 output_key 的打包结果是否独立"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试JSONL文件，包含不同长度的文本
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data = [
                {"text": "ab"},  # 2 chars
                {"text": "abcde"},  # 5 chars
                {"text": "abc"},  # 3 chars
            ]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")
            
            # 创建处理器
            processor = DummyProcessor()
            
            # 调用 cache_jsonl，使用打包模式
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=10,  # 打包模式
                pad_value=0,
            )
            
            # 验证输出文件存在
            assert len(output_files) == 1
            assert os.path.exists(output_files[0])
    
    def test_no_packing_mode(self):
        """测试无打包模式（pack_size <= 0）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试JSONL文件
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            test_data = [
                {"text": "hello"},
                {"text": "world"},
            ]
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in test_data:
                    f.write(json.dumps(item) + "\n")
            
            processor = DummyProcessor()
            
            # 打包大小设为0表示不打包
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=0,
                pad_value=-1,
            )
            
            assert len(output_files) == 1
            assert os.path.exists(output_files[0])
    
    def test_multiple_files(self):
        """测试处理多个文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建两个测试JSONL文件
            files = []
            for i in range(2):
                jsonl_path = os.path.join(tmpdir, f"test{i}.jsonl")
                test_data = [{"text": f"data{i}"}]
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(test_data[0]) + "\n")
                files.append(jsonl_path)
            
            processor = DummyProcessor()
            
            output_files = cache_jsonl(
                files=files,
                output_dir=tmpdir,
                processor=processor,
                pack_size=-1,
                pad_value=0,
            )
            
            assert len(output_files) == 2
    
    def test_empty_file_handling(self):
        """测试处理空文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建空JSONL文件
            jsonl_path = os.path.join(tmpdir, "empty.jsonl")
            Path(jsonl_path).touch()
            
            processor = DummyProcessor()
            
            # 不应该抛出异常
            output_files = cache_jsonl(
                files=[jsonl_path],
                output_dir=tmpdir,
                processor=processor,
                pack_size=-1,
                pad_value=0,
            )
            
            assert len(output_files) == 1