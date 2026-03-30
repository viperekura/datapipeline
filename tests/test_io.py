"""单元测试：pipeline.io 模块中的 IOHandler 类"""

import os
import tempfile
import pytest
import torch
import h5py
from pathlib import Path

from pipeline.io import IOHandler


class TestIOHandler:
    """IOHandler 类的测试套件"""
    
    def test_fetch_files_in_directory(self):
        """测试 fetch_files 方法能正确获取目录中的文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            test_file1 = os.path.join(tmpdir, "file1.txt")
            test_file2 = os.path.join(tmpdir, "file2.txt")
            Path(test_file1).touch()
            Path(test_file2).touch()
            
            # 创建子目录和文件
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            test_file3 = os.path.join(subdir, "file3.txt")
            Path(test_file3).touch()
            
            # 获取文件列表
            files = IOHandler.fetch_files(tmpdir)
            
            # 验证
            assert len(files) == 3
            assert any("file1.txt" in f for f in files)
            assert any("file2.txt" in f for f in files)
            assert any("file3.txt" in f for f in files)
    
    def test_fetch_files_empty_directory(self):
        """测试空目录返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = IOHandler.fetch_files(tmpdir)
            assert files == []
    
    def test_fetch_folders_in_directory(self):
        """测试 fetch_folders 方法能正确获取子目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建子目录
            subdir1 = os.path.join(tmpdir, "folder1")
            subdir2 = os.path.join(tmpdir, "folder2")
            os.makedirs(subdir1)
            os.makedirs(subdir2)
            
            # 创建嵌套子目录
            nested = os.path.join(subdir1, "nested")
            os.makedirs(nested)
            
            # 获取文件夹列表
            folders = IOHandler.fetch_folders(tmpdir)
            
            # 验证
            assert len(folders) == 3
            assert any("folder1" in f for f in folders)
            assert any("folder2" in f for f in folders)
            assert any("nested" in f for f in folders)
    
    def test_fetch_folders_with_filter(self):
        """测试 fetch_folders 方法的过滤功能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建子目录
            subdir1 = os.path.join(tmpdir, "folder1")
            subdir2 = os.path.join(tmpdir, "folder2")
            os.makedirs(subdir1)
            os.makedirs(subdir2)
            
            # 使用过滤器只获取 folder1
            folders = IOHandler.fetch_folders(
                tmpdir, 
                filter_func=lambda x: "folder1" in x
            )
            
            # 验证
            assert len(folders) == 1
            assert "folder1" in folders[0]
    
    def test_save_and_load_h5(self):
        """测试 save_h5 和 load_h5 方法的读写功能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试数据
            tensor_group = {
                "sequence": [torch.tensor([1, 2, 3], dtype=torch.int32)],
                "labels": [torch.tensor([4, 5], dtype=torch.int32)],
            }
            
            # 保存
            IOHandler.save_h5(tmpdir, "test", tensor_group)
            
            # 验证文件已创建
            h5_path = os.path.join(tmpdir, "test.h5")
            assert os.path.exists(h5_path)
            
            # 加载 - 传入目录而不是单个文件
            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            
            # 验证数据
            assert "sequence" in loaded
            assert "labels" in loaded
            assert len(loaded["sequence"]) == 1
            assert len(loaded["labels"]) == 1
            assert torch.equal(loaded["sequence"][0], torch.tensor([1, 2, 3], dtype=torch.int32))
            assert torch.equal(loaded["labels"][0], torch.tensor([4, 5], dtype=torch.int32))
    
    def test_save_h5_creates_directory(self):
        """测试 save_h5 自动创建输出目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "output")
            
            tensor_group = {
                "data": [torch.tensor([1, 2, 3])],
            }
            
            # 保存到不存在的目录
            IOHandler.save_h5(output_dir, "test", tensor_group)
            
            # 验证目录已创建
            assert os.path.exists(output_dir)
            assert os.path.exists(os.path.join(output_dir, "test.h5"))
    
    def test_load_h5_multiple_files(self):
        """测试 load_h5 方法能处理多个 H5 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建第一个 H5 文件
            h5_path1 = os.path.join(tmpdir, "file1.h5")
            with h5py.File(h5_path1, 'w') as f:
                grp = f.create_group("data")
                grp.create_dataset('data_0', data=[1, 2, 3])
            
            # 创建第二个 H5 文件
            h5_path2 = os.path.join(tmpdir, "file2.h5")
            with h5py.File(h5_path2, 'w') as f:
                grp = f.create_group("data")
                grp.create_dataset('data_0', data=[4, 5, 6])
            
            # 加载目录
            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            
            # 验证
            assert "data" in loaded
            assert len(loaded["data"]) == 2
    
    def test_load_h5_with_rglob(self):
        """测试 load_h5 能递归查找 H5 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 在子目录中创建 H5 文件
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            h5_path = os.path.join(subdir, "nested.h5")
            
            with h5py.File(h5_path, 'w') as f:
                grp = f.create_group("test")
                grp.create_dataset('data_0', data=[1, 2])
            
            # 加载根目录
            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            
            # 验证能找到子目录中的文件
            assert "test" in loaded
            assert len(loaded["test"]) == 1
    
    def test_save_h5_multiple_tensors_per_key(self):
        """测试 save_h5 能保存多个张量到同一键"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_group = {
                "batch": [
                    torch.tensor([1, 2]),
                    torch.tensor([3, 4, 5]),
                    torch.tensor([6]),
                ],
            }
            
            IOHandler.save_h5(tmpdir, "multi", tensor_group)
            
            # 加载目录而不是单个文件
            loaded = IOHandler.load_h5(tmpdir, share_memory=False)
            
            assert len(loaded["batch"]) == 3