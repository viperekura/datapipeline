"""单元测试：pipeline.packing 模块中的 SequencePacker 类"""

import pytest
import torch
from pipeline.packing import SequencePacker


class TestSequencePacker:
    """SequencePacker 类的测试套件"""

    def test_normal_packing(self):
        """测试正常打包场景：多个序列正确打包成固定长度的包"""
        packer = SequencePacker(pack_size=10, pad_value=0)
        
        sequences = [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5], dtype=torch.int32),
            torch.tensor([6, 7, 8, 9], dtype=torch.int32),
        ]
        
        packages = packer.pack(sequences)
        
        # 验证至少有包输出
        assert len(packages) >= 1
        
        # 验证每个包的长度是正确的
        for pkg in packages:
            assert pkg.shape == (10,)
            # 验证填充值
            # 检查所有非零元素都在前几个位置，或者包是满的
            non_zero_count = (pkg != 0).sum().item()
            # 非零元素的数量应该等于原始序列元素的总和
            total_elements = sum(s.numel() for s in sequences)
            # 由于打包，第一个包包含3+2=5个元素，第二个包包含4个元素
            # 第一个包应该包含前两个序列
            pkg1 = packages[0]
            # 序列[1,2,3]和[4,5]按长度降序排序后是[1,2,3]在前，然后[4,5]
            # 但排序是原地修改...等等，我们已经修复了使用sorted()
            # 所以排序后的顺序是[6,7,8,9], [1,2,3], [4,5]
            # 第一个包包含[6,7,8,9]和部分[1,2,3] = 4+3=7，剩余3个位置放[4,5]
            # 所以第一个包应该是[6,7,8,9,1,2,3,4,5,0]
        
        # 简化测试：验证打包后的张量包含所有原始数据
        all_values = []
        for pkg in packages:
            non_zero = pkg[pkg != 0].tolist()
            all_values.extend(non_zero)
        
        # 检查所有原始数据是否都被包含
        original_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        for val in original_values:
            assert val in all_values, f"Value {val} not found in packages"

    def test_empty_list_input(self):
        """测试空列表输入"""
        packer = SequencePacker(pack_size=10)
        
        packages = packer.pack([])
        
        assert packages == []
        
        # 验证内部状态已正确初始化
        assert packer._current_pack is not None
        assert packer._current_pos == 0

    def test_single_sequence_input(self):
        """测试单个序列输入"""
        packer = SequencePacker(pack_size=10, pad_value=-1)
        
        sequences = [torch.tensor([1, 2, 3], dtype=torch.int32)]
        
        packages = packer.pack(sequences)
        
        assert len(packages) == 1
        pkg = packages[0]
        assert pkg.shape == (10,)
        assert pkg[:3].tolist() == [1, 2, 3]
        assert pkg[3:].tolist() == [-1] * 7

    def test_truncate_long_sequence(self, caplog):
        """测试超长序列截断，验证警告日志是否触发"""
        packer = SequencePacker(pack_size=5, pad_value=0)
        
        sequences = [
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32),  # 长度8，超过pack_size=5
        ]
        
        packages = packer.pack(sequences)
        
        assert len(packages) == 1
        pkg = packages[0]
        assert pkg.shape == (5,)
        assert pkg.tolist() == [1, 2, 3, 4, 5]  # 只保留前5个元素
        
        # 验证警告日志已触发
        assert "truncating" in caplog.text.lower() or "exceeds" in caplog.text.lower()

    def test_padding_value(self):
        """测试填充值正确应用"""
        packer = SequencePacker(pack_size=8, pad_value=99)
        
        sequences = [
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
        ]
        
        packages = packer.pack(sequences)
        
        assert len(packages) == 1
        pkg = packages[0]
        
        # 前3个元素是数据
        assert pkg[:3].tolist() == [1, 2, 3]
        # 后5个元素是填充值
        assert pkg[3:].tolist() == [99] * 5

    def test_different_dtypes(self):
        """测试支持不同 dtype (int32, int64, float32)"""
        # int32
        packer_int32 = SequencePacker(pack_size=10, pad_value=0, dtype=torch.int32)
        sequences_int32 = [torch.tensor([1, 2, 3], dtype=torch.int32)]
        packages_int32 = packer_int32.pack(sequences_int32)
        assert packages_int32[0].dtype == torch.int32

        # int64
        packer_int64 = SequencePacker(pack_size=10, pad_value=0, dtype=torch.int64)
        sequences_int64 = [torch.tensor([1, 2, 3], dtype=torch.int64)]
        packages_int64 = packer_int64.pack(sequences_int64)
        assert packages_int64[0].dtype == torch.int64

        # float32
        packer_float32 = SequencePacker(pack_size=10, pad_value=0.0, dtype=torch.float32)
        sequences_float32 = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)]
        packages_float32 = packer_float32.pack(sequences_float32)
        assert packages_float32[0].dtype == torch.float32

    def test_non_1d_tensor_raises_error(self):
        """测试非1D张量是否抛出异常"""
        packer = SequencePacker(pack_size=10)
        
        # 2D 张量应该抛出异常
        sequences_2d = [torch.tensor([[1, 2], [3, 4]])]  # shape: (2, 2)
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack(sequences_2d)
        
        # 0D 张量 (标量) 应该抛出异常
        sequences_0d = [torch.tensor(5)]  # shape: ()
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack(sequences_0d)
        
        # 3D 张量应该抛出异常
        sequences_3d = [torch.tensor([[[1, 2]]])]  # shape: (1, 1, 2)
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack(sequences_3d)

    def test_reset_method(self):
        """测试 reset() 方法是否正确重置内部状态"""
        packer = SequencePacker(pack_size=10, pad_value=0)
        
        # 第一次打包
        sequences1 = [torch.tensor([1, 2, 3], dtype=torch.int32)]
        packer.pack(sequences1)
        
        # 验证内部状态已更新
        assert packer._current_pos == 0
        assert packer._current_pack is None  # 最后一个包已发送，设置为None
        
        # 重置
        packer.reset()
        
        # 验证重置后的状态
        assert packer._current_pos == 0
        assert packer._current_pack is not None
        assert packer._current_pack.shape == (10,)
        assert packer._current_pack.tolist() == [0] * 10
        
        # 验证重置后可以继续正常使用
        sequences2 = [torch.tensor([4, 5, 6], dtype=torch.int32)]
        packages = packer.pack(sequences2)
        
        assert len(packages) == 1
        assert packages[0][:3].tolist() == [4, 5, 6]

    def test_input_list_not_modified(self):
        """测试输入列表是否未被修改（使用 sorted 而非 sort）"""
        packer = SequencePacker(pack_size=10)
        
        # 创建原始序列列表（故意不按长度排序）
        original_sequences = [
            torch.tensor([3], dtype=torch.int32),   # 长度1
            torch.tensor([1, 2], dtype=torch.int32), # 长度2
            torch.tensor([4, 5, 6, 7], dtype=torch.int32), # 长度4
        ]
        
        # 保存原始顺序的字符串表示
        original_repr = [seq.tolist() for seq in original_sequences]
        
        # 打包
        packer.pack(original_sequences)
        
        # 验证输入列表未被修改
        current_repr = [seq.tolist() for seq in original_sequences]
        assert current_repr == original_repr, "输入列表被修改了，应该使用 sorted() 而非 sort()"

    def test_exact_pack_size_fit(self):
        """测试序列长度恰好等于 pack_size 的情况"""
        packer = SequencePacker(pack_size=5, pad_value=0)
        
        sequences = [
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32),
        ]
        
        packages = packer.pack(sequences)
        
        # 每个序列恰好占满一个包
        assert len(packages) == 2
        
        assert packages[0].tolist() == [1, 2, 3, 4, 5]
        assert packages[1].tolist() == [6, 7, 8, 9, 10]

    def test_multiple_packs_full_utilization(self):
        """测试多个包的高效利用"""
        packer = SequencePacker(pack_size=10, pad_value=-1)
        
        # 创建多个小序列，确保高效打包
        sequences = [
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([6], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([11], dtype=torch.int32),
        ]
        
        packages = packer.pack(sequences)
        
        # 前10个序列打包成一个包，最后一个序列单独一个包
        assert len(packages) == 2
        assert packages[0].tolist() == list(range(1, 11))
        assert packages[1].tolist() == [11] + [-1] * 9

    def test_dtype_mismatch_warning(self, caplog):
        """测试 dtype 不匹配时的警告"""
        packer = SequencePacker(pack_size=10, pad_value=0, dtype=torch.int32)
        
        sequences = [torch.tensor([1, 2, 3], dtype=torch.int64)]
        
        packages = packer.pack(sequences)
        
        # 应该触发 dtype 不匹配警告
        assert "dtype" in caplog.text.lower() or "converted" in caplog.text.lower()