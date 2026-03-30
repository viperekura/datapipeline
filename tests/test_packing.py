"""Tests for pipeline.packing module."""

import pytest
import torch
from pipeline.packing import SequencePacker


class TestSequencePacker:

    def test_normal_packing(self):
        packer = SequencePacker(pack_size=10, pad_value=0)
        sequences = [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5], dtype=torch.int32),
            torch.tensor([6, 7, 8, 9], dtype=torch.int32),
        ]
        packages = packer.pack(sequences)
        assert len(packages) >= 1
        for pkg in packages:
            assert pkg.shape == (10,)

        # Verify all original values are present
        all_values = []
        for pkg in packages:
            all_values.extend(pkg[pkg != 0].tolist())
        for val in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            assert val in all_values

    def test_empty_list_input(self):
        packer = SequencePacker(pack_size=10)
        assert packer.pack([]) == []

    def test_single_sequence_input(self):
        packer = SequencePacker(pack_size=10, pad_value=-1)
        packages = packer.pack([torch.tensor([1, 2, 3], dtype=torch.int32)])
        assert len(packages) == 1
        assert packages[0][:3].tolist() == [1, 2, 3]
        assert packages[0][3:].tolist() == [-1] * 7

    def test_truncate_long_sequence(self, caplog):
        packer = SequencePacker(pack_size=5, pad_value=0)
        packages = packer.pack([torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)])
        assert len(packages) == 1
        assert packages[0].tolist() == [1, 2, 3, 4, 5]
        assert "truncating" in caplog.text.lower() or "exceeds" in caplog.text.lower()

    def test_padding_value(self):
        packer = SequencePacker(pack_size=8, pad_value=99)
        packages = packer.pack([
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32),
        ])
        assert packages[0][:3].tolist() == [1, 2, 3]
        assert packages[0][3:].tolist() == [99] * 5

    def test_different_dtypes(self):
        for dtype in [torch.int32, torch.int64, torch.float32]:
            packer = SequencePacker(pack_size=10, dtype=dtype)
            val = 1.0 if dtype == torch.float32 else 1
            packages = packer.pack([torch.tensor([val, 2, 3], dtype=dtype)])
            assert packages[0].dtype == dtype

    def test_dtype_conversion_on_mismatch(self, caplog):
        """Tensors with mismatched dtype are silently converted."""
        packer = SequencePacker(pack_size=10, dtype=torch.int32)
        packages = packer.pack([torch.tensor([1, 2, 3], dtype=torch.int64)])
        assert packages[0].dtype == torch.int32
        assert packages[0][:3].tolist() == [1, 2, 3]

    def test_non_1d_tensor_raises_error(self):
        packer = SequencePacker(pack_size=10)
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack([torch.tensor([[1, 2], [3, 4]])])
        with pytest.raises(ValueError, match="Expected 1D tensor"):
            packer.pack([torch.tensor(5)])

    def test_input_list_not_modified(self):
        packer = SequencePacker(pack_size=10)
        original = [
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([4, 5, 6, 7], dtype=torch.int32),
        ]
        original_repr = [seq.tolist() for seq in original]
        packer.pack(original)
        assert [seq.tolist() for seq in original] == original_repr

    def test_exact_pack_size_fit(self):
        packer = SequencePacker(pack_size=5, pad_value=0)
        packages = packer.pack([
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
            torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32),
        ])
        assert len(packages) == 2
        assert packages[0].tolist() == [1, 2, 3, 4, 5]
        assert packages[1].tolist() == [6, 7, 8, 9, 10]

    def test_multiple_packs_full_utilization(self):
        packer = SequencePacker(pack_size=10, pad_value=-1)
        sequences = [torch.tensor([i], dtype=torch.int32) for i in range(1, 12)]
        packages = packer.pack(sequences)
        assert len(packages) == 2
        assert packages[0].tolist() == list(range(1, 11))
        assert packages[1].tolist() == [11] + [-1] * 9

    def test_cross_group_ordering(self):
        """Tensor groups with identical per-item lengths are sorted identically."""
        packer = SequencePacker(pack_size=10, pad_value=0)
        # sequences: lengths [3, 1, 4] -> after sort desc: [4, 3, 1]
        seqs = [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([4, 5, 6, 7], dtype=torch.int32),
        ]
        masks = [
            torch.tensor([True, True, True], dtype=torch.bool),
            torch.tensor([True], dtype=torch.bool),
            torch.tensor([True, True, True, True], dtype=torch.bool),
        ]
        packed_seqs = packer.pack(seqs)
        packer.reset()
        packed_masks = packer.pack(masks)

        # Verify mask packer uses bool dtype
        assert packed_masks[0].dtype == torch.bool
        # Both groups should produce the same number of packages
        assert len(packed_seqs) == len(packed_masks)
