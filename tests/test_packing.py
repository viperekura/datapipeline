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
        assert len(packages) == 1
        for pkg in packages:
            assert pkg.shape == (10,)

        # Verify all original values are present in order
        assert packages[0][:9].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert packages[0][9] == 0  # padding

    def test_empty_list_input(self):
        packer = SequencePacker(pack_size=10)
        assert packer.pack([]) == []

    def test_single_sequence_input(self):
        packer = SequencePacker(pack_size=10, pad_value=-1)
        packages = packer.pack([torch.tensor([1, 2, 3], dtype=torch.int32)])
        assert len(packages) == 1
        assert packages[0][:3].tolist() == [1, 2, 3]
        assert packages[0][3:].tolist() == [-1] * 7

    def test_long_sequence_split_across_chunks(self):
        """Sequences longer than pack_size are split across multiple chunks."""
        packer = SequencePacker(pack_size=5, pad_value=0)
        packages = packer.pack(
            [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)]
        )
        assert len(packages) == 2
        assert packages[0].tolist() == [1, 2, 3, 4, 5]
        assert packages[1].tolist() == [6, 7, 8, 0, 0]

    def test_padding_value(self):
        packer = SequencePacker(pack_size=8, pad_value=99)
        packages = packer.pack(
            [
                torch.tensor([1, 2], dtype=torch.int32),
                torch.tensor([3], dtype=torch.int32),
            ]
        )
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
        packages = packer.pack(
            [
                torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
                torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32),
            ]
        )
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
        """Separate packers for different dtypes produce identical chunk boundaries."""
        seq_packer = SequencePacker(pack_size=10, pad_value=0, dtype=torch.int32)
        mask_packer = SequencePacker(pack_size=10, pad_value=False, dtype=torch.bool)
        # sequences: lengths [3, 1, 4]
        seqs = [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
            torch.tensor([4, 5, 6, 7], dtype=torch.int32),
        ]
        masks = [
            torch.tensor([False, False, True], dtype=torch.bool),
            torch.tensor([False], dtype=torch.bool),
            torch.tensor([False, False, False, True], dtype=torch.bool),
        ]
        packed_seqs = seq_packer.pack(seqs)
        packed_masks = mask_packer.pack(masks)

        # Verify mask packer uses bool dtype
        assert packed_masks[0].dtype == torch.bool
        # Both groups should produce the same number of packages
        assert len(packed_seqs) == len(packed_masks)

    def test_stream_split_across_chunks(self):
        """Sequences are split across chunks in streaming mode."""
        packer = SequencePacker(pack_size=5, pad_value=0)
        packages = packer.pack(
            [
                torch.tensor([1, 2, 3], dtype=torch.int32),
                torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32),
            ]
        )
        assert len(packages) == 2
        # First chunk: [1, 2, 3, 4, 5] — first seq + part of second
        assert packages[0].tolist() == [1, 2, 3, 4, 5]
        # Second chunk: [6, 7, 8, 0, 0] — rest of second + padding
        assert packages[1].tolist() == [6, 7, 8, 0, 0]

    def test_reset_method(self):
        packer = SequencePacker(pack_size=10, pad_value=0)
        seqs = [torch.tensor([1, 2, 3], dtype=torch.int32)]
        packer.pack(seqs)
        assert len(packer._packages) == 1
        packer.reset()
        assert len(packer._packages) == 0
        assert packer._pos == 0
        assert packer._buffer == []

    def test_no_sorting_needed(self):
        """Streaming concat preserves input order, no sorting."""
        packer = SequencePacker(pack_size=4, pad_value=-1)
        # short then long (fits in 2 chunks)
        packages = packer.pack(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.int32),
            ]
        )
        assert packages[0].tolist() == [1, 2, 3, 4]
        assert packages[1].tolist() == [5, 6, 7, -1]
