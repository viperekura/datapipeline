import logging
from typing import List
import torch
from torch import Tensor
from pipeline.utils import error_handler

logger = logging.getLogger(__name__)


class SequencePacker:
    """
    Stream-concatenation packer for LLM training sequences.

    Algorithm (streaming concat):

        Input:  sequences = [A(len=3), B(len=5), C(len=2)], pack_size = 6

        1. Validate & Normalize
           - Check 1D dimension, unify dtype, warn on overlong sequences
           - Result: [A, B, C]

        2. Stream into buffer, slice off full chunks
           - buffer += A(3) -> [a1 a2 a3], pos=3
           - buffer += B(5) -> [a1 a2 a3 b1 b2 b3 b4 b5], pos=8
             pos >= 6 -> flush [a1 a2 a3 b1 b2 b3], buffer=[b4 b5], pos=2
           - buffer += C(2) -> [b4 b5 c1 c2], pos=4
             loop ends -> flush tail [b4 b5 c1 c2 PAD PAD]

        Output: [[a1 a2 a3 b1 b2 b3], [b4 b5 c1 c2 PAD PAD]]

    Samples may be split across chunks — this is intentional and standard
    practice in LLM training (TRL, Megatron-LM, etc.).

    Cross-group consistency:
        Different tensor groups (e.g. input_ids, loss_masks) packed with
        separate packer instances on samples with matching lengths produce
        identical chunk boundaries. Element-level correspondence is preserved.
    """

    def __init__(
        self, pack_size: int, pad_value: int = 0, dtype: torch.dtype = torch.int32
    ):
        self.pack_size = pack_size
        self.pad_value = pad_value
        self.dtype = dtype
        self._buffer: List[int] = []
        self._pos: int = 0
        self._packages: List[Tensor] = []

    def reset(self) -> None:
        """Reset packer state for instance reuse."""
        self._buffer = []
        self._pos = 0
        self._packages = []

    @error_handler()
    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        """
        Pack sequences via streaming concatenation into fixed-size chunks.

        Sequences are concatenated in order and sliced at pack_size boundaries.
        The final chunk is padded with pad_value.

        Args:
            sequences: List of 1D input tensors.

        Returns:
            List of packed tensors, each with length equal to pack_size.
        """
        if not sequences:
            return []

        # --- validate & normalize ---
        normalized: List[Tensor] = []
        for i, seq in enumerate(sequences):
            if seq.dim() != 1:
                raise ValueError(
                    f"Expected 1D tensor at index {i}, got {seq.dim()}D tensor with shape {seq.shape}"
                )
            if seq.dtype != self.dtype:
                seq = seq.to(self.dtype)
            normalized.append(seq)

        # --- stream into buffer, slice off full chunks ---
        self._buffer = []
        self._packages = []
        pack_size = self.pack_size
        buf = self._buffer

        for seq in normalized:
            buf.extend(seq.tolist())
            while len(buf) >= pack_size:
                self._packages.append(torch.tensor(buf[:pack_size], dtype=self.dtype))
                buf = buf[pack_size:]

        # flush tail with padding
        if buf:
            padded = buf + [self.pad_value] * (pack_size - len(buf))
            self._packages.append(torch.tensor(padded, dtype=self.dtype))

        self._pos = len(buf)
        return self._packages
