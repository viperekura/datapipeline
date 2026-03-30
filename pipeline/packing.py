import logging
from typing import List
import torch
from torch import Tensor
from pipeline.utils import error_handler

logger = logging.getLogger(__name__)


class SequencePacker:
    """
    Packs variable-length sequences into fixed-size tensors, suitable for
    concatenating unequal-length training samples into uniform shapes
    for DataLoader / model training.

    Algorithm (Sorted Greedy Fill, based on First-Fit Decreasing heuristic):

        Input:  sequences = [A(len=5), B(len=2), C(len=3)], pack_size = 8

        1. Validate & Normalize
           - Check 1D dimension, unify dtype, truncate overlong sequences with warning
           - Result: [(A,5), (B,2), (C,3)]

        2. Sort by length descending (FFD)
           - Result: [(A,5), (C,3), (B,2)]

        3. Greedy fill: write into a pre-allocated buffer sequentially, flush when full
           - Write A(5) -> buffer = [A A A A A _ _ _], pos=5
           - Write C(3) -> pos+3=8 <= 8 -> buffer = [A A A A A C C C], pos=8
           - Buffer full  -> flush as package[0], reset buffer & pos=0
           - Write B(2) -> buffer = [B B _ _ _ _ _ _], pos=2
           - Loop ends   -> flush tail -> package[1] = [B B 0 0 0 0 0 0]

        Output: [package[0], package[1]]

    Cross-group consistency:
        When packing different key groups (e.g. sequences and loss_masks)
        with separate pack() calls, tensors at the same index always have
        identical lengths, so the descending sort produces the exact same
        ordering. Element-level correspondence across groups is preserved.

    Performance:
        - Pre-allocated buffer reused via fill_() to avoid repeated tensor creation
        - Attributes cached as local variables inside the loop to reduce lookup overhead
    """

    def __init__(self, pack_size: int, pad_value: int = 0, dtype: torch.dtype = None):
        self.pack_size = pack_size
        self.pad_value = pad_value
        self.dtype = dtype  # None = follow input dtype
        self._buffer: Tensor | None = None
        self._pos = 0
        self._packages: List[Tensor] = []

    def reset(self) -> None:
        """Reset packer state for instance reuse, unlocking dtype."""
        self.dtype = None
        self._buffer = None
        self._pos = 0
        self._packages = []

    @error_handler()
    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        """
        Pack sequences into fixed-size packages using First-Fit Decreasing.

        Sequences are sorted by length descending to minimize wasted padding.
        All tensor groups (e.g. sequences, loss_masks) with matching per-item
        lengths produce identical ordering, so cross-group correspondence is preserved.

        Args:
            sequences: List of 1D input tensors.

        Returns:
            List of packed tensors, each with length equal to pack_size.
        """
        if not sequences:
            return []

        # --- validate & normalize in a single pass ---
        normalized: list[tuple[Tensor, int]] = []
        target_dtype = self.dtype if self.dtype is not None else sequences[0].dtype
        for i, seq in enumerate(sequences):
            if seq.dim() != 1:
                raise ValueError(
                    f"Expected 1D tensor at index {i}, got {seq.dim()}D tensor with shape {seq.shape}"
                )
            if seq.dtype != target_dtype:
                seq = seq.to(target_dtype)
            length = seq.numel()
            if length > self.pack_size:
                seq = seq[: self.pack_size]
                length = self.pack_size
            normalized.append((seq, length))

        # --- reset internal state ---
        buf = self._buffer
        if buf is None or buf.dtype != target_dtype:
            buf = torch.full((self.pack_size,), self.pad_value, dtype=target_dtype)
            self._buffer = buf
        buf.fill_(self.pad_value)
        self._pos = 0
        self._packages = []

        # --- sort by length descending (FFD heuristic) ---
        normalized.sort(key=lambda x: x[1], reverse=True)

        # --- greedy fill ---
        buf = self._buffer
        pos = self._pos
        packages = self._packages
        pack_size = self.pack_size
        pad_value = self.pad_value

        for tensor, length in normalized:
            if pos + length > pack_size:
                # flush current package
                packages.append(buf.clone())
                buf.fill_(pad_value)
                pos = 0
            buf[pos : pos + length] = tensor
            pos += length

        # flush the last (possibly partial) package
        if pos > 0:
            packages.append(buf.clone())

        # write back state
        self._pos = pos
        return self._packages
