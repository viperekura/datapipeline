import logging
from typing import List
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class SequencePacker:

    def __init__(self, pack_size: int, pad_value: int = 0, dtype=torch.int32):
        self.pack_size = pack_size
        self.pad_value = pad_value
        self.dtype = dtype
        self._reset()

    def _reset(self) -> None:
        """Reset internal state for instance reuse."""
        self._current_pack = torch.full(
            (self.pack_size,), self.pad_value, dtype=self.dtype
        )
        self._current_pos = 0

    def pack(self, sequences: List[Tensor]) -> List[Tensor]:
        """
        Pack sequences into fixed-size packages.

        Args:
            sequences: List of input tensors

        Returns:
            List of packed tensors, each with length equal to pack_size
        """
        # Input validation
        if not sequences:
            return []
        for i, seq in enumerate(sequences):
            if seq.dim() != 1:
                raise ValueError(
                    f"Expected 1D tensor at index {i}, got {seq.dim()}D tensor with shape {seq.shape}"
                )
            # Check dtype compatibility and warn if mismatched
            if seq.dtype != self.dtype:
                logger.warning(
                    f"Input tensor dtype {seq.dtype} does not match packer dtype {self.dtype}, "
                    f"will be converted. This may affect packing efficiency."
                )

        packages = []
        # Sort by length in descending order to improve packing efficiency
        # Use sorted() to avoid modifying the input list
        sorted_sequences = sorted(sequences, key=lambda x: x.numel(), reverse=True)

        for tensor in sorted_sequences:
            # Truncate sequences that exceed pack_size
            if tensor.numel() > self.pack_size:
                logger.warning(
                    f"Sequence length {tensor.numel()} exceeds pack_size {self.pack_size}, truncating"
                )
                tensor = tensor[: self.pack_size]
            tensor_size = tensor.numel()

            # Current package is full, create a new one
            if self._current_pos + tensor_size > self.pack_size:
                packages.append(self._current_pack)
                self._current_pack = torch.full(
                    (self.pack_size,), self.pad_value, dtype=self.dtype
                )
                self._current_pos = 0

            # Place tensor in current package (remaining positions stay as pad_value)
            self._current_pack[self._current_pos : self._current_pos + tensor_size] = (
                tensor
            )
            self._current_pos += tensor_size

        # Handle the last package
        if self._current_pos > 0:
            packages.append(self._current_pack)
            self._current_pack = None
            self._current_pos = 0

        return packages

    def reset(self) -> None:
        """Reset packer state for reuse. More efficient than creating a new instance."""
        self._reset()