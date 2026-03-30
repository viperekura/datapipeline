import logging
from typing import List, Optional
import torch
from torch import Tensor

from .utils import error_handler

logger = logging.getLogger(__name__)


class SequencePacker:

    def __init__(self, pack_size: int, pad_value: int = 0, dtype: torch.dtype = torch.int32):
        self.pack_size = pack_size
        self.pad_value = pad_value
        self.dtype = dtype
        # Pre-allocate buffer for better performance
        self._buffer: Optional[Tensor] = None
        self._reset()

    def _reset(self) -> None:
        """Reset internal state for instance reuse."""
        # Reuse buffer instead of creating new tensors
        if self._buffer is None or self._buffer.shape[0] != self.pack_size:
            self._buffer = torch.full(
                (self.pack_size,), self.pad_value, dtype=self.dtype
            )
        else:
            self._buffer.fill_(self.pad_value)
        self._current_pos = 0
        self._packages: List[Tensor] = []
        # Backward compatibility: maintain _current_pack reference
        self._current_pack = self._buffer

    @error_handler()
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
        
        # Validate and cache tensor sizes in one pass
        tensor_sizes = []
        for i, seq in enumerate(sequences):
            if seq.dim() != 1:
                raise ValueError(
                    f"Expected 1D tensor at index {i}, got {seq.dim()}D tensor with shape {seq.shape}"
                )
            tensor_sizes.append(seq.numel())
            if seq.dtype != self.dtype:
                logger.warning(
                    f"Input tensor dtype {seq.dtype} does not match packer dtype {self.dtype}, "
                    f"will be converted. This may affect packing efficiency."
                )

        # Reset state for new packing
        self._packages = []
        self._reset()
        
        # Combine sequences with their sizes for sorting
        indexed_seqs = list(zip(sequences, tensor_sizes))
        # Sort by size descending (First-Fit Decreasing algorithm)
        indexed_seqs.sort(key=lambda x: x[1], reverse=True)

        for tensor, tensor_size in indexed_seqs:
            # Truncate sequences that exceed pack_size
            if tensor_size > self.pack_size:
                logger.warning(
                    f"Sequence length {tensor_size} exceeds pack_size {self.pack_size}, truncating"
                )
                tensor_size = self.pack_size
                tensor = tensor[: self.pack_size]

            # Current package is full, create a new one
            if self._current_pos + tensor_size > self.pack_size:
                # Finish current package (pad to pack_size)
                package = self._buffer.clone()
                self._packages.append(package)
                # Reset buffer for reuse
                self._buffer.fill_(self.pad_value)
                self._current_pos = 0

            # Place tensor in current package
            self._buffer[self._current_pos : self._current_pos + tensor_size] = tensor
            self._current_pos += tensor_size

        # Handle the last package (pad to pack_size)
        if self._current_pos > 0:
            package = self._buffer.clone()
            self._packages.append(package)

        # Clear buffer and reset state for backward compatibility
        self._buffer = None
        self._current_pack = None
        self._current_pos = 0
        
        return self._packages

    def reset(self) -> None:
        """Reset packer state for reuse. More efficient than creating a new instance."""
        self._reset()