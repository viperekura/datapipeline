"""HDF5 storage operations for tensor data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import torch
from torch import Tensor

from pipeline.utils import error_handler

logger = logging.getLogger(__name__)


class HDF5Handler:
    """Handler for reading and writing tensor data to HDF5 files.

    Example::

        handler = HDF5Handler()
        handler.save(output_dir, "data", {"input_ids": [tensor1, tensor2]})

        loaded = handler.load("./output/data.h5")
        for tensor in loaded["input_ids"]:
            print(tensor.shape)
    """

    @staticmethod
    @error_handler()
    def save(
        output_dir: str,
        file_name: str,
        tensor_group: Dict[str, List[Tensor]],
        extension: str = ".h5",
    ) -> str:
        """Save tensor groups to HDF5 file.

        Args:
            output_dir: Output directory path.
            file_name: Base name for the output file (without extension).
            tensor_group: Dictionary mapping group names to tensor lists.
            extension: File extension (default: ".h5").

        Returns:
            Path to the saved file.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{file_name}{extension}")

        with h5py.File(full_path, "w") as f:
            for key, tensors in tensor_group.items():
                grp = f.create_group(key)
                for idx, tensor in enumerate(tensors):
                    grp.create_dataset(f"data_{idx}", data=tensor.cpu().numpy())

        logger.info(f"Saved HDF5 file: {full_path}")
        return full_path

    @staticmethod
    @error_handler()
    def load(
        file_path: str,
        share_memory: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[Tensor]]:
        """Load tensor groups from HDF5 file.

        Args:
            file_path: Path to HDF5 file or directory containing HDF5 files.
            share_memory: Whether to use shared memory for tensors.
            device: Target device for tensors (default: CPU).

        Returns:
            Dictionary mapping group names to tensor lists.
        """
        root_path = Path(file_path)
        h5_files = []

        if root_path.is_file():
            h5_files = [root_path]
        else:
            h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))

        if not h5_files:
            logger.warning(f"No HDF5 files found at: {file_path}")
            return {}

        tensor_group: Dict[str, List[Tensor]] = {}

        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                for key in f.keys():
                    grp = f[key]
                    dsets = []
                    for dset_name in grp.keys():
                        dset = grp[dset_name]
                        tensor = torch.from_numpy(dset[:])

                        if device is not None:
                            tensor = tensor.to(device)
                        elif share_memory:
                            tensor = tensor.share_memory_()

                        dsets.append(tensor)

                    if tensor_group.get(key) is None:
                        tensor_group[key] = []
                    tensor_group[key].extend(dsets)

        logger.info(f"Loaded HDF5: {len(tensor_group)} groups, "
                    f"{sum(len(v) for v in tensor_group.values())} total tensors")

        return tensor_group

    @staticmethod
    def get_metadata(file_path: str) -> Dict[str, int]:
        """Get metadata about an HDF5 file without loading full data.

        Args:
            file_path: Path to HDF5 file.

        Returns:
            Dictionary with group names and tensor counts.
        """
        metadata: Dict[str, int] = {}

        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                metadata[key] = len(f[key].keys())

        return metadata
