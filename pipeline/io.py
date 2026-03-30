from pathlib import Path
from typing import Dict, List, Optional, Callable
import os
import h5py
import torch
from torch import Tensor

from .utils import error_handler


class IOHandler:
    """File and HDF5 read/write operations."""

    @staticmethod
    def fetch_files(directory: str) -> List[str]:
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files
        ]

    @staticmethod
    def fetch_folders(root_dir: str, filter_func: Optional[Callable[[str], bool]] = None) -> List[str]:
        folders = []
        for root, dirs, _ in os.walk(root_dir):
            for dir_name in dirs:
                folder_path = os.path.join(root, dir_name)
                if filter_func is None or filter_func(folder_path):
                    folders.append(folder_path)
        return folders

    @staticmethod
    @error_handler()
    def save_h5(output_dir: str, file_name: str, tensor_group: Dict[str, List[Tensor]]) -> None:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, f"{file_name}.h5")
        with h5py.File(full_path, 'w') as f:
            for key, tensors in tensor_group.items():
                grp = f.create_group(key)
                for idx, tensor in enumerate(tensors):
                    grp.create_dataset(f'data_{idx}', data=tensor.cpu().numpy())

    @staticmethod
    @error_handler()
    def load_h5(file_path: str, share_memory=True) -> Dict[str, List[Tensor]]:
        tensor_group: Dict[str, List[Tensor]] = {}

        root_path = Path(file_path)
        h5_files = list(root_path.rglob("*.h5")) + list(root_path.rglob("*.hdf5"))
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                for key in f.keys():
                    grp = f[key]
                    dsets = []
                    for dset_name in grp.keys():
                        dset = grp[dset_name]
                        tensor = torch.from_numpy(dset[:])
                        if share_memory:
                            tensor = tensor.share_memory_()
                        dsets.append(tensor)
                
                    if tensor_group.get(key) is None:
                        tensor_group[key] = []
                    tensor_group[key].extend(dsets)

        return tensor_group