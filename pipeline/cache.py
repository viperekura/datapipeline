"""Tokenize JSONL files and pack them into HDF5 storage."""
import json
import os
import logging
from typing import List
from pathlib import Path

from tqdm import tqdm

from .processors import BaseProcessor
from .packing import SequencePacker
from .io import IOHandler
from .utils import error_handler

logger = logging.getLogger(__name__)


@error_handler()
def cache_jsonl(
    files: List[str],
    output_dir: str,
    processor: BaseProcessor,
    *,
    pack_size: int = -1,
    pad_value: int = 1,
) -> List[str]:
    """
    Tokenize JSONL files and pack them into HDF5 storage.

    Args:
        files: List of JSONL file paths
        output_dir: H5 output directory
        processor: Initialized Processor instance
        pack_size: Packing length, <=0 means no packing
        pad_value: Padding value

    Returns:
        List of generated H5 file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files: List[str] = []

    for file_path in files:
        file_name = Path(file_path).stem

        arrows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(tqdm(f, desc=f"Processing {file_name}", leave=False), start=1):
                try:
                    arrow = processor.process(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {file_path} line {line_num}: {e}. Skipping line.")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error processing line {line_num} in {file_path}: {e}. Skipping line.")
                    continue
                if arrow is not None:
                    arrows.append(arrow)

        package = {key: [a[key] for a in arrows] for key in processor.output_keys}

        output = {}
        for key in processor.output_keys:
            if pack_size > 0:
                packer = SequencePacker(pack_size, pad_value)  # independent instance per key
                output[key] = packer.pack(package[key])
            else:
                output[key] = package[key]

        IOHandler.save_h5(output_dir, file_name, output)
        h5_path = os.path.join(output_dir, f"{file_name}.h5")
        output_files.append(h5_path)
        logger.info(f"Saved {h5_path}")

    return output_files
