"""将 JSONL 文件 tokenize 后打包存储为 HDF5"""
import json
import os
from typing import List
from pathlib import Path

from tqdm import tqdm

from .processors import BaseProcessor
from .packing import SequencePacker
from .io import IOHandler


def cache_jsonl(
    files: List[str],
    output_dir: str,
    processor: BaseProcessor,
    *,
    pack_size: int = -1,
    pad_value: int = 1,
) -> List[str]:
    """
    将 JSONL 文件 tokenize 后打包存储为 HDF5。

    Args:
        files: JSONL 文件路径列表
        output_dir: H5 输出目录
        processor: 已初始化的 Processor 实例
        pack_size: 打包长度，<=0 表示不打包
        pad_value: 填充值

    Returns:
        生成的 H5 文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files: List[str] = []

    for file_path in files:
        file_name = Path(file_path).stem

        arrows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {file_name}", leave=False):
                arrow = processor.process(json.loads(line))
                if arrow is not None:
                    arrows.append(arrow)

        package = {key: [a[key] for a in arrows] for key in processor.output_keys}

        output = {}
        for key in processor.output_keys:
            if pack_size > 0:
                packer = SequencePacker(pack_size, pad_value)  # 每个键独立实例
                output[key] = packer.pack(package[key])
            else:
                output[key] = package[key]

        IOHandler.save_h5(output_dir, file_name, output)
        h5_path = os.path.join(output_dir, f"{file_name}.h5")
        output_files.append(h5_path)
        print(f"Saved {h5_path}")

    return output_files
