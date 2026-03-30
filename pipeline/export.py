"""Export HuggingFace Dataset to JSONL files in chunks."""
import json
import os
import logging
from typing import Callable, Optional, List, Union, Dict, Any

from datasets import Dataset
from .utils import error_handler

logger = logging.getLogger(__name__)


@error_handler()
def export_dataset(
    dataset: Dataset,
    output_dir: str,
    output_prefix: str,
    *,
    chunk_size: int = 1_000_000,
    max_chunks: Optional[int] = None,
    process_func: Optional[Callable[[Dict[str, Any]], Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    column: str = "text",
) -> List[str]:
    """
    Export HuggingFace Dataset to JSONL files in chunks.

    Args:
        dataset: HuggingFace Dataset object
        output_dir: Output directory
        output_prefix: Output file name prefix, e.g., "chinese-c4-pretrain"
        chunk_size: Maximum number of samples per file
        max_chunks: Maximum number of chunks to process (for debugging)
        process_func: Single sample transformation function (dict) -> dict | list[dict]
        column: Default text column name (only used when process_func is None)

    Returns:
        List of generated file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(dataset)
    num_chunks = (total + chunk_size - 1) // chunk_size
    lim = min(max_chunks, num_chunks) if max_chunks else num_chunks

    output_files: List[str] = []
    for i in range(lim):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = dataset.select(range(start, end))

        path = os.path.join(output_dir, f"{output_prefix}_chunk_{i}.jsonl")
        try:
            with open(path, "w", encoding="utf-8") as f:
                for example in chunk:
                    processed = process_func(example) if process_func else {column: example[column]}
                    items = processed if isinstance(processed, list) else [processed]
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_files.append(path)
            logger.info(f"[{i + 1}/{lim}] Saved {path}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to write chunk {i} to {path}: {e}")

    return output_files
