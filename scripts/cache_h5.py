"""JSONL to H5 caching script.

Tokenize JSONL files and pack them into HDF5 format.

Usage:
    python scripts/cache_h5.py pt ./dataset/chinese-c4-pretrain
    python scripts/cache_h5.py sft ./dataset/belle-sft --pack-size 4096 --strategy alpaca
    python scripts/cache_h5.py sft ./dataset/Ling-Coder-sft --tokenizer ./my_tokenizer.json
"""

import argparse
import os

from pipeline import AutoTokenizer, ProcessorFactory, cache_jsonl
from pipeline.io import IOHandler


def main():
    parser = argparse.ArgumentParser(description="JSONL -> H5 cache")
    parser.add_argument("type", choices=["pt", "sft", "dpo"], help="Processor type")
    parser.add_argument("input_dir", help="Directory containing JSONL files")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="H5 output dir (default: <input_dir>/cached)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="./tokenizer.json",
        help="Tokenizer path (default: ./tokenizer.json)",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        default=None,
        help="Prompt strategy: chatml, alpaca (default: chatml)",
    )
    parser.add_argument(
        "-p",
        "--pack-size",
        type=int,
        default=-1,
        help="Pack size, <=0 to disable (default: -1)",
    )
    parser.add_argument(
        "--pad-value", type=int, default=1, help="Padding value (default: 1)"
    )
    args = parser.parse_args()

    jsonl_files = IOHandler.fetch_files(args.input_dir, suffix=".jsonl")
    if not jsonl_files:
        print(f"[ERROR] No JSONL files found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f}")

    if not os.path.exists(args.tokenizer):
        print(f"[ERROR] Tokenizer not found: {args.tokenizer}")
        return
    tokenizer = AutoTokenizer(args.tokenizer)
    print(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    if args.strategy:
        processor = ProcessorFactory.create_with_strategy_name(
            args.type, tokenizer, args.strategy
        )
    else:
        processor = ProcessorFactory.create(args.type, tokenizer)

    print(f"Processor: {args.type} ({processor.__class__.__name__})")
    print(f"Output keys: {processor.output_keys}")

    output_dir = args.output_dir or os.path.join(args.input_dir, "cached")

    print(f"\nStart caching...")
    if args.pack_size > 0:
        print(f"  pack_size={args.pack_size}, pad_value={args.pad_value}")
    else:
        print(f"  no packing")

    cache_jsonl(
        files=jsonl_files,
        output_dir=output_dir,
        processor=processor,
        pack_size=args.pack_size,
        pad_value=args.pad_value,
    )
    print(f"\nDone! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
