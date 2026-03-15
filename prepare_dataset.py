#!/usr/bin/env python3
"""
prepare_dataset.py - JSONL Dataset Shard Preparator for CPT

Purpose:
- Append only EOS token `<|endoftext|>` to each text (idempotent).
- Distribute lines across deterministic shards (approx target size) using stable hashing.
- Stream large files without loading into RAM.

Example usage (activate env first):
  conda activate tr_mamba_cpt
  python prepare_dataset.py --input_dir /path/to/jsonl --output_dir /path/to/out --target_mb 300 --seed 42
"""

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import shutil

EOS_TOKEN = "<|endoftext|>"


def scan_input_files(input_dir: Path):
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".jsonl":
            files.append(p)
    return files


def stable_shard_id(line_or_text: str, seed: int, shard_count: int) -> int:
    # Deterministic hash using blake2b; avoid Python built-in hash().
    h = hashlib.blake2b(digest_size=8)
    h.update(f"{seed}\n".encode("utf-8"))
    h.update(line_or_text.encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest(), "big") % shard_count


def ensure_eos(text: str) -> str:
    # Only append EOS once; enforce a single space before EOS.
    base = text.rstrip()
    if base.endswith(EOS_TOKEN):
        return base
    return f"{base} {EOS_TOKEN}"


class ShardWriterCache:
    def __init__(self, output_dir: Path, shard_count: int, max_open_files: int, tmp_dir: Path):
        self.output_dir = output_dir
        self.shard_count = shard_count
        self.max_open_files = max_open_files
        self.tmp_dir = tmp_dir
        self._cache = OrderedDict()

    def _shard_tmp_path(self, shard_id: int) -> Path:
        return self.tmp_dir / f"shard-{shard_id:05d}.jsonl.tmp"

    def get_handle(self, shard_id: int):
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        if len(self._cache) >= self.max_open_files:
            _, fh = self._cache.popitem(last=False)
            fh.close()

        path = self._shard_tmp_path(shard_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "ab", buffering=1024 * 1024)
        self._cache[shard_id] = fh
        return fh

    def close_all(self):
        for fh in self._cache.values():
            fh.close()
        self._cache.clear()


def process_all_files(
    input_files,
    output_dir: Path,
    target_mb: int,
    seed: int,
    text_key: str,
    max_open_files: int,
):
    total_bytes = sum(p.stat().st_size for p in input_files)
    target_bytes = target_mb * 1024 * 1024
    shard_count = max(1, math.ceil(total_bytes / target_bytes))

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / ".tmp_shards"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    shard_sizes = [0 for _ in range(shard_count)]
    cache = ShardWriterCache(output_dir, shard_count, max_open_files, tmp_dir)

    total_lines = 0
    written_lines = 0
    skipped_lines = 0
    eos_added = 0
    missing_text = 0
    non_string_text = 0
    invalid_json = 0

    for p in input_files:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                line = line.rstrip("\n")
                if not line:
                    skipped_lines += 1
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    skipped_lines += 1
                    continue

                if text_key not in obj:
                    missing_text += 1
                    skipped_lines += 1
                    continue

                text_val = obj[text_key]
                if not isinstance(text_val, str):
                    non_string_text += 1
                    text_val = str(text_val)

                new_text = ensure_eos(text_val)
                if new_text != text_val.rstrip():
                    eos_added += 1
                obj[text_key] = new_text

                shard_id = stable_shard_id(new_text, seed, shard_count)
                out_line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
                out_bytes = out_line.encode("utf-8")

                fh = cache.get_handle(shard_id)
                fh.write(out_bytes)

                shard_sizes[shard_id] += len(out_bytes)
                written_lines += 1

    cache.close_all()

    # Finalize: move tmp to final shard names
    final_paths = []
    for shard_id in range(shard_count):
        tmp_path = tmp_dir / f"shard-{shard_id:05d}.jsonl.tmp"
        final_path = output_dir / f"shard-{shard_id:05d}.jsonl"
        if tmp_path.exists():
            tmp_path.replace(final_path)
        else:
            # create empty shard to keep numbering consistent
            open(final_path, "ab").close()
        final_paths.append(final_path)

    return {
        "shard_count": shard_count,
        "shard_sizes": shard_sizes,
        "total_lines": total_lines,
        "written_lines": written_lines,
        "skipped_lines": skipped_lines,
        "eos_added": eos_added,
        "missing_text": missing_text,
        "non_string_text": non_string_text,
        "invalid_json": invalid_json,
        "final_paths": final_paths,
        "target_bytes": target_bytes,
        "tmp_dir": tmp_dir,
    }


def split_oversized_shards(output_dir: Path, shard_sizes, target_bytes, seed: int, oversize_ratio: float):
    new_sizes = {}
    oversized = []

    for shard_id, size in enumerate(shard_sizes):
        if size > oversize_ratio * target_bytes:
            oversized.append((shard_id, size))

    if not oversized:
        return shard_sizes, []

    tmp_dir = output_dir / ".tmp_split"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    split_outputs = []

    for shard_id, size in oversized:
        src_path = output_dir / f"shard-{shard_id:05d}.jsonl"
        part_count = max(2, math.ceil(size / target_bytes))
        part_sizes = [0 for _ in range(part_count)]
        part_paths = [tmp_dir / f"shard-{shard_id:05d}-split-{i:05d}.jsonl.tmp" for i in range(part_count)]
        part_handles = [open(p, "ab", buffering=1024 * 1024) for p in part_paths]

        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                part_id = stable_shard_id(line, seed + shard_id, part_count)
                out_line = line + "\n"
                out_bytes = out_line.encode("utf-8")
                part_handles[part_id].write(out_bytes)
                part_sizes[part_id] += len(out_bytes)

        for fh in part_handles:
            fh.close()

        # Replace original shard with split outputs
        src_path.unlink(missing_ok=True)
        for i, tmp_path in enumerate(part_paths):
            final_path = output_dir / f"shard-{shard_id:05d}-split-{i:05d}.jsonl"
            tmp_path.replace(final_path)
            new_sizes[final_path.name] = part_sizes[i]
            split_outputs.append(final_path)

    shutil.rmtree(tmp_dir)

    # Build updated sizes list for original shards, excluding those split
    updated_sizes = []
    split_ids = {sid for sid, _ in oversized}
    for shard_id, size in enumerate(shard_sizes):
        if shard_id in split_ids:
            updated_sizes.append(0)
        else:
            updated_sizes.append(size)

    return updated_sizes, split_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare JSONL dataset shards with EOS appended and deterministic sharding.",
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing .jsonl files")
    parser.add_argument("--output_dir", required=True, help="Output directory for shards")
    parser.add_argument("--target_mb", type=int, default=300, help="Target shard size in MB")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic hashing")
    parser.add_argument("--text_key", type=str, default="text", help="JSON key for text field")
    parser.add_argument("--split_oversize", action="store_true", help="Split oversized shards")
    parser.add_argument("--oversize_ratio", type=float, default=1.2, help="Oversize ratio threshold")
    parser.add_argument("--max_open_files", type=int, default=64, help="Max open file handles")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    input_files = scan_input_files(input_dir)
    if not input_files:
        print("No .jsonl files found in input directory.", file=sys.stderr)
        sys.exit(1)

    result = process_all_files(
        input_files=input_files,
        output_dir=output_dir,
        target_mb=args.target_mb,
        seed=args.seed,
        text_key=args.text_key,
        max_open_files=args.max_open_files,
    )

    shard_sizes = result["shard_sizes"]

    split_outputs = []
    if args.split_oversize:
        shard_sizes, split_outputs = split_oversized_shards(
            output_dir=output_dir,
            shard_sizes=shard_sizes,
            target_bytes=result["target_bytes"],
            seed=args.seed,
            oversize_ratio=args.oversize_ratio,
        )

    # Report
    print("\nSummary")
    print(f"Processed input files: {len(input_files)}")
    print(f"Total lines: {result['total_lines']}")
    print(f"Written lines: {result['written_lines']}")
    print(f"Skipped lines: {result['skipped_lines']}")
    print(f"EOS added lines: {result['eos_added']}")
    print(f"Missing text_key ({args.text_key}): {result['missing_text']}")
    print(f"Non-string text values: {result['non_string_text']}")
    print(f"Invalid JSON lines: {result['invalid_json']}")

    print("\nShard sizes (MB):")
    for shard_id, size in enumerate(shard_sizes):
        if size == 0:
            continue
        mb = size / (1024 * 1024)
        print(f"  shard-{shard_id:05d}.jsonl: {mb:.2f} MB")

    if split_outputs:
        print("\nSplit outputs:")
        for p in split_outputs:
            size = p.stat().st_size
            mb = size / (1024 * 1024)
            print(f"  {p.name}: {mb:.2f} MB")


if __name__ == "__main__":
    main()
