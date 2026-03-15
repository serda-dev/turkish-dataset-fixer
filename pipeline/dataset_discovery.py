"""
dataset_discovery.py — Recursive dataset file discovery with gz + parquet support.

Walks directories recursively to find .json, .jsonl, .json.gz, .jsonl.gz,
and .parquet files.
Provides transparent decompression for gzipped files and streaming row
iteration for parquet.
"""

import gzip
import json
import logging
import os
from pathlib import Path
from typing import IO, Dict, Generator, List, Optional, Set

logger = logging.getLogger(__name__)

# Supported raw extensions and their gzipped variants
SUPPORTED_EXTENSIONS: Set[str] = {
    '.json', '.jsonl',
    '.json.gz', '.jsonl.gz',
    '.parquet',
}


def _get_double_suffix(path: Path) -> str:
    """Get double suffix for .json.gz style extensions."""
    suffixes = path.suffixes
    if len(suffixes) >= 2:
        return ''.join(suffixes[-2:]).lower()
    return path.suffix.lower()


def is_parquet_file(path: Path) -> bool:
    """Check if a file is a parquet file."""
    return path.suffix.lower() == '.parquet'


def dataset_output_stem(path: Path) -> str:
    """
    Normalize a supported dataset filename to a stable output stem.

    Examples:
      - sample.jsonl -> sample
      - sample.jsonl.gz -> sample
      - sample.parquet -> sample
    """
    path = Path(path)
    name = path.name
    for suffix in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def is_supported_file(path: Path) -> bool:
    """Check if a file has a supported extension."""
    double = _get_double_suffix(path)
    if double in SUPPORTED_EXTENSIONS:
        return True
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_dataset_files(
    root_dir: Path,
    extensions: Optional[Set[str]] = None,
) -> List[Path]:
    """
    Recursively discover dataset files in a directory.

    Args:
        root_dir: Root directory to search.
        extensions: Set of extensions to match (default: SUPPORTED_EXTENSIONS).

    Returns:
        Sorted list of discovered file paths.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    root_dir = Path(root_dir)
    if not root_dir.exists():
        logger.warning("Discovery root does not exist: %s", root_dir)
        return []

    discovered = []

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            double = _get_double_suffix(filepath)
            single = filepath.suffix.lower()

            if double in extensions or single in extensions:
                discovered.append(filepath)

    discovered.sort()

    logger.info(
        "Discovered %d dataset files in %s (extensions: %s)",
        len(discovered), root_dir,
        ', '.join(sorted(extensions)),
    )

    return discovered


def open_data_file(path: Path, encoding: str = 'utf-8') -> IO:
    """
    Open a text-based data file, transparently handling gzip compression.

    For JSONL / JSON / JSON.gz / JSONL.gz files only.
    For parquet, use iterate_records() instead.

    Args:
        path: Path to the file.
        encoding: Text encoding (default: utf-8).

    Returns:
        A file-like object yielding text lines.
    """
    double = _get_double_suffix(path)
    if double.endswith('.gz'):
        return gzip.open(path, 'rt', encoding=encoding, errors='replace')
    return open(path, 'r', encoding=encoding, errors='replace')


def iterate_parquet_records(
    path: Path,
    text_key: str = 'text',
    batch_size: int = 4096,
) -> Generator[Dict, None, None]:
    """
    Stream records from a parquet file in batches (memory-friendly).

    Uses pyarrow to read row groups / record batches without loading the
    entire file into RAM.

    Args:
        path: Path to the .parquet file.
        text_key: Name of the text column (for validation only — all
                  columns are yielded as dict keys).
        batch_size: Number of rows per batch (controls peak RAM).

    Yields:
        dict per row (column_name -> value).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema_names = pf.schema_arrow.names
    logger.debug("Parquet %s: %d row groups, columns=%s",
                 path.name, pf.metadata.num_row_groups, schema_names)

    for batch in pf.iter_batches(batch_size=batch_size):
        # batch is a pyarrow.RecordBatch
        cols = {name: batch.column(name).to_pylist()
                for name in batch.schema.names}
        n_rows = batch.num_rows
        for i in range(n_rows):
            yield {name: cols[name][i] for name in cols}


def iterate_records(
    path: Path,
    text_key: str = 'text',
) -> Generator[Dict, None, None]:
    """
    Unified record iterator for any supported file format.

    Handles:
      - .jsonl / .json   — one JSON object per line
      - .jsonl.gz / .json.gz — gzipped variant
      - .parquet          — columnar, streamed via pyarrow

    Each yielded value is a dict.  Malformed JSON lines yield
    ``{"_malformed": True, "_raw": <first 500 chars>}``.

    Yields:
        dict per record.
    """
    if is_parquet_file(path):
        yield from iterate_parquet_records(path, text_key=text_key)
    else:
        with open_data_file(path) as f:
            for raw_line in f:
                raw_line = raw_line.rstrip('\n')
                if not raw_line.strip():
                    continue
                try:
                    yield json.loads(raw_line)
                except json.JSONDecodeError:
                    yield {"_malformed": True, "_raw": raw_line[:500]}
