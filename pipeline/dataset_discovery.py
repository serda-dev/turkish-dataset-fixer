"""
dataset_discovery.py — Recursive dataset file discovery with gz support.

Walks directories recursively to find .json, .jsonl, .json.gz, .jsonl.gz files.
Provides transparent decompression for gzipped files.
"""

import gzip
import logging
import os
from pathlib import Path
from typing import IO, List, Optional, Set

logger = logging.getLogger(__name__)

# Supported raw extensions and their gzipped variants
SUPPORTED_EXTENSIONS: Set[str] = {
    '.json', '.jsonl',
    '.json.gz', '.jsonl.gz',
}


def _get_double_suffix(path: Path) -> str:
    """Get double suffix for .json.gz style extensions."""
    suffixes = path.suffixes
    if len(suffixes) >= 2:
        return ''.join(suffixes[-2:]).lower()
    return path.suffix.lower()


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
    Open a data file, transparently handling gzip compression.

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
