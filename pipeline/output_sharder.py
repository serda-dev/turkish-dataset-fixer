"""
output_sharder.py — Configurable output shard writer.

Accumulates filtered JSONL records and rolls to a new shard file
when the target size is reached. Optionally invokes a sink for
incremental upload after each shard is finalized.
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SHARD_SIZE_MB = 55


class OutputSharder:
    """
    Writes filtered records to JSONL shard files with configurable target size.

    Usage:
        sharder = OutputSharder(output_dir, target_mb=55)
        for record in filtered_records:
            sharder.write_record(record)
        sharder.close()

    Each completed shard triggers on_shard_complete callback if set.
    """

    def __init__(
        self,
        output_dir: Path,
        target_mb: int = DEFAULT_TARGET_SHARD_SIZE_MB,
        prefix: str = 'filtered-shard',
        on_shard_complete: Optional[Callable[[Path, str], None]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.target_bytes = target_mb * 1024 * 1024
        self.prefix = prefix
        self.on_shard_complete = on_shard_complete

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._shard_index = 0
        self._current_path: Optional[Path] = None
        self._current_handle = None
        self._current_bytes = 0
        self._total_records = 0
        self._total_shards = 0
        self._shard_sizes = []

        self._open_new_shard()

    def _shard_name(self, index: int) -> str:
        return f"{self.prefix}-{index:05d}.jsonl"

    def _open_new_shard(self):
        """Open a new shard file for writing."""
        name = self._shard_name(self._shard_index)
        self._current_path = self.output_dir / name
        self._current_handle = open(
            self._current_path, 'w', encoding='utf-8', buffering=1024 * 1024,
        )
        self._current_bytes = 0
        logger.debug("Opened new shard: %s", name)

    def _close_current_shard(self):
        """Close the current shard and trigger callback."""
        if self._current_handle is None:
            return

        self._current_handle.close()
        self._current_handle = None

        shard_name = self._shard_name(self._shard_index)
        size_mb = self._current_bytes / (1024 * 1024)

        if self._current_bytes > 0:
            self._shard_sizes.append(self._current_bytes)
            self._total_shards += 1
            logger.info("Shard closed: %s (%.2f MB)", shard_name, size_mb)

            if self.on_shard_complete:
                try:
                    self.on_shard_complete(self._current_path, shard_name)
                except Exception as e:
                    logger.error("Shard callback failed for %s: %s", shard_name, e)
        else:
            # Remove empty shard
            self._current_path.unlink(missing_ok=True)

        self._shard_index += 1

    def write_record(self, record: dict, text_key: str = 'text'):
        """
        Write a single record to the current shard.

        Automatically rolls to a new shard if the target size would be exceeded.
        """
        line = json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'
        line_bytes = len(line.encode('utf-8'))

        # Roll to new shard if current would exceed target (but allow at least 1 record)
        if self._current_bytes > 0 and (self._current_bytes + line_bytes) > self.target_bytes:
            self._close_current_shard()
            self._open_new_shard()

        self._current_handle.write(line)
        self._current_bytes += line_bytes
        self._total_records += 1

    def close(self):
        """Finalize: close the last shard."""
        self._close_current_shard()

    @property
    def stats(self) -> dict:
        return {
            'total_shards_produced': self._total_shards,
            'total_records_written': self._total_records,
            'shard_sizes_mb': [
                round(s / (1024 * 1024), 2) for s in self._shard_sizes
            ],
            'target_shard_size_mb': round(self.target_bytes / (1024 * 1024), 2),
        }
