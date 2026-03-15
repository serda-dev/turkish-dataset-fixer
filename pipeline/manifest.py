"""
manifest.py — JSON-based progress tracking for VPS resilience.

Tracks processed files and uploaded shards so interrupted runs can resume
without reprocessing everything.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ProcessingManifest:
    """
    Simple JSON manifest for tracking pipeline progress.

    Supports:
      - Marking input files as processed
      - Marking output shards as uploaded
      - Resume: skip already-done items on restart
    """

    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        self._data: Dict = {
            'version': 1,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'processed_files': [],
            'produced_shards': [],
            'uploaded_shards': [],
            'errors': [],
        }
        self._processed_set: Set[str] = set()
        self._uploaded_set: Set[str] = set()

    def load(self) -> bool:
        """Load manifest from disk. Returns True if loaded successfully."""
        if not self.path.exists():
            logger.info("No existing manifest found at %s", self.path)
            return False

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)

            self._processed_set = set(self._data.get('processed_files', []))
            self._uploaded_set = set(self._data.get('uploaded_shards', []))

            logger.info(
                "Resumed manifest: %d files processed, %d shards uploaded",
                len(self._processed_set), len(self._uploaded_set),
            )
            return True
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt manifest at %s, starting fresh: %s", self.path, e)
            return False

    def save(self):
        """Persist manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        self._data['processed_files'] = sorted(self._processed_set)
        self._data['uploaded_shards'] = sorted(self._uploaded_set)

        tmp_path = self.path.with_suffix('.json.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.path)

    def is_file_done(self, file_path: str) -> bool:
        """Check if an input file has already been processed."""
        return str(file_path) in self._processed_set

    def mark_file_done(self, file_path: str):
        """Mark an input file as processed."""
        self._processed_set.add(str(file_path))

    def is_shard_uploaded(self, shard_name: str) -> bool:
        """Check if a shard has already been uploaded."""
        return shard_name in self._uploaded_set

    def mark_shard_uploaded(self, shard_name: str):
        """Mark a shard as uploaded."""
        self._uploaded_set.add(shard_name)

    def mark_shard_produced(self, shard_name: str):
        """Record a produced shard."""
        produced = self._data.setdefault('produced_shards', [])
        if shard_name not in produced:
            produced.append(shard_name)

    def record_error(self, file_path: str, error: str):
        """Record a processing error."""
        self._data.setdefault('errors', []).append({
            'file': str(file_path),
            'error': error,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        })

    @property
    def processed_count(self) -> int:
        return len(self._processed_set)

    @property
    def uploaded_count(self) -> int:
        return len(self._uploaded_set)

    @property
    def error_count(self) -> int:
        return len(self._data.get('errors', []))
