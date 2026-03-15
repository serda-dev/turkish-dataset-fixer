"""
sinks.py — Dataset output sink abstraction.

Supports:
  - Local directory sink
  - Hugging Face repo sink (incremental per-shard upload)
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LocalSink:
    """Write output shards to a local directory."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.sink_type = 'local'
        self._uploaded_count = 0

    def ensure_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_shard(self, shard_path: Path, shard_name: str):
        """For local sink, shards are already written to output_dir; this is a no-op."""
        self._uploaded_count += 1
        size_mb = shard_path.stat().st_size / (1024 * 1024)
        logger.info("Local shard ready: %s (%.2f MB)", shard_name, size_mb)

    def finalize(self):
        logger.info("Local sink finalized: %d shards in %s",
                     self._uploaded_count, self.output_dir)

    @property
    def stats(self) -> dict:
        return {
            'sink_type': 'local',
            'output_path': str(self.output_dir),
            'shards_written': self._uploaded_count,
        }


class HuggingFaceSink:
    """Upload output shards to a Hugging Face repo incrementally."""

    def __init__(self, repo_id: str, token: Optional[str] = None,
                 repo_type: str = 'dataset', private: bool = False):
        self.repo_id = repo_id
        self.token = token or os.environ.get('HF_TOKEN', '')
        self.repo_type = repo_type
        self.private = private
        self.sink_type = 'huggingface'
        self._api = None
        self._uploaded_count = 0
        self._uploaded_bytes = 0
        self._failed_uploads = []

    def _get_api(self):
        """Lazy-init HfApi."""
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token or None)
        return self._api

    def ensure_dir(self):
        """Create the remote repo if it doesn't exist."""
        api = self._get_api()
        api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            private=self.private,
            exist_ok=True,
        )
        logger.info("HF repo ensured: %s (type=%s, private=%s)",
                     self.repo_id, self.repo_type, self.private)

    def write_shard(self, shard_path: Path, shard_name: str):
        """Upload a single shard file to the HF repo."""
        api = self._get_api()
        size_bytes = shard_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        logger.info("Uploading shard to HF: %s (%.2f MB) → %s",
                     shard_name, size_mb, self.repo_id)

        try:
            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=shard_name,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
            self._uploaded_count += 1
            self._uploaded_bytes += size_bytes
            logger.info("Upload complete: %s (%.2f MB)", shard_name, size_mb)
        except Exception as e:
            logger.error("Upload failed for %s: %s", shard_name, e)
            self._failed_uploads.append(shard_name)

    def finalize(self):
        total_mb = self._uploaded_bytes / (1024 * 1024)
        logger.info(
            "HF sink finalized: %d shards uploaded (%.2f MB total) to %s",
            self._uploaded_count, total_mb, self.repo_id,
        )
        if self._failed_uploads:
            logger.warning(
                "Failed uploads (%d): %s",
                len(self._failed_uploads), ', '.join(self._failed_uploads),
            )

    @property
    def stats(self) -> dict:
        return {
            'sink_type': 'huggingface',
            'repo_id': self.repo_id,
            'shards_uploaded': self._uploaded_count,
            'total_uploaded_mb': round(self._uploaded_bytes / (1024 * 1024), 2),
            'failed_uploads': self._failed_uploads,
        }


def resolve_sink(
    output_dir: Optional[str] = None,
    output_repo: Optional[str] = None,
    hf_token: str = '',
    private: bool = False,
):
    """
    Factory function: resolve the appropriate sink from CLI arguments.

    Priority:
      1. If output_repo is set → HuggingFaceSink
      2. If output_dir is set → LocalSink
      3. Raise error if neither

    Returns:
        A sink object (LocalSink or HuggingFaceSink).
    """
    if output_repo:
        logger.info("Resolved output sink: HuggingFace repo %s", output_repo)
        return HuggingFaceSink(
            repo_id=output_repo,
            token=hf_token,
            private=private,
        )

    if output_dir:
        logger.info("Resolved output sink: local directory %s", output_dir)
        return LocalSink(output_dir)

    raise ValueError("Either --output-dir or --output-repo must be specified.")
