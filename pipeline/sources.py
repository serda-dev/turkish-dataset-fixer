"""
sources.py — Dataset source abstraction.

Supports:
  - Local directory
  - Hugging Face dataset/model repo
  - GitHub repo

Each source resolves to a local directory containing dataset files.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from pipeline.dataset_discovery import discover_dataset_files

logger = logging.getLogger(__name__)

# ── Patterns for detecting repo types ─────────────────────────────────
_HF_REPO_PATTERN = re.compile(
    r'^(?:https?://huggingface\.co/(?:datasets/)?)?'
    r'([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)$'
)
_GITHUB_URL_PATTERN = re.compile(
    r'^https?://github\.com/([^/]+/[^/]+?)(?:\.git)?(?:/.*)?$'
)


def detect_source_type(source: str) -> str:
    """
    Detect whether a source string is a local path, HF repo, or GitHub repo.

    Returns: 'local', 'huggingface', or 'github'
    """
    # Explicit GitHub URL
    if _GITHUB_URL_PATTERN.match(source):
        return 'github'

    # Local path that exists on disk
    if os.path.exists(source):
        return 'local'

    # HF repo pattern: username/repo-name (no github.com)
    if _HF_REPO_PATTERN.match(source) and 'github.com' not in source:
        return 'huggingface'

    # Fallback: if it looks like a path, treat as local
    if '/' in source and not source.startswith('http'):
        return 'local'

    # Default to HF if it has the user/repo format
    if '/' in source:
        return 'huggingface'

    return 'local'


class LocalSource:
    """Local directory source — no download needed."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.source_type = 'local'

    def prepare(self) -> Path:
        if not self.path.exists():
            raise FileNotFoundError(f"Local source directory not found: {self.path}")
        if not self.path.is_dir():
            raise NotADirectoryError(f"Local source is not a directory: {self.path}")
        logger.info("Using local source: %s", self.path)
        return self.path

    def discover_files(self, extensions=None) -> List[Path]:
        return discover_dataset_files(self.path, extensions)

    def cleanup(self):
        pass  # nothing to clean up


class HuggingFaceSource:
    """Hugging Face repo source — downloads via huggingface_hub."""

    def __init__(self, repo_id: str, token: Optional[str] = None,
                 cache_dir: str = '.cache'):
        # Extract clean repo_id from URL or direct ID
        match = _HF_REPO_PATTERN.match(repo_id)
        self.repo_id = match.group(1) if match else repo_id
        self.token = token or os.environ.get('HF_TOKEN', '')
        self.cache_dir = Path(cache_dir) / 'hf_input'
        self.local_path: Optional[Path] = None
        self.source_type = 'huggingface'

    def prepare(self) -> Path:
        from huggingface_hub import snapshot_download

        logger.info("Downloading HF repo: %s", self.repo_id)
        token_display = (self.token[:4] + '****') if self.token else '<none>'
        logger.info("Using HF token: %s", token_display)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.local_path = Path(snapshot_download(
                repo_id=self.repo_id,
                repo_type='dataset',
                local_dir=str(self.cache_dir / self.repo_id.replace('/', '_')),
                token=self.token or None,
            ))
        except Exception:
            # Try as model repo if dataset fails
            logger.info("Dataset download failed, trying as model repo...")
            self.local_path = Path(snapshot_download(
                repo_id=self.repo_id,
                repo_type='model',
                local_dir=str(self.cache_dir / self.repo_id.replace('/', '_')),
                token=self.token or None,
            ))

        logger.info("HF repo downloaded to: %s", self.local_path)
        return self.local_path

    def discover_files(self, extensions=None) -> List[Path]:
        if self.local_path is None:
            raise RuntimeError("Source not prepared. Call prepare() first.")
        return discover_dataset_files(self.local_path, extensions)

    def cleanup(self):
        if self.local_path and self.local_path.exists():
            logger.info("Cleaning up HF cache: %s", self.local_path)
            shutil.rmtree(self.local_path, ignore_errors=True)


class GitHubSource:
    """GitHub repo source — clones via git."""

    def __init__(self, repo_url: str, token: Optional[str] = None,
                 cache_dir: str = '.cache'):
        self.repo_url = repo_url
        self.token = token or os.environ.get('GITHUB_TOKEN', '')
        self.cache_dir = Path(cache_dir) / 'github_input'
        self.local_path: Optional[Path] = None
        self.source_type = 'github'

    def _build_auth_url(self) -> str:
        """Inject token into GitHub URL for private repos."""
        if not self.token:
            return self.repo_url
        # https://github.com/user/repo -> https://TOKEN@github.com/user/repo
        return self.repo_url.replace(
            'https://github.com',
            f'https://{self.token}@github.com',
        )

    def prepare(self) -> Path:
        match = _GITHUB_URL_PATTERN.match(self.repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {self.repo_url}")

        repo_name = match.group(1).replace('/', '_')
        self.local_path = self.cache_dir / repo_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.local_path.exists():
            logger.info("GitHub repo already cached: %s", self.local_path)
            return self.local_path

        logger.info("Cloning GitHub repo: %s", self.repo_url)
        auth_url = self._build_auth_url()

        try:
            subprocess.run(
                ['git', 'clone', '--depth', '1', auth_url, str(self.local_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Git clone failed: %s", e.stderr)
            raise RuntimeError(f"Failed to clone {self.repo_url}: {e.stderr}") from e

        logger.info("GitHub repo cloned to: %s", self.local_path)
        return self.local_path

    def discover_files(self, extensions=None) -> List[Path]:
        if self.local_path is None:
            raise RuntimeError("Source not prepared. Call prepare() first.")
        return discover_dataset_files(self.local_path, extensions)

    def cleanup(self):
        if self.local_path and self.local_path.exists():
            logger.info("Cleaning up GitHub cache: %s", self.local_path)
            shutil.rmtree(self.local_path, ignore_errors=True)


def resolve_source(
    input_dir: Optional[str] = None,
    input_repo: Optional[str] = None,
    hf_token: str = '',
    github_token: str = '',
    cache_dir: str = '.cache',
):
    """
    Factory function: resolve the appropriate source from CLI arguments.

    Priority:
      1. If input_repo is set → detect type (HF or GitHub)
      2. If input_dir is set → LocalSource
      3. Raise error if neither

    Returns:
        A source object (LocalSource, HuggingFaceSource, or GitHubSource).
    """
    if input_repo:
        source_type = detect_source_type(input_repo)
        logger.info("Resolved input source type: %s (from: %s)", source_type, input_repo)

        if source_type == 'github':
            return GitHubSource(
                repo_url=input_repo,
                token=github_token,
                cache_dir=cache_dir,
            )
        elif source_type == 'huggingface':
            return HuggingFaceSource(
                repo_id=input_repo,
                token=hf_token,
                cache_dir=cache_dir,
            )
        else:
            # Treat as local path
            return LocalSource(input_repo)

    if input_dir:
        return LocalSource(input_dir)

    raise ValueError("Either --input-dir or --input-repo must be specified.")
