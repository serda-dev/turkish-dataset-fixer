"""
dedup.py — Stage 5: Deduplication.

Exact dedup: Global xxhash-based hash set across all shards.
Near-dedup: Optional MinHash LSH via datasketch (modular, can be enabled later).
"""

import logging
from typing import Optional, Set

import xxhash

logger = logging.getLogger(__name__)


class ExactDeduplicator:
    """
    Global exact deduplicator using xxhash64.

    Memory estimate: ~15.5M records × 8 bytes/hash ≈ 124 MB.
    Fits comfortably in RAM.
    """

    def __init__(self):
        self._seen: Set[int] = set()
        self._dup_count: int = 0
        self._total_checked: int = 0

    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is an exact duplicate.
        First occurrence is kept (returns False).
        Subsequent occurrences return True.
        """
        self._total_checked += 1
        # Hash normalized text (stripped, lowered) for robust matching
        normalized = text.strip().lower()
        h = xxhash.xxh64_intdigest(normalized.encode('utf-8', errors='replace'))

        if h in self._seen:
            self._dup_count += 1
            return True

        self._seen.add(h)
        return False

    @property
    def stats(self) -> dict:
        return {
            'total_checked': self._total_checked,
            'unique_count': len(self._seen),
            'duplicate_count': self._dup_count,
            'duplicate_ratio': round(self._dup_count / max(1, self._total_checked), 4),
        }

    def __len__(self):
        return len(self._seen)


class NearDeduplicator:
    """
    Optional MinHash LSH near-deduplicator.

    Uses character n-gram shingling + MinHash for approximate matching.
    This is heavier and should only be enabled when needed.
    """

    def __init__(self, threshold: float = 0.80, num_perm: int = 128,
                 shingle_size: int = 5):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._lsh = None
        self._dup_count = 0
        self._total_checked = 0

        try:
            from datasketch import MinHash, MinHashLSH
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._MinHash = MinHash
            logger.info("Near-dedup initialized: threshold=%.2f, perms=%d, shingle=%d",
                       threshold, num_perm, shingle_size)
        except ImportError:
            logger.warning("datasketch not available, near-dedup disabled")

    def _get_shingles(self, text: str) -> set:
        """Generate character n-gram shingles."""
        text = text.strip().lower()
        if len(text) < self.shingle_size:
            return {text}
        return {text[i:i + self.shingle_size]
                for i in range(len(text) - self.shingle_size + 1)}

    def _create_minhash(self, text: str):
        """Create MinHash for text."""
        m = self._MinHash(num_perm=self.num_perm)
        for shingle in self._get_shingles(text):
            m.update(shingle.encode('utf-8'))
        return m

    def is_near_duplicate(self, text: str, doc_id: str) -> bool:
        """
        Check if text is a near-duplicate of any previously seen text.

        Args:
            text: text to check
            doc_id: unique identifier for this document

        Returns:
            True if near-duplicate found
        """
        if self._lsh is None:
            return False

        self._total_checked += 1
        mh = self._create_minhash(text)

        # Query for similar documents
        try:
            results = self._lsh.query(mh)
            if results:
                self._dup_count += 1
                return True

            # Insert into LSH index
            self._lsh.insert(doc_id, mh)
            return False
        except Exception as e:
            # On error (e.g., duplicate key), just skip
            logger.debug("Near-dedup error for %s: %s", doc_id, e)
            return False

    @property
    def stats(self) -> dict:
        return {
            'total_checked': self._total_checked,
            'near_duplicate_count': self._dup_count,
            'near_duplicate_ratio': round(
                self._dup_count / max(1, self._total_checked), 4
            ),
        }
