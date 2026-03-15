"""
inspect_dataset.py — Stage 0: Automated dataset inspection.

Scans shards, samples records, reports field distributions,
length statistics, and edge cases before processing.
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

TURKISH_SPECIFIC_CHARS = set('çÇğĞıİöÖşŞüÜ')


def inspect_dataset(input_dir: str, sample_per_shard: int = 100) -> Dict:
    """
    Inspect dataset structure and produce summary report.

    Returns dict with inspection results.
    """
    input_path = Path(input_dir)
    shard_files = sorted(input_path.glob('*.jsonl'))

    if not shard_files:
        logger.error("No .jsonl files found in %s", input_dir)
        return {'error': 'no_files'}

    logger.info("Found %d shard files in %s", len(shard_files), input_dir)

    # ── Gather statistics ─────────────────────────────────────────────
    total_size_bytes = 0
    file_info = []
    field_counter = Counter()
    total_records_sampled = 0
    malformed_lines = 0
    length_stats = {'min': float('inf'), 'max': 0, 'total': 0, 'count': 0}
    tr_char_total = 0
    total_chars = 0
    has_html = 0
    has_url = 0
    sample_texts = []  # first few texts for display

    for shard in shard_files:
        size = shard.stat().st_size
        total_size_bytes += size

        # Count lines and sample
        line_count = 0
        with open(shard, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                line_count += 1
                if i >= sample_per_shard:
                    continue  # still count lines

                line = line.strip()
                if not line:
                    continue

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue

                total_records_sampled += 1
                for key in rec.keys():
                    field_counter[key] += 1

                text = rec.get('text', '')
                if isinstance(text, str):
                    tlen = len(text)
                    length_stats['count'] += 1
                    length_stats['total'] += tlen
                    length_stats['min'] = min(length_stats['min'], tlen)
                    length_stats['max'] = max(length_stats['max'], tlen)
                    total_chars += tlen
                    tr_char_total += sum(1 for c in text if c in TURKISH_SPECIFIC_CHARS)

                    if '<html' in text.lower() or '<div' in text.lower():
                        has_html += 1
                    if 'http://' in text or 'https://' in text:
                        has_url += 1

                    if len(sample_texts) < 5:
                        sample_texts.append(text[:300])

        file_info.append({
            'name': shard.name,
            'size_mb': round(size / (1024 * 1024), 2),
            'line_count': line_count,
        })

    avg_len = length_stats['total'] // max(1, length_stats['count'])
    tr_ratio = tr_char_total / max(1, total_chars)

    result = {
        'shard_count': len(shard_files),
        'total_size_gb': round(total_size_bytes / (1024**3), 2),
        'total_records_sampled': total_records_sampled,
        'malformed_lines': malformed_lines,
        'field_frequencies': dict(field_counter),
        'text_length_stats': {
            'min': length_stats['min'],
            'max': length_stats['max'],
            'avg': avg_len,
            'count': length_stats['count'],
        },
        'turkish_char_ratio': round(tr_ratio, 4),
        'records_with_html': has_html,
        'records_with_url': has_url,
        'sample_texts': sample_texts,
        'file_info': file_info[:10],  # first 10 shards
    }

    # ── Log summary ──────────────────────────────────────────────────
    logger.info("=== Dataset Inspection Summary ===")
    logger.info("Shards: %d, Total size: %.2f GB", result['shard_count'], result['total_size_gb'])
    logger.info("Records sampled: %d, Malformed: %d", total_records_sampled, malformed_lines)
    logger.info("Fields found: %s", dict(field_counter))
    logger.info("Text length — min: %d, max: %d, avg: %d",
                length_stats['min'], length_stats['max'], avg_len)
    logger.info("Turkish char ratio: %.4f", tr_ratio)
    logger.info("Records with HTML: %d, with URL: %d", has_html, has_url)

    return result
