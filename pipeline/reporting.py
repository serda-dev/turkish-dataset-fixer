"""
reporting.py — Generate reports and audit sample files.

Creates:
- Per-shard JSON reports
- Global summary report (JSON + TXT)
- Audit sample files (accepted, rejected, borderline)
- Score distribution CSV
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from pipeline.dataset_discovery import dataset_output_stem

logger = logging.getLogger(__name__)


def save_per_shard_report(stats: Dict, cfg):
    """Save per-shard statistics as JSON report."""
    shard_name = stats.get('shard', 'unknown')
    report_name = dataset_output_stem(Path(shard_name)) + '_report.json'
    report_path = cfg.per_shard_reports_dir / report_name

    # Remove large lists from report (keep summaries only)
    report = {k: v for k, v in stats.items()
              if k not in ('audit_samples', 'kenlm_scores',
                          'kept_lengths', 'rejected_lengths')}
    report['kenlm_scores_count'] = len(stats.get('kenlm_scores', []))
    report['kept_lengths_count'] = len(stats.get('kept_lengths', []))
    report['rejected_lengths_count'] = len(stats.get('rejected_lengths', []))

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.debug("Per-shard report saved: %s", report_path)


def generate_global_report(
    all_stats: List[Dict],
    dedup_stats: Dict,
    cfg,
):
    """Generate global summary report (JSON + TXT) and audit files."""
    reports_dir = cfg.reports_dir
    audit_dir = cfg.audit_dir

    # ── Aggregate statistics ──────────────────────────────────────────
    total_files = len(all_stats)
    total_lines = sum(s.get('total_lines', 0) for s in all_stats)
    total_malformed = sum(s.get('malformed_lines', 0) for s in all_stats)
    total_missing_text = sum(s.get('missing_text', 0) for s in all_stats)
    total_kept = sum(s.get('kept', 0) for s in all_stats)
    total_rejected = sum(s.get('rejected', 0) for s in all_stats)
    total_exact_dup = sum(s.get('exact_duplicates', 0) for s in all_stats)
    total_near_dup = sum(s.get('near_duplicates', 0) for s in all_stats)

    # Aggregate rejection reasons
    rejection_reasons = Counter()
    for s in all_stats:
        for reason, count in s.get('rejection_reasons', {}).items():
            rejection_reasons[reason] += count

    # Aggregate language distribution
    lang_dist = Counter()
    for s in all_stats:
        for lang, count in s.get('lang_distribution', {}).items():
            lang_dist[lang] += count

    # Aggregate KenLM scores
    all_kenlm = []
    for s in all_stats:
        all_kenlm.extend(s.get('kenlm_scores', []))

    # Aggregate length distributions
    all_kept_lengths = []
    all_rejected_lengths = []
    for s in all_stats:
        all_kept_lengths.extend(s.get('kept_lengths', []))
        all_rejected_lengths.extend(s.get('rejected_lengths', []))

    # Collect audit samples
    all_accepted_samples = []
    all_rejected_samples = []
    all_borderline_samples = []
    for s in all_stats:
        audit = s.get('audit_samples', {})
        all_accepted_samples.extend(audit.get('accepted', []))
        all_rejected_samples.extend(audit.get('rejected', []))
        all_borderline_samples.extend(audit.get('borderline', []))

    # ── KenLM score distribution stats ────────────────────────────────
    kenlm_dist = {}
    if all_kenlm:
        sorted_kenlm = sorted(all_kenlm)
        kenlm_dist = {
            'count': len(sorted_kenlm),
            'min': sorted_kenlm[0],
            'max': sorted_kenlm[-1],
            'median': sorted_kenlm[len(sorted_kenlm) // 2],
            'p10': sorted_kenlm[int(len(sorted_kenlm) * 0.1)],
            'p25': sorted_kenlm[int(len(sorted_kenlm) * 0.25)],
            'p75': sorted_kenlm[int(len(sorted_kenlm) * 0.75)],
            'p90': sorted_kenlm[int(len(sorted_kenlm) * 0.9)],
            'p95': sorted_kenlm[int(len(sorted_kenlm) * 0.95)],
            'p99': sorted_kenlm[int(len(sorted_kenlm) * 0.99)],
        }

    # Length distribution stats
    def _length_percentiles(lengths):
        if not lengths:
            return {}
        s = sorted(lengths)
        return {
            'count': len(s), 'min': s[0], 'max': s[-1],
            'median': s[len(s) // 2],
            'p25': s[int(len(s) * 0.25)],
            'p75': s[int(len(s) * 0.75)],
        }

    # ── Build global report dict ──────────────────────────────────────
    global_report = {
        'total_files_processed': total_files,
        'total_records_seen': total_lines,
        'total_malformed_lines': total_malformed,
        'total_missing_text': total_missing_text,
        'total_kept': total_kept,
        'total_rejected': total_rejected,
        'keep_ratio': round(total_kept / max(1, total_lines), 4),
        'reject_ratio': round(total_rejected / max(1, total_lines), 4),
        'exact_duplicate_count': total_exact_dup,
        'near_duplicate_count': total_near_dup,
        'dedup_stats': dedup_stats,
        'top_rejection_reasons': dict(rejection_reasons.most_common(20)),
        'language_distribution': dict(lang_dist.most_common(20)),
        'kenlm_score_distribution': kenlm_dist,
        'kept_length_distribution': _length_percentiles(all_kept_lengths),
        'rejected_length_distribution': _length_percentiles(all_rejected_lengths),
    }

    # ── Save JSON report ──────────────────────────────────────────────
    json_path = reports_dir / 'global_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False)
    logger.info("Global JSON report: %s", json_path)

    # ── Save TXT report ───────────────────────────────────────────────
    txt_path = reports_dir / 'global_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  TURKISH DATASET QUALITY FILTERING — GLOBAL SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Files processed:     {total_files}\n")
        f.write(f"Total records:       {total_lines:,}\n")
        f.write(f"Malformed lines:     {total_malformed:,}\n")
        f.write(f"Missing text:        {total_missing_text:,}\n")
        f.write(f"Kept:                {total_kept:,} ({global_report['keep_ratio']:.1%})\n")
        f.write(f"Rejected:            {total_rejected:,} ({global_report['reject_ratio']:.1%})\n")
        f.write(f"Exact duplicates:    {total_exact_dup:,}\n")
        f.write(f"Near duplicates:     {total_near_dup:,}\n\n")

        f.write("Top Rejection Reasons:\n")
        f.write("-" * 40 + "\n")
        for reason, count in rejection_reasons.most_common(20):
            f.write(f"  {reason:40s}  {count:>10,}\n")
        f.write("\n")

        f.write("Language Distribution:\n")
        f.write("-" * 40 + "\n")
        for lang, count in lang_dist.most_common(20):
            pct = count / max(1, total_lines) * 100
            f.write(f"  {lang:10s}  {count:>10,}  ({pct:.1f}%)\n")
        f.write("\n")

        if kenlm_dist:
            f.write("KenLM Perplexity Distribution:\n")
            f.write("-" * 40 + "\n")
            for k, v in kenlm_dist.items():
                f.write(f"  {k:10s}  {v:>12.1f}\n")
            f.write("\n")

    logger.info("Global TXT report: %s", txt_path)

    # ── Save score distributions CSV ──────────────────────────────────
    if all_kenlm:
        csv_path = reports_dir / 'kenlm_score_distribution.csv'
        with open(csv_path, 'w') as f:
            f.write('perplexity\n')
            for score in sorted(all_kenlm):
                f.write(f'{score}\n')
        logger.info("KenLM score CSV: %s", csv_path)

    # ── Save audit sample files ───────────────────────────────────────
    import random

    def _save_audit(samples, filename, max_samples):
        if not samples:
            return
        path = audit_dir / filename
        chosen = random.sample(samples, min(len(samples), max_samples))
        with open(path, 'w', encoding='utf-8') as f:
            for sample in chosen:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info("Audit file: %s (%d samples)", path, len(chosen))

    _save_audit(all_accepted_samples, 'accepted_sample.jsonl', cfg.audit_sample_size)
    _save_audit(all_rejected_samples, 'rejected_sample.jsonl', cfg.audit_sample_size)
    _save_audit(all_borderline_samples, 'borderline_sample.jsonl', cfg.audit_sample_size)

    return global_report
