"""
process_shard.py — Per-shard processing orchestrator.

Streams records from JSONL or Parquet files, applies all pipeline stages,
writes to filtered/rejected output files (always JSONL),
and collects statistics and audit samples.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.text_normalization import normalize_text
from pipeline.heuristic_features import compute_features, apply_heuristic_filters
from pipeline.language_validation import validate_language
from pipeline.kenlm_scorer import score_text, evaluate_kenlm_quality
from pipeline.decision_logic import make_decision
from pipeline.dataset_discovery import iterate_records

logger = logging.getLogger(__name__)


def process_shard(
    shard_path: Path,
    cfg,
    deduplicator=None,
    near_deduplicator=None,
    shard_index: int = 0,
) -> Dict:
    """
    Process a single JSONL shard through all pipeline stages.

    Returns per-shard statistics dict.
    """
    shard_name = shard_path.name
    logger.info("Processing shard: %s", shard_name)

    # ── Output paths (always JSONL, even if input is parquet) ─────────
    out_stem = shard_path.stem
    # Strip .parquet / .jsonl.gz etc. and always emit .jsonl
    if out_stem.endswith('.jsonl') or out_stem.endswith('.json'):
        out_stem = Path(out_stem).stem  # handle double extensions like .jsonl.gz
    out_name = out_stem + '.jsonl'
    filtered_path = cfg.filtered_dir / out_name
    rejected_path = cfg.rejected_dir / out_name

    # ── Statistics ────────────────────────────────────────────────────
    stats = {
        'shard': shard_name,
        'total_lines': 0,
        'malformed_lines': 0,
        'missing_text': 0,
        'kept': 0,
        'rejected': 0,
        'rejection_reasons': {},
        'exact_duplicates': 0,
        'near_duplicates': 0,
        'lang_distribution': {},
        'kenlm_scores': [],     # sampled for distribution
        'kept_lengths': [],     # sampled for distribution
        'rejected_lengths': [], # sampled for distribution
    }

    # ── Audit sample collectors ───────────────────────────────────────
    accepted_samples = []
    rejected_samples = []
    borderline_samples = []
    max_audit_samples = cfg.audit_sample_size // 2  # per shard contribution

    # ── Process records (format-agnostic) ──────────────────────────────
    with open(filtered_path, 'w', encoding='utf-8') as out_f, \
         open(rejected_path, 'w', encoding='utf-8') as rej_f:

        for line_num, record in enumerate(iterate_records(
                shard_path, text_key=cfg.text_key)):
            stats['total_lines'] += 1

            # ── Handle malformed records ──────────────────────────────
            if record.get('_malformed', False):
                stats['malformed_lines'] += 1
                try:
                    rej_f.write(json.dumps({
                        'text': record.get('_raw', '')[:500],
                        '_rejection_reasons': ['malformed_json'],
                        '_line_num': line_num,
                    }, ensure_ascii=False) + '\n')
                except Exception:
                    pass
                continue

            # ── Extract text ──────────────────────────────────────────
            text = record.get(cfg.text_key)
            if text is None:
                stats['missing_text'] += 1
                continue
            if not isinstance(text, str):
                text = str(text)

            # ── Stage 1: Normalize ────────────────────────────────────
            normalized_text = normalize_text(text)

            # ── Stage 2: Heuristic features ───────────────────────────
            features = compute_features(normalized_text)
            heuristic_reject, heuristic_reasons = apply_heuristic_filters(
                normalized_text, features, cfg
            )

            # ── Stage 3: Language validation ──────────────────────────
            lang_decision, lang_info = validate_language(
                normalized_text, features, cfg
            )

            # Track language distribution
            detected = lang_info.get('detected_lang', 'unknown')
            stats['lang_distribution'][detected] = \
                stats['lang_distribution'].get(detected, 0) + 1

            # ── Stage 4: KenLM scoring ────────────────────────────────
            kenlm_result = score_text(normalized_text, cfg)
            kenlm_reject, kenlm_reason = evaluate_kenlm_quality(kenlm_result, cfg)

            # Sample KenLM scores for distribution
            ppl = kenlm_result.get('kenlm_perplexity', 0.0)
            if ppl > 0 and random.random() < 0.01:  # 1% sample
                stats['kenlm_scores'].append(round(ppl, 1))

            # ── Stage 5: Dedup check ──────────────────────────────────
            is_exact_dup = False
            is_near_dup = False

            if deduplicator and cfg.enable_exact_dedup:
                is_exact_dup = deduplicator.is_duplicate(normalized_text)
                if is_exact_dup:
                    stats['exact_duplicates'] += 1

            if near_deduplicator and cfg.enable_near_dedup and not is_exact_dup:
                doc_id = f"{shard_name}:{line_num}"
                is_near_dup = near_deduplicator.is_near_duplicate(
                    normalized_text, doc_id
                )
                if is_near_dup:
                    stats['near_duplicates'] += 1

            # ── Stage 6: Decision ─────────────────────────────────────
            decision_result = make_decision(
                text=normalized_text,
                features=features,
                heuristic_reject=heuristic_reject,
                heuristic_reasons=heuristic_reasons,
                lang_decision=lang_decision,
                lang_info=lang_info,
                kenlm_result=kenlm_result,
                kenlm_reject=kenlm_reject if kenlm_reject is not None else False,
                kenlm_reason=kenlm_reason,
                is_exact_dup=is_exact_dup,
                is_near_dup=is_near_dup,
                cfg=cfg,
            )

            # ── Write output ──────────────────────────────────────────
            if decision_result['decision'] == 'keep':
                stats['kept'] += 1
                # Write original record with normalized text, preserve metadata
                out_record = dict(record)
                out_record[cfg.text_key] = normalized_text
                out_f.write(json.dumps(out_record, ensure_ascii=False) + '\n')

                # Sample for distribution
                if random.random() < 0.005:
                    stats['kept_lengths'].append(features['char_count'])

                # Audit sample
                if len(accepted_samples) < max_audit_samples and random.random() < 0.01:
                    accepted_samples.append({
                        'text': normalized_text[:1000],
                        'quality_score': decision_result['quality_score'],
                        'kenlm_perplexity': decision_result['kenlm_perplexity'],
                        'lang': decision_result['lang_info'],
                        'shard': shard_name,
                        'line': line_num,
                    })
            else:
                stats['rejected'] += 1
                # Write slim rejected record for debugging
                rej_record = {
                    cfg.text_key: normalized_text[:500],  # truncated for space
                    '_rejection_reasons': decision_result['reasons'],
                    '_quality_score': decision_result['quality_score'],
                    '_shard': shard_name,
                    '_line_num': line_num,
                }
                rej_f.write(json.dumps(rej_record, ensure_ascii=False) + '\n')

                # Track rejection reasons
                for reason in decision_result['reasons']:
                    stats['rejection_reasons'][reason] = \
                        stats['rejection_reasons'].get(reason, 0) + 1

                # Sample for distribution
                if random.random() < 0.01:
                    stats['rejected_lengths'].append(features['char_count'])

                # Audit sample
                if len(rejected_samples) < max_audit_samples and random.random() < 0.02:
                    rejected_samples.append({
                        'text': normalized_text[:1000],
                        'reasons': decision_result['reasons'],
                        'quality_score': decision_result['quality_score'],
                        'kenlm_perplexity': decision_result['kenlm_perplexity'],
                        'lang': decision_result['lang_info'],
                        'shard': shard_name,
                        'line': line_num,
                    })

            # Borderline samples
            if decision_result.get('is_borderline', False):
                if len(borderline_samples) < max_audit_samples and random.random() < 0.05:
                    borderline_samples.append({
                        'text': normalized_text[:1000],
                        'decision': decision_result['decision'],
                        'reasons': decision_result['reasons'],
                        'quality_score': decision_result['quality_score'],
                        'kenlm_perplexity': decision_result['kenlm_perplexity'],
                        'lang': decision_result['lang_info'],
                        'shard': shard_name,
                        'line': line_num,
                    })

    # ── Finalize stats ────────────────────────────────────────────────
    stats['audit_samples'] = {
        'accepted': accepted_samples,
        'rejected': rejected_samples,
        'borderline': borderline_samples,
    }

    logger.info("Shard %s: %d total, %d kept, %d rejected, %d malformed, %d exact_dup",
                shard_name, stats['total_lines'], stats['kept'],
                stats['rejected'], stats['malformed_lines'],
                stats['exact_duplicates'])

    return stats
