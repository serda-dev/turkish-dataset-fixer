"""
decision_logic.py — Stage 6: Final keep/reject decision.

Layered decision policy:
1. Hard reject: heuristic flags → immediate reject
2. Language reject: non-Turkish → reject
3. Duplicate reject: exact/near duplicate → reject
4. Soft scoring: weighted quality score combining KenLM + language + heuristics
5. Soft threshold: quality score below threshold → reject

Each record gets a full decision metadata dict for traceability.
"""

from typing import Dict, List, Tuple


def compute_quality_score(
    features: Dict,
    lang_info: Dict,
    kenlm_result: Dict,
    cfg,
) -> float:
    """
    Compute a weighted quality score in [0, 1].

    Components:
    - KenLM: perplexity mapped to [0, 1] (lower perplexity = higher quality)
    - Language confidence: lang ID confidence for Turkish
    - Heuristic: composite of alpha ratio, token diversity, etc.
    """
    # ── KenLM component ──────────────────────────────────────────────
    kenlm_score = 0.5  # neutral default
    if kenlm_result.get('kenlm_available', False):
        ppl = kenlm_result.get('kenlm_perplexity', 0.0)
        if ppl > 0:
            # Map perplexity to [0, 1]: lower perplexity → higher score
            # Using logistic-like mapping centered at threshold
            max_ppl = cfg.kenlm_max_perplexity
            # Score = 1 - (ppl / max_ppl), clamped to [0, 1]
            kenlm_score = max(0.0, min(1.0, 1.0 - (ppl / max_ppl)))
        elif ppl < 0:
            # Sentinel for too-short text — neutral score
            kenlm_score = 0.5

    # ── Language component ────────────────────────────────────────────
    lang_score = 0.5  # neutral default
    lang_conf = lang_info.get('confidence', 0.0)
    detected = lang_info.get('detected_lang', 'unknown')
    if detected == 'tr':
        lang_score = min(1.0, lang_conf + 0.2)  # boost Turkish
    elif lang_conf > 0:
        lang_score = max(0.0, 0.5 - lang_conf * 0.5)  # penalize non-Turkish

    # ── Heuristic component ───────────────────────────────────────────
    # Composite of several features
    alpha_score = min(1.0, features.get('alpha_ratio', 0.5) / 0.7)  # normalize
    diversity_score = min(1.0, features.get('unique_token_ratio', 0.5) / 0.5)
    stopword_score = min(1.0, features.get('stopword_coverage', 0.0) / 0.05)
    tr_char_score = min(1.0, features.get('turkish_char_ratio', 0.0) / 0.05)

    heuristic_score = (
        alpha_score * 0.3 +
        diversity_score * 0.2 +
        stopword_score * 0.3 +
        tr_char_score * 0.2
    )

    # ── Weighted combination ──────────────────────────────────────────
    quality_score = (
        kenlm_score * cfg.weight_kenlm +
        lang_score * cfg.weight_lang_confidence +
        heuristic_score * cfg.weight_heuristic
    )

    return round(max(0.0, min(1.0, quality_score)), 4)


def make_decision(
    text: str,
    features: Dict,
    heuristic_reject: bool,
    heuristic_reasons: List[str],
    lang_decision: str,
    lang_info: Dict,
    kenlm_result: Dict,
    kenlm_reject: bool,
    kenlm_reason: str,
    is_exact_dup: bool,
    is_near_dup: bool,
    cfg,
) -> Dict:
    """
    Make final keep/reject decision for a record.

    Returns a decision dict with all metadata.
    """
    all_reasons = []
    decision = 'keep'

    # ── Layer 1: Hard heuristic reject ────────────────────────────────
    if heuristic_reject:
        decision = 'reject'
        all_reasons.extend(heuristic_reasons)

    # ── Layer 2: Language reject ──────────────────────────────────────
    if lang_decision == 'reject':
        decision = 'reject'
        all_reasons.append('non_turkish')

    # ── Layer 3: Exact duplicate reject ───────────────────────────────
    if is_exact_dup:
        decision = 'reject'
        all_reasons.append('exact_duplicate')

    # ── Layer 3b: Near duplicate reject ───────────────────────────────
    if is_near_dup:
        decision = 'reject'
        all_reasons.append('near_duplicate')

    # ── Layer 4: KenLM reject ─────────────────────────────────────────
    if kenlm_reject is True and decision == 'keep':
        decision = 'reject'
        all_reasons.append(kenlm_reason)

    # ── Layer 5: Soft quality scoring ─────────────────────────────────
    quality_score = compute_quality_score(features, lang_info, kenlm_result, cfg)

    if decision == 'keep' and quality_score < cfg.soft_score_threshold:
        decision = 'reject'
        all_reasons.append(f'low_quality_score_{quality_score:.3f}')

    # ── Determine if borderline ───────────────────────────────────────
    is_borderline = (
        abs(quality_score - cfg.soft_score_threshold) < cfg.borderline_margin
    )

    return {
        'decision': decision,
        'reasons': all_reasons if all_reasons else [],
        'quality_score': quality_score,
        'is_borderline': is_borderline,
        'heuristic_reject': heuristic_reject,
        'heuristic_reasons': heuristic_reasons,
        'lang_decision': lang_decision,
        'lang_info': {
            'detected_lang': lang_info.get('detected_lang', 'unknown'),
            'confidence': lang_info.get('confidence', 0.0),
            'method': lang_info.get('method', 'none'),
        },
        'kenlm_perplexity': kenlm_result.get('kenlm_perplexity', 0.0),
        'kenlm_oov_ratio': kenlm_result.get('kenlm_oov_ratio', 0.0),
        'is_exact_dup': is_exact_dup,
        'is_near_dup': is_near_dup,
        'features': features,
    }
