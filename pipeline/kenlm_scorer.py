"""
kenlm_scorer.py — Stage 4: KenLM-based quality scoring.

Scores Turkish text using a pre-trained KenLM model.
Computes per-word perplexity as a quality signal.

Scoring approach:
- KenLM returns log10(p(sentence))
- Per-word perplexity = 10^(-score / num_words)
- Clean Turkish prose: ~100-500 perplexity
- Acceptable informal: ~500-2000
- Likely garbage: >5000

Short-text handling:
- <10 words: skip (unreliable), return sentinel
- 10-30 words: use relaxed threshold (1.5× normal)
- >30 words: standard threshold
"""

import logging
import os
import math
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded model
_kenlm_model = None
_kenlm_path = None


def _load_model(model_path: str):
    """Load KenLM model (lazy, singleton)."""
    global _kenlm_model, _kenlm_path
    if _kenlm_model is not None and _kenlm_path == model_path:
        return _kenlm_model
    if not os.path.exists(model_path):
        logger.warning("KenLM model not found at %s", model_path)
        return None
    try:
        import kenlm
        _kenlm_model = kenlm.Model(model_path)
        _kenlm_path = model_path
        logger.info("Loaded KenLM model from %s (order=%d)",
                    model_path, _kenlm_model.order)
        return _kenlm_model
    except Exception as e:
        logger.error("Failed to load KenLM model: %s", e)
        return None


def score_text(text: str, cfg) -> Dict:
    """
    Score text using KenLM model.

    Returns dict with:
    - kenlm_score: raw log10 score
    - kenlm_perplexity: per-word perplexity
    - kenlm_num_tokens: number of tokens scored
    - kenlm_available: whether scoring was performed
    - kenlm_oov_ratio: ratio of out-of-vocabulary words
    """
    result = {
        'kenlm_score': 0.0,
        'kenlm_perplexity': 0.0,
        'kenlm_num_tokens': 0,
        'kenlm_available': False,
        'kenlm_oov_ratio': 0.0,
    }

    model = _load_model(cfg.kenlm_model_path)
    if model is None:
        return result

    # Prepare text: lowercase, single line
    clean_text = text.lower().replace('\n', ' ').strip()
    tokens = clean_text.split()
    num_tokens = len(tokens)

    result['kenlm_num_tokens'] = num_tokens
    result['kenlm_available'] = True

    # Skip very short texts (unreliable scoring)
    if num_tokens < cfg.kenlm_min_tokens_for_scoring:
        result['kenlm_perplexity'] = -1.0  # sentinel: too short
        return result

    # For very long texts, sample the middle portion to avoid edge effects
    # and keep scoring fast
    max_score_tokens = 500
    if num_tokens > max_score_tokens:
        # Take beginning, middle, and end portions
        chunk_size = max_score_tokens // 3
        beginning = tokens[:chunk_size]
        mid_start = (num_tokens - chunk_size) // 2
        middle = tokens[mid_start:mid_start + chunk_size]
        end = tokens[-chunk_size:]
        score_tokens = beginning + middle + end
        score_text_str = ' '.join(score_tokens)
        effective_tokens = len(score_tokens)
    else:
        score_text_str = ' '.join(tokens)
        effective_tokens = num_tokens

    # Score with KenLM (log10 probability)
    try:
        raw_score = model.score(score_text_str, bos=True, eos=True)
    except Exception as e:
        logger.debug("KenLM scoring error: %s", e)
        return result

    result['kenlm_score'] = raw_score

    # Compute per-word perplexity: 10^(-log10_score / num_words)
    if effective_tokens > 0:
        avg_log_prob = raw_score / effective_tokens
        # Perplexity = 10^(-avg_log10_prob)
        try:
            perplexity = 10.0 ** (-avg_log_prob)
        except OverflowError:
            perplexity = float('inf')
        result['kenlm_perplexity'] = round(perplexity, 2)

    # Estimate OOV ratio (words not in model vocabulary)
    try:
        # Score each word individually and check for heavy backoff
        words_scored = list(model.full_scores(score_text_str, bos=True, eos=True))
        oov_count = sum(1 for _, _, is_oov in words_scored if is_oov)
        result['kenlm_oov_ratio'] = round(oov_count / max(1, effective_tokens), 4)
    except Exception:
        pass

    return result


def evaluate_kenlm_quality(
    kenlm_result: Dict,
    cfg
) -> Tuple[Optional[bool], str]:
    """
    Evaluate text quality based on KenLM score.

    Returns:
        (reject_decision, reason) where:
        - reject_decision: True=reject, False=keep, None=no opinion (skip)
        - reason: explanation string
    """
    if not kenlm_result.get('kenlm_available', False):
        return None, 'kenlm_not_available'

    perplexity = kenlm_result.get('kenlm_perplexity', 0.0)
    num_tokens = kenlm_result.get('kenlm_num_tokens', 0)

    # Too short to score reliably
    if perplexity < 0:
        return None, 'too_short_for_kenlm'

    # Determine threshold based on text length
    threshold = cfg.kenlm_max_perplexity
    if num_tokens < cfg.kenlm_short_text_threshold:
        threshold *= cfg.kenlm_short_text_multiplier

    # Very high perplexity = likely garbage
    if perplexity > threshold:
        return True, 'high_kenlm_perplexity'

    # Very high OOV + high perplexity = suspicious
    oov_ratio = kenlm_result.get('kenlm_oov_ratio', 0.0)
    if oov_ratio > 0.5 and perplexity > threshold * 0.5:
        return True, 'high_oov_and_perplexity'

    return False, 'kenlm_ok'
