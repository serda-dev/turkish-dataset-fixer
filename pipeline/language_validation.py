"""
language_validation.py — Stage 3: Turkish language validation.

Uses fastText lid.176.bin as primary language ID, with Turkish char
and stopword coverage as backup signals.
Falls back to langdetect if fastText model is unavailable.
"""

import os
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded models
_fasttext_model = None
_fasttext_path = None


def _load_fasttext(model_path: str):
    """Load fastText model (lazy, singleton)."""
    global _fasttext_model, _fasttext_path
    if _fasttext_model is not None and _fasttext_path == model_path:
        return _fasttext_model
    if not os.path.exists(model_path):
        logger.warning("fastText model not found at %s, will use langdetect fallback", model_path)
        return None
    try:
        import fasttext
        # Suppress fasttext warnings about deprecated load_model
        fasttext.FastText.eprint = lambda x: None
        # Fix numpy 2.x compatibility: monkey-patch np.array in fasttext
        try:
            import numpy as np
            _orig_array = np.array
            def _compat_array(*args, **kwargs):
                kwargs.pop('copy', None)
                return _orig_array(*args, **kwargs)
            np.array = _compat_array
        except Exception:
            pass
        _fasttext_model = fasttext.load_model(model_path)
        _fasttext_path = model_path
        logger.info("Loaded fastText model from %s", model_path)
        return _fasttext_model
    except Exception as e:
        logger.warning("Failed to load fastText model: %s, using langdetect", e)
        return None


def _fasttext_predict(model, text: str, k: int = 3) -> list:
    """
    Predict language using fastText.
    Returns list of (lang_code, confidence) tuples.
    """
    # fastText expects single-line input
    clean_text = text.replace('\n', ' ').strip()[:5000]  # limit length
    if not clean_text:
        return []
    try:
        labels, scores = model.predict(clean_text, k=k)
        results = []
        # Handle both numpy array and tuple returns
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        if isinstance(scores, (list, tuple)):
            for label, score in zip(labels, scores):
                lang = label.replace('__label__', '')
                results.append((lang, float(score)))
        return results
    except ValueError:
        # numpy 2.x compat: try alternative approach
        try:
            labels = model.predict(clean_text, k=1)[0]
            lang = labels[0].replace('__label__', '')
            return [(lang, 0.8)]  # approximate confidence
        except Exception:
            return []
    except Exception:
        return []


def _langdetect_predict(text: str) -> list:
    """
    Fallback language detection using langdetect.
    Returns list of (lang_code, confidence) tuples.
    """
    try:
        from langdetect import detect_langs
        clean_text = text[:5000]
        results = detect_langs(clean_text)
        return [(r.lang, r.prob) for r in results[:3]]
    except Exception:
        return []


def validate_language(
    text: str,
    features: Dict[str, float],
    cfg,  # PipelineConfig
) -> Tuple[str, Dict]:
    """
    Validate that text is Turkish.

    Returns:
        (decision, info) where:
        - decision: "accept", "reject", or "borderline"
        - info: dict with lang_id details
    """
    info = {
        'detected_lang': 'unknown',
        'confidence': 0.0,
        'top_languages': [],
        'turkish_char_ratio': features.get('turkish_char_ratio', 0.0),
        'stopword_coverage': features.get('stopword_coverage', 0.0),
        'method': 'none',
    }

    # ── Try fastText first ────────────────────────────────────────────
    ft_model = _load_fasttext(cfg.fasttext_model_path)

    if ft_model is not None:
        predictions = _fasttext_predict(ft_model, text)
        info['method'] = 'fasttext'
    else:
        predictions = _langdetect_predict(text)
        info['method'] = 'langdetect'

    if predictions:
        info['top_languages'] = predictions[:3]
        info['detected_lang'] = predictions[0][0]
        info['confidence'] = predictions[0][1]

    # ── Turkish signals ───────────────────────────────────────────────
    tr_char_ratio = features.get('turkish_char_ratio', 0.0)
    stopword_cov = features.get('stopword_coverage', 0.0)
    has_turkish_chars = tr_char_ratio >= cfg.min_turkish_char_ratio
    has_stopwords = stopword_cov >= cfg.min_stopword_coverage

    # ── Decision logic ────────────────────────────────────────────────
    detected = info['detected_lang']
    conf = info['confidence']

    # Check if Turkish is in top predictions
    tr_in_top = any(lang == 'tr' for lang, _ in predictions[:2])
    tr_confidence = 0.0
    for lang, score in predictions:
        if lang == 'tr':
            tr_confidence = score
            break

    # High confidence Turkish → accept
    if detected == 'tr' and conf >= cfg.lang_high_confidence:
        return 'accept', info

    # Medium confidence Turkish + backup signals → accept
    if tr_in_top and tr_confidence >= cfg.lang_min_confidence:
        if has_turkish_chars or has_stopwords:
            return 'accept', info

    # Strong backup signals even without lang ID confidence
    if has_turkish_chars and has_stopwords and tr_in_top:
        return 'accept', info

    # Confident non-Turkish with no Turkish signals → reject
    if detected != 'tr' and conf >= cfg.lang_reject_confidence:
        if not has_turkish_chars and not has_stopwords:
            return 'reject', info

    # If we have some Turkish signals but lang ID is uncertain → borderline
    if has_turkish_chars or has_stopwords or tr_in_top:
        return 'borderline', info

    # No Turkish signals at all and lang ID says something else
    if detected != 'tr' and not has_turkish_chars and not has_stopwords:
        return 'reject', info

    # Default: borderline (let soft scoring decide)
    return 'borderline', info
