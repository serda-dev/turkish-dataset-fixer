"""
heuristic_features.py — Stage 2: Compute features and fast heuristic filtering.

Computes a feature vector for each text sample and classifies obvious junk.
Only rejects clearly bad content — thresholds are intentionally conservative.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

# ── Turkish-specific constants ──────────────────────────────────────────
TURKISH_SPECIFIC_CHARS = set('çÇğĞıİöÖşŞüÜ')

# Common Turkish stopwords (for stopword coverage estimation)
TURKISH_STOPWORDS = {
    'bir', 've', 'bu', 'da', 'de', 'ile', 'için', 'olan', 'gibi',
    'daha', 'en', 'çok', 'ne', 'var', 'ben', 'sen', 'biz', 'siz',
    'o', 'onlar', 'ama', 'fakat', 'ancak', 'veya', 'ya', 'ki',
    'mi', 'mı', 'mu', 'mü', 'değil', 'her', 'kadar', 'sonra',
    'önce', 'şu', 'olarak', 'üzere', 'ise', 'hem', 'çünkü',
    'ayrıca', 'böyle', 'şekilde', 'oldu', 'olan', 'olduğu',
    'olduğunu', 'olduğundan', 'yapılan', 'tarafından', 'ilgili',
    'birçok', 'dolayı', 'nasıl', 'neden', 'zaman', 'yani',
    'ise', 'bile', 'artık', 'bunu', 'buna', 'bunun', 'bunlar',
    'şey', 'diğer', 'hangi', 'kendi', 'aynı', 'bazı', 'büyük',
    'küçük', 'iyi', 'kötü', 'yeni', 'eski', 'olan', 'olma',
    'olup', 'ederek', 'edilmiş', 'eden', 'etmek',
}

# Boilerplate/spam indicators
BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'cookie\s*(policy|kullan|tercih|ayar)',
        r'çerez\s*(politika|kullan|tercih|ayar)',
        r'gizlilik\s*(politika|sözleşm|bildirim)',
        r'privacy\s*policy',
        r'terms\s*(of|and)\s*(service|use)',
        r'kullanım\s*(koşul|şart|sözleşm)',
        r'aydınlatma\s*metni',
        r'kvkk',
        r'kişisel\s*veri',
        r'copyright\s*©',
        r'tüm\s*hakları\s*saklıdır',
        r'all\s*rights\s*reserved',
        r'ana\s*sayfa\s*[|>»]',
        r'site\s*haritası',
        r'iletişim\s*formu',
        r'hakkımızda\s*[|>»]',
        r'menü\s*kapat',
        r'abone\s*ol\w*\s*bülten',
        r'subscribe',
        r'click\s*here',
        r'read\s*more',
        r'devamını\s*oku',
    ]
]

# HTML/script/style remnant patterns
HTML_TAG_RE = re.compile(r'<[a-zA-Z/][^>]*>')
URL_RE = re.compile(r'https?://\S+')
EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

# Sentence-ending punctuation (rough estimate)
SENTENCE_END_RE = re.compile(r'[.!?…]\s')


def compute_features(text: str) -> Dict[str, float]:
    """
    Compute a comprehensive feature vector for the given text.
    All features are numeric (int or float).
    """
    if not text:
        return _empty_features()

    chars = list(text)
    char_count = len(text)

    # Word-level analysis
    words = text.split()
    word_count = len(words)

    # Line-level analysis
    lines = text.split('\n')
    line_count = len(lines)
    non_empty_lines = [l.strip() for l in lines if l.strip()]

    # Character class counts
    alpha_count = sum(1 for c in chars if c.isalpha())
    digit_count = sum(1 for c in chars if c.isdigit())
    space_count = sum(1 for c in chars if c.isspace())
    punct_count = sum(1 for c in chars if not c.isalnum() and not c.isspace())
    turkish_char_count = sum(1 for c in chars if c in TURKISH_SPECIFIC_CHARS)

    # Ratios (safe division)
    alpha_ratio = alpha_count / max(1, char_count)
    digit_ratio = digit_count / max(1, char_count)
    whitespace_ratio = space_count / max(1, char_count)
    punctuation_ratio = punct_count / max(1, char_count)
    turkish_char_ratio = turkish_char_count / max(1, char_count)

    # URL/email/HTML counts
    url_count = len(URL_RE.findall(text))
    email_count = len(EMAIL_RE.findall(text))
    html_tag_count = len(HTML_TAG_RE.findall(text))

    # Token diversity
    words_lower = [w.lower() for w in words]
    unique_tokens = set(words_lower)
    unique_token_ratio = len(unique_tokens) / max(1, word_count)

    # Repeated word analysis
    word_freqs = Counter(words_lower)
    if word_freqs:
        most_common_word, most_common_count = word_freqs.most_common(1)[0]
        repeated_word_frac = most_common_count / max(1, word_count)
    else:
        repeated_word_frac = 0.0

    # Repeated line analysis
    if non_empty_lines:
        line_freqs = Counter(non_empty_lines)
        unique_lines = len(line_freqs)
        repeated_line_ratio = 1.0 - (unique_lines / len(non_empty_lines))
    else:
        repeated_line_ratio = 0.0

    # Average word length
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
    else:
        avg_word_length = 0.0

    # Sentence count estimate
    sentence_count = len(SENTENCE_END_RE.findall(text)) + 1  # at least 1

    # Turkish stopword coverage
    if word_count > 0:
        stopword_hits = sum(1 for w in words_lower if w in TURKISH_STOPWORDS)
        stopword_coverage = stopword_hits / word_count
    else:
        stopword_coverage = 0.0

    # Boilerplate keyword hits
    boilerplate_hits = sum(1 for p in BOILERPLATE_PATTERNS if p.search(text))

    return {
        'char_count': char_count,
        'word_count': word_count,
        'line_count': line_count,
        'sentence_count': sentence_count,
        'alpha_ratio': round(alpha_ratio, 4),
        'digit_ratio': round(digit_ratio, 4),
        'punctuation_ratio': round(punctuation_ratio, 4),
        'whitespace_ratio': round(whitespace_ratio, 4),
        'turkish_char_ratio': round(turkish_char_ratio, 4),
        'url_count': url_count,
        'email_count': email_count,
        'html_tag_count': html_tag_count,
        'unique_token_ratio': round(unique_token_ratio, 4),
        'repeated_word_frac': round(repeated_word_frac, 4),
        'repeated_line_ratio': round(repeated_line_ratio, 4),
        'avg_word_length': round(avg_word_length, 2),
        'stopword_coverage': round(stopword_coverage, 4),
        'boilerplate_hits': boilerplate_hits,
    }


def _empty_features() -> Dict[str, float]:
    """Return zeroed features for empty text."""
    return {
        'char_count': 0, 'word_count': 0, 'line_count': 0,
        'sentence_count': 0, 'alpha_ratio': 0.0, 'digit_ratio': 0.0,
        'punctuation_ratio': 0.0, 'whitespace_ratio': 0.0,
        'turkish_char_ratio': 0.0, 'url_count': 0, 'email_count': 0,
        'html_tag_count': 0, 'unique_token_ratio': 0.0,
        'repeated_word_frac': 0.0, 'repeated_line_ratio': 0.0,
        'avg_word_length': 0.0, 'stopword_coverage': 0.0,
        'boilerplate_hits': 0,
    }


def apply_heuristic_filters(
    text: str,
    features: Dict[str, float],
    cfg  # PipelineConfig
) -> Tuple[bool, List[str]]:
    """
    Apply fast heuristic filters. Returns (reject, reasons).

    Only rejects OBVIOUSLY bad content. Does NOT aggressively filter.
    """
    reasons = []

    # Empty or whitespace-only
    if not text or not text.strip():
        return True, ['empty_text']

    cc = features['char_count']
    wc = features['word_count']

    # Too short
    if cc < cfg.min_text_length:
        reasons.append('too_short')
    if wc < cfg.min_word_count:
        reasons.append('too_few_words')

    # Pathologically long
    if cc > cfg.max_text_length:
        reasons.append('too_long')

    # Character composition
    if features['alpha_ratio'] < cfg.min_alpha_ratio:
        reasons.append('low_alpha_ratio')
    if features['digit_ratio'] > cfg.max_digit_ratio:
        reasons.append('high_digit_ratio')
    if features['punctuation_ratio'] > cfg.max_punctuation_ratio:
        reasons.append('high_punctuation_ratio')

    # URLs / emails / HTML
    if features['url_count'] > cfg.max_url_count:
        reasons.append('too_many_urls')
    if features['email_count'] > cfg.max_email_count:
        reasons.append('too_many_emails')
    if features['html_tag_count'] > cfg.max_html_tag_count:
        reasons.append('html_remnants')

    # Repetition
    if features['repeated_line_ratio'] > cfg.max_repeated_line_ratio:
        reasons.append('excessive_repeated_lines')
    if features['repeated_word_frac'] > cfg.max_repeated_word_frac:
        reasons.append('excessive_repeated_word')
    if features['unique_token_ratio'] < cfg.min_unique_token_ratio and wc > 20:
        reasons.append('low_token_diversity')

    # Average word length (corruption/gibberish detector)
    if wc > 10:
        if features['avg_word_length'] > cfg.max_avg_word_length:
            reasons.append('abnormal_avg_word_length_high')
        if features['avg_word_length'] < cfg.min_avg_word_length:
            reasons.append('abnormal_avg_word_length_low')

    # Boilerplate
    if features['boilerplate_hits'] > cfg.max_boilerplate_score:
        reasons.append('boilerplate_content')

    reject = len(reasons) > 0
    return reject, reasons
