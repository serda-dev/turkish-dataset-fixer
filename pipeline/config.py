"""
config.py — Central configuration for the Turkish filtering pipeline.

All thresholds are intentionally lenient to minimize false positives.
Tune these values based on audit sample inspection.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """All configurable parameters for the pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────
    input_dir: str = "input"
    output_dir: str = "output"

    @property
    def filtered_dir(self) -> Path:
        return Path(self.output_dir) / "filtered"

    @property
    def rejected_dir(self) -> Path:
        return Path(self.output_dir) / "rejected"

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_dir) / "reports"

    @property
    def per_shard_reports_dir(self) -> Path:
        return Path(self.output_dir) / "reports" / "per_shard"

    @property
    def audit_dir(self) -> Path:
        return Path(self.output_dir) / "audit"

    @property
    def kenlm_dir(self) -> Path:
        return Path(self.output_dir) / "kenlm"

    @property
    def tmp_dir(self) -> Path:
        return Path(self.output_dir) / "tmp"

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.filtered_dir, self.rejected_dir, self.reports_dir,
                  self.per_shard_reports_dir, self.audit_dir,
                  self.kenlm_dir, self.tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── KenLM paths ───────────────────────────────────────────────────
    kenlm_binary_path: str = ""  # path to lmplz / build_binary binaries dir
    kenlm_model_path: str = ""   # auto-set to kenlm_dir / model.binary
    kenlm_arpa_path: str = ""    # auto-set to kenlm_dir / model.arpa
    kenlm_seed_corpus_path: str = ""  # auto-set to kenlm_dir / seed_corpus.txt
    kenlm_order: int = 5  # 5-gram: good for agglutinative Turkish morphology

    # ── FastText / Language ID ─────────────────────────────────────────
    fasttext_model_path: str = ""  # auto-set to kenlm_dir / lid.176.bin

    # ── Heuristic thresholds (Stage 2) — intentionally lenient ─────────
    min_text_length: int = 50           # chars — very short = junk
    max_text_length: int = 500_000      # chars — pathological length
    min_word_count: int = 5             # minimum words
    min_alpha_ratio: float = 0.40       # < 40% alphabetic = junk
    max_digit_ratio: float = 0.50       # > 50% digits = junk
    max_punctuation_ratio: float = 0.40 # > 40% punctuation = junk
    max_url_count: int = 20             # too many URLs
    max_email_count: int = 10           # too many emails
    max_html_tag_count: int = 10        # HTML remnants
    max_repeated_line_ratio: float = 0.70  # > 70% repeated lines
    max_repeated_word_frac: float = 0.50   # most-repeated word > 50% of all words
    min_unique_token_ratio: float = 0.10   # < 10% unique tokens
    max_avg_word_length: float = 45.0      # average word too long (corruption)
    min_avg_word_length: float = 1.5       # average word too short
    max_boilerplate_score: int = 5         # boilerplate keyword hits

    # ── Language validation thresholds (Stage 3) ───────────────────────
    lang_min_confidence: float = 0.3     # min fasttext Turkish confidence
    lang_high_confidence: float = 0.5    # high confidence = auto-accept
    lang_reject_confidence: float = 0.7  # non-Turkish with this confidence = reject
    min_turkish_char_ratio: float = 0.015  # min Turkish-specific char ratio
    min_stopword_coverage: float = 0.005   # min stopword coverage

    # ── KenLM quality scoring (Stage 4) ────────────────────────────────
    kenlm_max_perplexity: float = 3000.0   # reject above this perplexity
    kenlm_short_text_multiplier: float = 1.5  # relaxed threshold for short texts
    kenlm_min_tokens_for_scoring: int = 10    # skip KenLM for very short texts
    kenlm_short_text_threshold: int = 30      # tokens below this = "short"

    # ── Deduplication (Stage 5) ────────────────────────────────────────
    enable_exact_dedup: bool = True
    enable_near_dedup: bool = False     # optional, heavier
    near_dedup_threshold: float = 0.80  # Jaccard similarity threshold
    near_dedup_num_perm: int = 128      # MinHash permutations
    near_dedup_shingle_size: int = 5    # character n-gram shingle size

    # ── Soft quality scoring (Stage 6) ──────────────────────────────────
    soft_score_threshold: float = 0.25  # reject below this quality score
    # Weights for quality score components
    weight_kenlm: float = 0.40
    weight_lang_confidence: float = 0.30
    weight_heuristic: float = 0.30

    # ── Reporting & audit ──────────────────────────────────────────────
    audit_sample_size: int = 200        # records per audit file
    borderline_margin: float = 0.10     # +/- margin around threshold

    # ── KenLM seed corpus building ─────────────────────────────────────
    seed_corpus_shards: int = 3         # how many shards to sample for seed corpus
    seed_corpus_max_records: int = 100_000  # max records in seed corpus

    # ── Remote source / sink ──────────────────────────────────────────
    input_repo: str = ""                # HF or GitHub repo URL/ID
    output_repo: str = ""               # HF repo ID for output upload
    hf_token: str = ""                  # from env / CLI, never logged
    github_token: str = ""              # from env / CLI, never logged
    cache_dir: str = ".cache"            # temp download cache

    # ── Output sharding ──────────────────────────────────────────────
    target_shard_size_mb: int = 55       # target shard size for output

    # ── Resume support ───────────────────────────────────────────────
    enable_resume: bool = True           # manifest-based resume
    manifest_path: str = ""              # auto-set to output_dir/manifest.json

    # ── Remote pipeline options ──────────────────────────────────────
    upload_private: bool = False         # create HF repo as private
    incremental_upload: bool = True      # upload each shard as it's produced

    # ── Processing ─────────────────────────────────────────────────────
    text_key: str = "text"              # JSON key for text field

    def __post_init__(self):
        """Set derived paths and resolve tokens from env."""
        if not self.kenlm_model_path:
            self.kenlm_model_path = str(Path(self.output_dir) / "kenlm" / "model.binary")
        if not self.kenlm_arpa_path:
            self.kenlm_arpa_path = str(Path(self.output_dir) / "kenlm" / "model.arpa")
        if not self.kenlm_seed_corpus_path:
            self.kenlm_seed_corpus_path = str(Path(self.output_dir) / "kenlm" / "seed_corpus.txt")
        if not self.fasttext_model_path:
            self.fasttext_model_path = str(Path(self.output_dir) / "kenlm" / "lid.176.bin")
        if not self.kenlm_binary_path:
            self.kenlm_binary_path = "/tmp/kenlm_build/build/bin"

        # Resolve tokens from environment if not explicitly set
        if not self.hf_token:
            self.hf_token = os.environ.get('HF_TOKEN', '')
        if not self.github_token:
            self.github_token = os.environ.get('GITHUB_TOKEN', '')

        # Auto-set manifest path
        if not self.manifest_path:
            self.manifest_path = str(Path(self.output_dir) / 'manifest.json')
