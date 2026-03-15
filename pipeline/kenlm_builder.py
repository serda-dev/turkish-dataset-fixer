"""
kenlm_builder.py — Build a Turkish KenLM n-gram language model.

Extracts clean Turkish text from the dataset after heuristic + language
filtering, then trains a 5-gram KenLM model using lmplz and build_binary.

5-gram justification:
  Turkish is agglutinative — words carry suffixes that change meaning.
  5-grams capture multi-word patterns and common suffix chains.
  3-gram is too short for Turkish morphological patterns.
  5-gram balances model size vs discriminative power.
"""

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import List

from pipeline.text_normalization import normalize_text
from pipeline.heuristic_features import compute_features, apply_heuristic_filters
from pipeline.language_validation import validate_language

logger = logging.getLogger(__name__)


def _tokenize_for_lm(text: str) -> str:
    """
    Simple whitespace tokenization and lowercasing for KenLM training.
    KenLM expects one sentence per line.
    """
    # Split into sentences (rough)
    import re
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    result_lines = []
    for sent in sentences:
        tokens = sent.lower().split()
        if len(tokens) >= 3:  # skip very short fragments
            result_lines.append(' '.join(tokens))
    return '\n'.join(result_lines)


def build_seed_corpus(cfg, shard_files: List[Path]) -> str:
    """
    Build a clean Turkish seed corpus by filtering sample shards.

    Process:
    1. Sample a few shards (cfg.seed_corpus_shards)
    2. Apply heuristic + language filters
    3. Take only clean-passing records
    4. Write one-sentence-per-line text file for KenLM

    Returns: path to seed corpus file.
    """
    seed_path = cfg.kenlm_seed_corpus_path
    os.makedirs(os.path.dirname(seed_path), exist_ok=True)

    # Sample shards evenly
    if len(shard_files) <= cfg.seed_corpus_shards:
        sample_shards = shard_files
    else:
        step = len(shard_files) // cfg.seed_corpus_shards
        sample_shards = shard_files[::step][:cfg.seed_corpus_shards]

    logger.info("Building seed corpus from %d shards: %s",
                len(sample_shards), [s.name for s in sample_shards])

    total_records = 0
    accepted_records = 0
    total_sentences = 0

    with open(seed_path, 'w', encoding='utf-8') as out_f:
        for shard_path in sample_shards:
            logger.info("Processing shard for seed corpus: %s", shard_path.name)
            with open(shard_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if accepted_records >= cfg.seed_corpus_max_records:
                        break

                    total_records += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = rec.get(cfg.text_key, '')
                    if not isinstance(text, str) or not text.strip():
                        continue

                    # Normalize
                    text = normalize_text(text)
                    if not text:
                        continue

                    # Fast heuristic check
                    features = compute_features(text)
                    heuristic_reject, _ = apply_heuristic_filters(text, features, cfg)
                    if heuristic_reject:
                        continue

                    # Language check
                    lang_decision, _ = validate_language(text, features, cfg)
                    if lang_decision == 'reject':
                        continue

                    # Convert to LM training format
                    lm_text = _tokenize_for_lm(text)
                    if lm_text.strip():
                        out_f.write(lm_text + '\n')
                        accepted_records += 1
                        total_sentences += lm_text.count('\n') + 1

            if accepted_records >= cfg.seed_corpus_max_records:
                break

    corpus_size_mb = os.path.getsize(seed_path) / (1024 * 1024)
    logger.info("Seed corpus: %d accepted from %d total records, "
                "%d sentences, %.1f MB",
                accepted_records, total_records, total_sentences, corpus_size_mb)

    return seed_path


def train_kenlm_model(cfg) -> str:
    """
    Train KenLM model using lmplz and convert to binary format.

    Steps:
    1. Run lmplz to create ARPA file
    2. Run build_binary to create binary model
    3. Verify model loads

    Returns: path to binary model file.
    """
    lmplz_bin = os.path.join(cfg.kenlm_binary_path, 'lmplz')
    build_binary_bin = os.path.join(cfg.kenlm_binary_path, 'build_binary')
    seed_path = cfg.kenlm_seed_corpus_path
    arpa_path = cfg.kenlm_arpa_path
    binary_path = cfg.kenlm_model_path

    # Verify binaries exist
    for binary, name in [(lmplz_bin, 'lmplz'), (build_binary_bin, 'build_binary')]:
        if not os.path.exists(binary):
            raise FileNotFoundError(
                f"KenLM binary '{name}' not found at {binary}. "
                f"Build KenLM first: cd /tmp/kenlm_build/build && make -j$(nproc)"
            )

    # Verify seed corpus exists
    if not os.path.exists(seed_path):
        raise FileNotFoundError(
            f"Seed corpus not found at {seed_path}. Run build_seed_corpus first."
        )

    seed_size = os.path.getsize(seed_path)
    if seed_size < 1024:
        raise ValueError(f"Seed corpus too small ({seed_size} bytes). Need more clean data.")

    # ── Step 1: Train ARPA model ──────────────────────────────────────
    # Pruning: keep all unigrams/bigrams, prune singletons at 3,4,5-gram
    logger.info("Training %d-gram KenLM model...", cfg.kenlm_order)
    lmplz_cmd = [
        lmplz_bin,
        '-o', str(cfg.kenlm_order),
        '--prune', '0', '0', '1', '1', '1',
        '--discount_fallback',
    ]

    with open(seed_path, 'r') as stdin_f, open(arpa_path, 'w') as stdout_f:
        proc = subprocess.run(
            lmplz_cmd,
            stdin=stdin_f,
            stdout=stdout_f,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,  # 1 hour max
        )

    if proc.returncode != 0:
        logger.error("lmplz failed: %s", proc.stderr[-2000:] if proc.stderr else "no output")
        raise RuntimeError(f"lmplz failed with exit code {proc.returncode}")

    arpa_size_mb = os.path.getsize(arpa_path) / (1024 * 1024)
    logger.info("ARPA model: %.1f MB at %s", arpa_size_mb, arpa_path)

    # ── Step 2: Convert to binary ─────────────────────────────────────
    logger.info("Converting ARPA to binary format...")
    build_cmd = [build_binary_bin, arpa_path, binary_path]

    proc = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if proc.returncode != 0:
        logger.error("build_binary failed: %s", proc.stderr[-2000:] if proc.stderr else "")
        raise RuntimeError(f"build_binary failed with exit code {proc.returncode}")

    binary_size_mb = os.path.getsize(binary_path) / (1024 * 1024)
    logger.info("Binary model: %.1f MB at %s", binary_size_mb, binary_path)

    # ── Step 3: Quick verification ────────────────────────────────────
    try:
        import kenlm
        model = kenlm.Model(binary_path)
        test_score = model.score("bu bir test cümlesidir", bos=True, eos=True)
        logger.info("Verification: test sentence score = %.2f", test_score)
        del model
    except Exception as e:
        logger.warning("Model verification warning: %s", e)

    return binary_path
