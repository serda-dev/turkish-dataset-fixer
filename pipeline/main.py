#!/usr/bin/env python3
"""
main.py — CLI entry point for the Turkish Dataset Quality Filtering Pipeline.

Phases:
  inspect     — Inspect dataset structure without processing
  build-kenlm — Build KenLM model from clean seed corpus
  filter      — Run full filtering pipeline on all/selected shards
  report      — Generate reports from existing per-shard stats
  all         — Run everything (inspect → build-kenlm → filter → report)

Supports both local and remote (HF / GitHub) input sources and output sinks.

Usage:
  # Local → local (classic)
  python -m pipeline.main --input-dir input --output-dir output --phase all

  # HF repo → local
  python -m pipeline.main --input-repo user/dataset --output-dir output --phase filter

  # HF repo → HF repo
  python -m pipeline.main --input-repo user/raw --output-repo user/clean --phase filter

  # GitHub → HF repo
  python -m pipeline.main --input-repo https://github.com/user/repo --output-repo user/clean --phase filter
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from pipeline.config import PipelineConfig


def setup_logging(output_dir: str):
    """Configure logging to console and file."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pipeline.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        ]
    )


def download_fasttext_model(cfg):
    """Download fastText lid.176.bin if not present."""
    import urllib.request

    model_path = cfg.fasttext_model_path
    if os.path.exists(model_path):
        logging.info("fastText model already exists: %s", model_path)
        return

    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    logging.info("Downloading fastText language ID model (~126MB)...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        urllib.request.urlretrieve(url, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logging.info("Downloaded fastText model: %.1f MB at %s", size_mb, model_path)
    except Exception as e:
        logging.error("Failed to download fastText model: %s", e)
        logging.error("Please download manually: wget %s -O %s", url, model_path)
        raise


def get_shard_files(cfg, shard_filter: str = None) -> list:
    """Get list of shard files to process."""
    from pipeline.dataset_discovery import discover_dataset_files

    input_path = Path(cfg.input_dir)
    all_shards = discover_dataset_files(input_path)

    if shard_filter:
        # Support comma-separated names or glob
        if ',' in shard_filter:
            wanted = set(s.strip() for s in shard_filter.split(','))
            all_shards = [s for s in all_shards if s.name in wanted]
        else:
            all_shards = [s for s in all_shards if shard_filter in s.name]

    return all_shards


def ensure_kenlm_model_available(cfg, shard_files: list | None = None):
    """Ensure the KenLM model exists, auto-building it from available shards if needed."""
    if os.path.exists(cfg.kenlm_model_path):
        logging.info("KenLM model: %s", cfg.kenlm_model_path)
        return

    if shard_files:
        from pipeline.kenlm_builder import build_seed_corpus, train_kenlm_model

        logging.info(
            "KenLM model not found at %s. Building it from %d discovered input files.",
            cfg.kenlm_model_path,
            len(shard_files),
        )
        logging.info("Step 1/2: Building KenLM seed corpus...")
        build_seed_corpus(cfg, shard_files)
        logging.info("Step 2/2: Training KenLM model...")
        train_kenlm_model(cfg)

        if os.path.exists(cfg.kenlm_model_path):
            logging.info("KenLM model built successfully: %s", cfg.kenlm_model_path)
            return

    raise FileNotFoundError(
        "KenLM model not found.\n"
        f"Resolved model path: {cfg.kenlm_model_path}\n"
        f"Resolved output dir: {cfg.output_dir}\n"
        "Relative paths are resolved against the repository root.\n"
        "Automatic build was not possible. Build the model first with "
        "`--phase build-kenlm` or pass an explicit `--output-dir` / "
        "`kenlm_model_path` that contains `kenlm/model.binary`."
    )


def phase_inspect(cfg):
    """Run dataset inspection."""
    from pipeline.inspect_dataset import inspect_dataset

    logging.info("=" * 60)
    logging.info("  PHASE: Dataset Inspection")
    logging.info("=" * 60)

    result = inspect_dataset(cfg.input_dir)

    # Save inspection report
    report_path = cfg.reports_dir / 'inspection_report.json'
    os.makedirs(cfg.reports_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info("Inspection report saved: %s", report_path)

    return result


def phase_build_kenlm(cfg, shard_files: list):
    """Build KenLM model from seed corpus."""
    from pipeline.kenlm_builder import build_seed_corpus, train_kenlm_model

    logging.info("=" * 60)
    logging.info("  PHASE: Build KenLM Model")
    logging.info("=" * 60)

    # Check if model already exists
    if os.path.exists(cfg.kenlm_model_path):
        size_mb = os.path.getsize(cfg.kenlm_model_path) / (1024 * 1024)
        logging.info("KenLM model already exists: %s (%.1f MB)",
                    cfg.kenlm_model_path, size_mb)
        logging.info("Delete it to rebuild. Skipping.")
        return

    # Ensure fastText model is available for language validation
    download_fasttext_model(cfg)

    # Step 1: Build seed corpus
    logging.info("Step 1/2: Building seed corpus...")
    t0 = time.time()
    seed_path = build_seed_corpus(cfg, shard_files)
    t1 = time.time()
    logging.info("Seed corpus built in %.1f seconds", t1 - t0)

    # Step 2: Train KenLM model
    logging.info("Step 2/2: Training KenLM model...")
    t0 = time.time()
    model_path = train_kenlm_model(cfg)
    t1 = time.time()
    logging.info("KenLM model trained in %.1f seconds", t1 - t0)


def phase_filter(cfg, shard_files: list):
    """Run filtering pipeline on shards."""
    from pipeline.dedup import ExactDeduplicator, NearDeduplicator
    from pipeline.process_shard import process_shard
    from pipeline.reporting import save_per_shard_report, generate_global_report

    logging.info("=" * 60)
    logging.info("  PHASE: Filter Pipeline")
    logging.info("=" * 60)

    # Ensure fastText model
    download_fasttext_model(cfg)
    ensure_kenlm_model_available(cfg, shard_files)

    # Initialize deduplicators
    deduplicator = ExactDeduplicator() if cfg.enable_exact_dedup else None
    near_deduplicator = None
    if cfg.enable_near_dedup:
        near_deduplicator = NearDeduplicator(
            threshold=cfg.near_dedup_threshold,
            num_perm=cfg.near_dedup_num_perm,
            shingle_size=cfg.near_dedup_shingle_size,
        )

    logging.info("Processing %d shards...", len(shard_files))
    all_stats = []
    t_start = time.time()

    for i, shard_path in enumerate(shard_files):
        t0 = time.time()
        stats = process_shard(
            shard_path=shard_path,
            cfg=cfg,
            deduplicator=deduplicator,
            near_deduplicator=near_deduplicator,
            shard_index=i,
        )
        t1 = time.time()

        # Save per-shard report
        save_per_shard_report(stats, cfg)
        all_stats.append(stats)

        # Progress
        elapsed = t1 - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(shard_files) - i - 1) / rate if rate > 0 else 0
        logging.info(
            "Progress: %d/%d shards (%.1f%%) — %.1fs/shard — ETA: %.0fs",
            i + 1, len(shard_files),
            (i + 1) / len(shard_files) * 100,
            t1 - t0, eta
        )

    # Get dedup stats
    dedup_stats = {}
    if deduplicator:
        dedup_stats.update(deduplicator.stats)
    if near_deduplicator:
        dedup_stats.update(near_deduplicator.stats)

    # Generate global report
    logging.info("Generating global report...")
    global_report = generate_global_report(all_stats, dedup_stats, cfg)

    total_time = time.time() - t_start
    logging.info("=" * 60)
    logging.info("  FILTERING COMPLETE")
    logging.info("  Total time: %.1f seconds (%.1f min)", total_time, total_time / 60)
    logging.info("  Kept:     %d (%.1f%%)", global_report['total_kept'],
                global_report['keep_ratio'] * 100)
    logging.info("  Rejected: %d (%.1f%%)", global_report['total_rejected'],
                global_report['reject_ratio'] * 100)
    logging.info("=" * 60)

    return global_report


# ═══════════════════════════════════════════════════════════════════════
#  REMOTE-AWARE FILTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def phase_remote_filter(cfg):
    """
    Remote-aware filtering pipeline.

    Handles the full flow:
      1. Resolve source (local / HF / GitHub)
      2. Download if remote
      3. Discover dataset files (recursive, gz support)
      4. Stream records through filter pipeline
      5. Write output shards (configurable size)
      6. Upload to sink if remote
      7. Track progress via manifest for resume
    """
    from pipeline.dataset_discovery import iterate_records
    from pipeline.dedup import ExactDeduplicator, NearDeduplicator
    from pipeline.manifest import ProcessingManifest
    from pipeline.output_sharder import OutputSharder
    from pipeline.sinks import resolve_sink
    from pipeline.sources import resolve_source
    from pipeline.text_normalization import normalize_text
    from pipeline.heuristic_features import compute_features, apply_heuristic_filters
    from pipeline.language_validation import validate_language
    from pipeline.kenlm_scorer import score_text, evaluate_kenlm_quality
    from pipeline.decision_logic import make_decision

    logging.info("=" * 60)
    logging.info("  PHASE: Remote-Aware Filter Pipeline")
    logging.info("=" * 60)

    t_start = time.time()

    # ── 1. Resolve source ─────────────────────────────────────────────
    source = resolve_source(
        input_dir=cfg.input_dir if not cfg.input_repo else None,
        input_repo=cfg.input_repo or None,
        hf_token=cfg.hf_token,
        github_token=cfg.github_token,
        cache_dir=cfg.cache_dir,
    )
    source_dir = source.prepare()
    logging.info("Source type: %s, resolved to: %s", source.source_type, source_dir)

    # ── 2. Discover dataset files ─────────────────────────────────────
    dataset_files = source.discover_files()
    if not dataset_files:
        logging.error("No dataset files found in source!")
        return None

    logging.info("Discovered %d dataset files", len(dataset_files))

    # Ensure fastText model and KenLM model after source discovery so we can
    # build the language model from the actual remote dataset if needed.
    download_fasttext_model(cfg)
    ensure_kenlm_model_available(cfg, dataset_files)

    # ── 3. Resolve sink ───────────────────────────────────────────────
    sink = resolve_sink(
        output_dir=cfg.output_dir if not cfg.output_repo else cfg.filtered_dir,
        output_repo=cfg.output_repo or None,
        hf_token=cfg.hf_token,
        private=cfg.upload_private,
    )
    sink.ensure_dir()

    # ── 4. Initialize manifest ────────────────────────────────────────
    manifest = ProcessingManifest(cfg.manifest_path)
    if cfg.enable_resume:
        manifest.load()

    # ── 5. Initialize deduplicators ───────────────────────────────────
    deduplicator = ExactDeduplicator() if cfg.enable_exact_dedup else None
    near_deduplicator = None
    if cfg.enable_near_dedup:
        near_deduplicator = NearDeduplicator(
            threshold=cfg.near_dedup_threshold,
            num_perm=cfg.near_dedup_num_perm,
            shingle_size=cfg.near_dedup_shingle_size,
        )

    # ── 6. Initialize output sharder ──────────────────────────────────
    def on_shard_complete(shard_path: Path, shard_name: str):
        """Called when a shard is complete: upload + mark in manifest."""
        if cfg.incremental_upload:
            sink.write_shard(shard_path, shard_name)
            manifest.mark_shard_uploaded(shard_name)
            manifest.mark_shard_produced(shard_name)
            manifest.save()

    output_dir_for_shards = cfg.filtered_dir
    Path(output_dir_for_shards).mkdir(parents=True, exist_ok=True)

    sharder = OutputSharder(
        output_dir=output_dir_for_shards,
        target_mb=cfg.target_shard_size_mb,
        on_shard_complete=on_shard_complete,
    )

    # ── 7. Process files ──────────────────────────────────────────────
    pipeline_stats = {
        'total_files_discovered': len(dataset_files),
        'files_processed': 0,
        'files_skipped_resume': 0,
        'files_errored': 0,
        'total_records_read': 0,
        'records_kept': 0,
        'records_rejected': 0,
        'records_malformed': 0,
        'records_missing_text': 0,
        'exact_duplicates': 0,
        'near_duplicates': 0,
        'rejection_reasons': {},
    }

    for file_idx, file_path in enumerate(dataset_files):
        file_key = str(file_path)

        # Resume: skip already-processed files
        if cfg.enable_resume and manifest.is_file_done(file_key):
            pipeline_stats['files_skipped_resume'] += 1
            logging.info("Skipping (already processed): %s", file_path.name)
            continue

        logging.info(
            "Processing file %d/%d: %s",
            file_idx + 1, len(dataset_files), file_path.name,
        )

        try:
            for line_num, record in enumerate(
                iterate_records(file_path, text_key=cfg.text_key)
            ):
                pipeline_stats['total_records_read'] += 1

                if record.get('_malformed', False):
                    pipeline_stats['records_malformed'] += 1
                    continue

                # Extract text
                text = record.get(cfg.text_key)
                if text is None:
                    pipeline_stats['records_missing_text'] += 1
                    continue
                if not isinstance(text, str):
                    text = str(text)

                # Stage 1: Normalize
                normalized_text = normalize_text(text)

                # Stage 2: Heuristic features
                features = compute_features(normalized_text)
                heuristic_reject, heuristic_reasons = apply_heuristic_filters(
                    normalized_text, features, cfg
                )

                # Stage 3: Language validation
                lang_decision, lang_info = validate_language(
                    normalized_text, features, cfg
                )

                # Stage 4: KenLM scoring
                kenlm_result = score_text(normalized_text, cfg)
                kenlm_reject, kenlm_reason = evaluate_kenlm_quality(
                    kenlm_result, cfg
                )

                # Stage 5: Dedup
                is_exact_dup = False
                is_near_dup = False

                if deduplicator and cfg.enable_exact_dedup:
                    is_exact_dup = deduplicator.is_duplicate(normalized_text)
                    if is_exact_dup:
                        pipeline_stats['exact_duplicates'] += 1

                if near_deduplicator and cfg.enable_near_dedup and not is_exact_dup:
                    doc_id = f"{file_path.name}:{line_num}"
                    is_near_dup = near_deduplicator.is_near_duplicate(
                        normalized_text, doc_id
                    )
                    if is_near_dup:
                        pipeline_stats['near_duplicates'] += 1

                # Stage 6: Decision
                decision_result = make_decision(
                    text=normalized_text,
                    features=features,
                    heuristic_reject=heuristic_reject,
                    heuristic_reasons=heuristic_reasons,
                    lang_decision=lang_decision,
                    lang_info=lang_info,
                    kenlm_result=kenlm_result,
                    kenlm_reject=(
                        kenlm_reject if kenlm_reject is not None else False
                    ),
                    kenlm_reason=kenlm_reason,
                    is_exact_dup=is_exact_dup,
                    is_near_dup=is_near_dup,
                    cfg=cfg,
                )

                if decision_result['decision'] == 'keep':
                    pipeline_stats['records_kept'] += 1
                    out_record = dict(record)
                    out_record[cfg.text_key] = normalized_text
                    sharder.write_record(out_record, text_key=cfg.text_key)
                else:
                    pipeline_stats['records_rejected'] += 1
                    for reason in decision_result.get('reasons', []):
                        pipeline_stats['rejection_reasons'][reason] = (
                            pipeline_stats['rejection_reasons'].get(reason, 0) + 1
                        )

            # Mark file as processed
            pipeline_stats['files_processed'] += 1
            manifest.mark_file_done(file_key)
            manifest.save()

            logging.info(
                "File done: %s | kept=%d rejected=%d total_read=%d",
                file_path.name,
                pipeline_stats['records_kept'],
                pipeline_stats['records_rejected'],
                pipeline_stats['total_records_read'],
            )

        except Exception as e:
            pipeline_stats['files_errored'] += 1
            manifest.record_error(file_key, str(e))
            manifest.save()
            logging.error("Error processing %s: %s — skipping", file_path.name, e)
            continue

    # ── 8. Finalize ───────────────────────────────────────────────────
    sharder.close()
    sink.finalize()

    # Dedup stats
    dedup_stats = {}
    if deduplicator:
        dedup_stats.update(deduplicator.stats)
    if near_deduplicator:
        dedup_stats.update(near_deduplicator.stats)

    total_time = time.time() - t_start

    # ── 9. Generate summary report ────────────────────────────────────
    sharder_stats = sharder.stats
    sink_stats = sink.stats

    summary = {
        'source_type': source.source_type,
        'sink_type': sink.sink_type,
        'total_time_seconds': round(total_time, 1),
        'total_time_minutes': round(total_time / 60, 1),
        **pipeline_stats,
        'dedup_stats': dedup_stats,
        'sharder_stats': sharder_stats,
        'sink_stats': sink_stats,
    }

    # Save summary report
    report_dir = cfg.reports_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = report_dir / 'remote_pipeline_summary.json'
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_txt_path = report_dir / 'remote_pipeline_summary.txt'
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  REMOTE PIPELINE — SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source type:           {source.source_type}\n")
        f.write(f"Sink type:             {sink.sink_type}\n")
        f.write(f"Total time:            {total_time:.1f}s ({total_time/60:.1f} min)\n\n")
        f.write(f"Files discovered:      {pipeline_stats['total_files_discovered']}\n")
        f.write(f"Files processed:       {pipeline_stats['files_processed']}\n")
        f.write(f"Files skipped (resume):{pipeline_stats['files_skipped_resume']}\n")
        f.write(f"Files errored:         {pipeline_stats['files_errored']}\n\n")
        f.write(f"Records read:          {pipeline_stats['total_records_read']:,}\n")
        f.write(f"Records kept:          {pipeline_stats['records_kept']:,}\n")
        f.write(f"Records rejected:      {pipeline_stats['records_rejected']:,}\n")
        f.write(f"Records malformed:     {pipeline_stats['records_malformed']:,}\n")
        f.write(f"Records missing text:  {pipeline_stats['records_missing_text']:,}\n")
        f.write(f"Exact duplicates:      {pipeline_stats['exact_duplicates']:,}\n")
        f.write(f"Near duplicates:       {pipeline_stats['near_duplicates']:,}\n\n")
        f.write(f"Shards produced:       {sharder_stats['total_shards_produced']}\n")
        f.write(f"Target shard size:     {sharder_stats['target_shard_size_mb']} MB\n")
        if sharder_stats['shard_sizes_mb']:
            f.write("Shard sizes (MB):     " +
                    ", ".join(f"{s:.1f}" for s in sharder_stats['shard_sizes_mb']) + "\n")
        f.write("\n")

        if pipeline_stats['rejection_reasons']:
            f.write("Top Rejection Reasons:\n")
            f.write("-" * 40 + "\n")
            sorted_reasons = sorted(
                pipeline_stats['rejection_reasons'].items(),
                key=lambda x: x[1], reverse=True,
            )
            for reason, count in sorted_reasons[:20]:
                f.write(f"  {reason:40s}  {count:>10,}\n")
            f.write("\n")

    logging.info("Summary report: %s", report_json_path)

    # Log final summary
    logging.info("=" * 60)
    logging.info("  REMOTE PIPELINE COMPLETE")
    logging.info("  Source: %s | Sink: %s", source.source_type, sink.sink_type)
    logging.info("  Time: %.1f seconds (%.1f min)", total_time, total_time / 60)
    logging.info("  Files: %d processed, %d skipped, %d errors",
                 pipeline_stats['files_processed'],
                 pipeline_stats['files_skipped_resume'],
                 pipeline_stats['files_errored'])
    logging.info("  Records: %d read, %d kept, %d rejected",
                 pipeline_stats['total_records_read'],
                 pipeline_stats['records_kept'],
                 pipeline_stats['records_rejected'])
    logging.info("  Dedup: %d exact, %d near",
                 pipeline_stats['exact_duplicates'],
                 pipeline_stats['near_duplicates'])
    logging.info("  Shards: %d produced (target %d MB)",
                 sharder_stats['total_shards_produced'],
                 sharder_stats['target_shard_size_mb'])
    logging.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Turkish Dataset Quality Filtering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on all shards (local → local)
  python -m pipeline.main --input-dir input --output-dir output --phase all

  # Inspect dataset only
  python -m pipeline.main --phase inspect

  # Build KenLM model only
  python -m pipeline.main --phase build-kenlm

  # Filter specific shards
  python -m pipeline.main --phase filter --shards "shard-00000.jsonl,shard-00001.jsonl"

  # HF repo → local
  python -m pipeline.main --input-repo user/dataset --output-dir output --phase filter

  # HF repo → HF repo
  python -m pipeline.main --input-repo user/raw --output-repo user/clean --phase filter

  # GitHub repo → HF repo
  python -m pipeline.main --input-repo https://github.com/user/repo --output-repo user/clean --phase filter

  # With dedup and custom shard size
  python -m pipeline.main --input-repo user/raw --output-repo user/clean --phase filter --target-shard-size-mb 55
        """
    )

    parser.add_argument('--input-dir', default='input',
                       help='Input directory with .json/.jsonl/.json.gz/.jsonl.gz/.parquet files (default: input)')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--phase', default='all',
                       choices=['inspect', 'build-kenlm', 'filter', 'all'],
                       help='Pipeline phase to run (default: all)')
    parser.add_argument('--shards', default=None,
                       help='Comma-separated shard names or substring filter')
    parser.add_argument('--kenlm-order', type=int, default=5,
                       help='KenLM n-gram order (default: 5)')
    parser.add_argument('--kenlm-binary-path', default='/tmp/kenlm_build/build/bin',
                       help='Path to KenLM binaries (lmplz, build_binary)')
    parser.add_argument('--no-dedup', action='store_true',
                       help='Disable exact deduplication')
    parser.add_argument('--near-dedup', action='store_true',
                       help='Enable near-deduplication (MinHash LSH)')
    parser.add_argument('--max-perplexity', type=float, default=3000.0,
                       help='KenLM max perplexity threshold (default: 3000)')
    parser.add_argument('--soft-threshold', type=float, default=0.25,
                       help='Soft quality score threshold (default: 0.25)')
    parser.add_argument('--seed-shards', type=int, default=3,
                       help='Number of shards for KenLM seed corpus (default: 3)')

    # ── Remote source / sink arguments ────────────────────────────────
    parser.add_argument('--input-repo', default='',
                       help='HF dataset/model repo or GitHub URL for input '
                            '(alternative to --input-dir)')
    parser.add_argument('--output-repo', default='',
                       help='HF repo ID for output upload '
                            '(alternative to --output-dir)')
    parser.add_argument('--hf-token', default='',
                       help='Hugging Face token (default: $HF_TOKEN env var)')
    parser.add_argument('--github-token', default='',
                       help='GitHub token (default: $GITHUB_TOKEN env var)')
    parser.add_argument('--cache-dir', default='.cache',
                       help='Cache directory for downloaded repos (default: .cache)')
    parser.add_argument('--target-shard-size-mb', type=int, default=55,
                       help='Target output shard size in MB (default: 55)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Disable manifest-based resume')
    parser.add_argument('--upload-private', action='store_true',
                       help='Create HF output repo as private')

    args = parser.parse_args()

    # ── Detect remote mode ────────────────────────────────────────────
    is_remote = bool(args.input_repo or args.output_repo)

    # Build config
    cfg = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        kenlm_order=args.kenlm_order,
        kenlm_binary_path=args.kenlm_binary_path,
        enable_exact_dedup=not args.no_dedup,
        enable_near_dedup=args.near_dedup,
        kenlm_max_perplexity=args.max_perplexity,
        soft_score_threshold=args.soft_threshold,
        seed_corpus_shards=args.seed_shards,
        input_repo=args.input_repo,
        output_repo=args.output_repo,
        hf_token=args.hf_token,
        github_token=args.github_token,
        cache_dir=args.cache_dir,
        target_shard_size_mb=args.target_shard_size_mb,
        enable_resume=not args.no_resume,
        upload_private=args.upload_private,
    )

    # Create output dirs
    cfg.ensure_dirs()

    # Setup logging
    setup_logging(cfg.output_dir)

    logging.info("=" * 60)
    logging.info("  TURKISH DATASET QUALITY FILTERING PIPELINE")
    logging.info("=" * 60)
    logging.info("Input:  %s", cfg.input_repo or cfg.input_dir)
    logging.info("Output: %s", cfg.output_repo or cfg.output_dir)
    logging.info("Phase:  %s", args.phase)
    logging.info("Remote: %s", is_remote)

    if is_remote:
        # ── Remote mode: source/sink aware pipeline ───────────────────
        if args.phase in ('all', 'filter'):
            phase_remote_filter(cfg)
        else:
            logging.warning(
                "Remote mode only supports 'filter' and 'all' phases. "
                "Falling back to local mode for phase '%s'.", args.phase
            )
            # Fall back to local mode for inspect / build-kenlm
            shard_files = get_shard_files(cfg, args.shards)
            logging.info("Shards: %d files", len(shard_files))

            if args.phase in ('all', 'inspect'):
                phase_inspect(cfg)
            if args.phase in ('all', 'build-kenlm'):
                phase_build_kenlm(cfg, shard_files)
    else:
        # ── Local mode: original behavior (unchanged) ─────────────────
        shard_files = get_shard_files(cfg, args.shards)
        logging.info("Shards: %d files", len(shard_files))

        if args.phase in ('all', 'inspect'):
            phase_inspect(cfg)

        if args.phase in ('all', 'build-kenlm', 'filter') and not shard_files:
            logging.error("No supported input files found!")
            sys.exit(1)

        if args.phase in ('all', 'build-kenlm'):
            phase_build_kenlm(cfg, shard_files)

        if args.phase in ('all', 'filter'):
            phase_filter(cfg, shard_files)

    logging.info("Done!")


if __name__ == '__main__':
    main()
