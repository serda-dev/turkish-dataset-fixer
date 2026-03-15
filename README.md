# Turkish Dataset Quality Filtering Pipeline

A modular, Turkish-aware pipeline for filtering large JSON, JSONL, and Parquet datasets using KenLM language model scoring, fastText language detection, heuristic filtering, and deduplication.

Supports both **local** and **remote** (Hugging Face / GitHub) input sources and output sinks. Designed for long-running VPS processes with manifest-based resume support.

## Quick Start

```bash
# 1. Install dependencies (KenLM must be built first — see below)
pip install -r requirements.txt

# 2. Run full pipeline (inspect → build KenLM → filter → report)
python -m pipeline.main --input-dir input --output-dir output --phase all

# 3. Smoke test on 1 shard/file
python -m pipeline.main --phase all --shards "shard-00000.jsonl" --seed-shards 1
```

## Phases

| Phase | Command | Description |
|---|---|---|
| `inspect` | `--phase inspect` | Scan dataset, report structure |
| `build-kenlm` | `--phase build-kenlm` | Build seed corpus + train 5-gram KenLM |
| `filter` | `--phase filter` | Run full filtering pipeline |
| `all` | `--phase all` | Run everything in order |

## CLI Options

### Core Options

```
--input-dir DIR       Input directory with .json/.jsonl/.json.gz/.jsonl.gz/.parquet files
--output-dir DIR      Output directory (default: output)
--phase PHASE         Pipeline phase: inspect|build-kenlm|filter|all
--shards FILTER       Comma-separated shard names or substring
--kenlm-order N       N-gram order (default: 5)
--max-perplexity F    KenLM rejection threshold (default: 3000)
--soft-threshold F    Quality score rejection threshold (default: 0.25)
--seed-shards N       Shards for seed corpus (default: 3)
--no-dedup            Disable exact deduplication
--near-dedup          Enable MinHash near-deduplication
```

### Remote Source / Sink Options

```
--input-repo REPO     HF dataset/model repo or GitHub URL (alternative to --input-dir)
--output-repo REPO    HF repo ID for output upload (alternative to --output-dir)
--hf-token TOKEN      Hugging Face token (default: $HF_TOKEN env var)
--github-token TOKEN  GitHub token (default: $GITHUB_TOKEN env var)
--cache-dir DIR       Cache directory for downloads (default: .cache)
--target-shard-size-mb N  Target output shard size in MB (default: 55)
--no-resume           Disable manifest-based resume
--upload-private      Create HF output repo as private
```

## Usage Examples

### Local → Local (classic)

```bash
python -m pipeline.main --input-dir input --output-dir output --phase all
```

### HF Repo → Local

```bash
export HF_TOKEN=hf_xxxxx
python -m pipeline.main \
  --input-repo username/dataset-repo \
  --output-dir ./output \
  --phase filter
```

### HF Repo → HF Repo

```bash
python -m pipeline.main \
  --input-repo username/raw-dataset \
  --output-repo username/filtered-dataset \
  --hf-token $HF_TOKEN \
  --phase filter
```

### GitHub Repo → HF Repo

```bash
python -m pipeline.main \
  --input-repo https://github.com/user/dataset-repo \
  --output-repo username/filtered-dataset \
  --hf-token $HF_TOKEN \
  --phase filter
```

### Dedup + Custom Shard Size

```bash
python -m pipeline.main \
  --input-repo username/raw-dataset \
  --output-repo username/filtered-dataset \
  --target-shard-size-mb 55 \
  --phase filter
```

### Disable Dedup + Disable Resume

```bash
python -m pipeline.main \
  --input-repo username/raw-dataset \
  --output-dir ./output \
  --no-dedup \
  --no-resume \
  --phase filter
```

## Output Structure

```
output/
├── filtered/        ← Accepted JSONL shards
├── rejected/        ← Rejected JSONL with reasons
├── reports/         ← JSON/TXT/CSV summaries
│   └── per_shard/
├── audit/           ← Human-readable samples
├── kenlm/           ← LM model + fastText model
├── manifest.json    ← Resume tracking (remote pipeline)
└── pipeline.log
```

## Remote Pipeline Features

- **Source auto-detection**: Automatically detects if input is a local path, HF repo, or GitHub URL
- **Recursive discovery**: Finds `.json`, `.jsonl`, `.json.gz`, `.jsonl.gz`, and `.parquet` files in nested directories
- **Configurable sharding**: Output shards target a configurable size (default: 55 MB)
- **Incremental upload**: Each shard is uploaded to HF as soon as it's produced
- **Manifest-based resume**: If interrupted, re-running skips already-processed files and uploaded shards
- **Token security**: Tokens from env vars (`HF_TOKEN`, `GITHUB_TOKEN`) or CLI args, never logged in full
- **VPS-resilient**: Designed for long-running background processes with per-file error handling

## Fast Hugging Face Upload

For large shard uploads, use the bundled uploader instead of `hf upload`:

```bash
pip install -r requirements.txt
hf auth login
python upload_hf_dataset.py serda-dev/turkish-raw-text-cleaned --source-dir output/filtered --workers 16
```

This uses `huggingface_hub.upload_large_folder()` plus `HF_XET_HIGH_PERFORMANCE=1` to speed up multi-file uploads and resume cleanly if interrupted.

## Dependencies

- Python 3.10+
- KenLM C++ tools (`lmplz`, `build_binary`) — build from [github.com/kpu/kenlm](https://github.com/kpu/kenlm)
- `kenlm` Python bindings (pip install, may need Cython regen for Python 3.13+)
- `fasttext` — language ID via `lid.176.bin` (auto-downloaded)
- `pyarrow` — required for streaming `.parquet` input
- `xxhash`, `datasketch`, `langdetect`
- `huggingface_hub` — for remote source/sink

### Building KenLM from Source

```bash
# Install cmake and boost
pip install cmake
conda install -y -c conda-forge boost-cpp

# Clone and build
git clone --depth 1 https://github.com/kpu/kenlm.git /tmp/kenlm_build
cd /tmp/kenlm_build && mkdir build && cd build
cmake .. && make -j$(nproc)

# Install Python bindings (for Python 3.13+, regenerate Cython first)
pip install cython
cd /tmp/kenlm_build && cython --cplus python/kenlm.pyx
pip install --no-build-isolation .
```

## Pipeline Stages

1. **Normalization** — Remove control chars, normalize whitespace (preserves Turkish)
2. **Heuristic filtering** — Reject obvious junk (too short, low alpha, HTML, boilerplate)
3. **Language validation** — fastText + Turkish char/stopword signals
4. **KenLM scoring** — 5-gram perplexity-based quality scoring
5. **Deduplication** — xxhash exact dedup (global across shards)
6. **Decision logic** — Layered hard+soft scoring with configurable thresholds

## Tuning

Key thresholds in `pipeline/config.py`:

| Parameter | Default | If too strict | If too lenient |
|---|---|---|---|
| `kenlm_max_perplexity` | 3000 | ↑ to 5000-8000 | ↓ to 1500-2000 |
| `soft_score_threshold` | 0.25 | ↓ to 0.15 | ↑ to 0.35 |
| `lang_min_confidence` | 0.30 | ↓ to 0.20 | ↑ to 0.50 |
| `min_text_length` | 50 | ↓ to 30 | ↑ to 100 |

Inspect `output/audit/rejected_sample.jsonl` to check for false positives.
