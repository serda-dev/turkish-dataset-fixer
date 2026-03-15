"""
test_smoke.py — Smoke tests for the dataset-fixer pipeline.

Tests core components of the remote source/sink pipeline:
  - Source type detection
  - Dataset file discovery (including .gz and .parquet)
  - Output sharder (configurable shard size)
  - Manifest (load/save/resume)
  - Dedup integration (exact duplicate removal)
  - End-to-end local pipeline run

Run:
  python -m pytest tests/test_smoke.py -v
  # or simply:
  python tests/test_smoke.py
"""

import gzip
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


def test_source_type_detection():
    """Verify resolve_source correctly identifies source type."""
    from pipeline.sources import detect_source_type

    # HF repos
    assert detect_source_type('username/dataset') == 'huggingface'
    assert detect_source_type('serda-dev/turkish-raw-text-cleaned') == 'huggingface'
    assert detect_source_type('https://huggingface.co/datasets/user/repo') == 'huggingface'

    # GitHub repos
    assert detect_source_type('https://github.com/user/repo') == 'github'
    assert detect_source_type('https://github.com/user/repo.git') == 'github'

    # Local paths
    assert detect_source_type('/tmp') == 'local'
    assert detect_source_type('.') == 'local'

    print("  ✓ Source type detection: PASS")


def test_dataset_discovery():
    """Verify recursive file discovery with gz and parquet support."""
    from pipeline.dataset_discovery import (
        discover_dataset_files,
        iterate_records,
        open_data_file,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create nested structure
        (root / 'sub1').mkdir()
        (root / 'sub1' / 'sub2').mkdir()

        # Create various file types
        (root / 'data.jsonl').write_text('{"text": "hello"}\n')
        (root / 'data.json').write_text('{"text": "world"}\n')
        (root / 'sub1' / 'nested.jsonl').write_text('{"text": "nested"}\n')
        (root / 'sub1' / 'ignore.txt').write_text('not a dataset\n')

        # Create gzipped file
        gz_path = root / 'sub1' / 'sub2' / 'compressed.jsonl.gz'
        with gzip.open(gz_path, 'wt', encoding='utf-8') as f:
            f.write('{"text": "compressed"}\n')

        parquet_path = root / 'sub1' / 'dataset.parquet'
        if pa is not None and pq is not None:
            table = pa.Table.from_pylist([
                {'text': 'parquet row 1', 'source': 'test'},
                {'text': 'parquet row 2', 'source': 'test'},
            ])
            pq.write_table(table, parquet_path)

        # Discover
        files = discover_dataset_files(root)
        names = {f.name for f in files}

        assert 'data.jsonl' in names
        assert 'data.json' in names
        assert 'nested.jsonl' in names
        assert 'compressed.jsonl.gz' in names
        if pa is not None and pq is not None:
            assert 'dataset.parquet' in names
        assert 'ignore.txt' not in names
        expected_count = 5 if pa is not None and pq is not None else 4
        assert len(files) == expected_count

        # Test transparent gz reading
        with open_data_file(gz_path) as f:
            line = f.readline().strip()
            rec = json.loads(line)
            assert rec['text'] == 'compressed'

        # Test normal file reading
        with open_data_file(root / 'data.jsonl') as f:
            line = f.readline().strip()
            rec = json.loads(line)
            assert rec['text'] == 'hello'

        if pa is not None and pq is not None:
            parquet_records = list(iterate_records(parquet_path))
            assert len(parquet_records) == 2
            assert parquet_records[0]['text'] == 'parquet row 1'

    print("  ✓ Dataset discovery: PASS")


def test_local_shard_discovery_supports_parquet():
    """Verify local pipeline shard discovery picks up parquet inputs."""
    from pipeline.main import get_shard_files

    if pa is None or pq is None:
        print("  - Skipping parquet shard discovery check (pyarrow not installed)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pq.write_table(pa.Table.from_pylist([{'text': 'hello parquet'}]), root / 'sample.parquet')

        cfg = SimpleNamespace(input_dir=str(root))
        files = get_shard_files(cfg)

        assert len(files) == 1
        assert files[0].name == 'sample.parquet'

    print("  ✓ Local parquet shard discovery: PASS")


def test_output_sharder():
    """Verify shard writer produces shards near target size."""
    from pipeline.output_sharder import OutputSharder

    with tempfile.TemporaryDirectory() as tmpdir:
        completed_shards = []

        def on_complete(path, name):
            completed_shards.append(name)

        # Use very small target (1 KB) for testing
        sharder = OutputSharder(
            output_dir=Path(tmpdir),
            target_mb=0,  # will use 0 bytes target → roll on every record
            prefix='test-shard',
            on_shard_complete=on_complete,
        )
        # Override target to 1 KB for practical test
        sharder.target_bytes = 1024

        # Write records that total ~3 KB
        for i in range(50):
            record = {'text': f'Record number {i} with some padding text ' * 3}
            sharder.write_record(record)

        sharder.close()

        # Should have produced multiple shards
        stats = sharder.stats
        assert stats['total_records_written'] == 50
        assert stats['total_shards_produced'] >= 2
        assert len(completed_shards) >= 2

        # Verify shard files exist
        shard_files = list(Path(tmpdir).glob('test-shard-*.jsonl'))
        assert len(shard_files) == stats['total_shards_produced']

        # Verify content is valid JSONL
        total_records = 0
        for sf in shard_files:
            with open(sf, 'r') as f:
                for line in f:
                    rec = json.loads(line.strip())
                    assert 'text' in rec
                    total_records += 1
        assert total_records == 50

    print("  ✓ Output sharder: PASS")


def test_manifest():
    """Verify manifest load/save/resume logic."""
    from pipeline.manifest import ProcessingManifest

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = str(Path(tmpdir) / 'manifest.json')

        # Create and populate
        m1 = ProcessingManifest(manifest_path)
        m1.mark_file_done('/data/file1.jsonl')
        m1.mark_file_done('/data/file2.jsonl')
        m1.mark_shard_produced('shard-00000.jsonl')
        m1.mark_shard_uploaded('shard-00000.jsonl')
        m1.record_error('/data/bad.jsonl', 'parse error')
        m1.save()

        assert m1.is_file_done('/data/file1.jsonl')
        assert not m1.is_file_done('/data/file3.jsonl')
        assert m1.is_shard_uploaded('shard-00000.jsonl')
        assert not m1.is_shard_uploaded('shard-00001.jsonl')

        # Load from disk (simulating restart)
        m2 = ProcessingManifest(manifest_path)
        loaded = m2.load()
        assert loaded

        assert m2.is_file_done('/data/file1.jsonl')
        assert m2.is_file_done('/data/file2.jsonl')
        assert not m2.is_file_done('/data/file3.jsonl')
        assert m2.is_shard_uploaded('shard-00000.jsonl')
        assert m2.processed_count == 2
        assert m2.uploaded_count == 1
        assert m2.error_count == 1

    print("  ✓ Manifest: PASS")


def test_dedup_integration():
    """Verify ExactDeduplicator removes exact duplicates."""
    from pipeline.dedup import ExactDeduplicator

    dedup = ExactDeduplicator()

    # First occurrence — not a duplicate
    assert dedup.is_duplicate("Hello World") is False
    # Exact duplicate
    assert dedup.is_duplicate("Hello World") is True
    # Different text
    assert dedup.is_duplicate("Hello Mars") is False
    # Whitespace-normalized duplicate
    assert dedup.is_duplicate("  Hello World  ") is True

    stats = dedup.stats
    assert stats['total_checked'] == 4
    assert stats['unique_count'] == 2
    assert stats['duplicate_count'] == 2

    print("  ✓ Dedup integration: PASS")


def test_local_source():
    """Verify LocalSource works correctly."""
    from pipeline.sources import LocalSource

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / 'test.jsonl').write_text('{"text": "test"}\n')

        source = LocalSource(tmpdir)
        resolved = source.prepare()
        assert resolved == root

        files = source.discover_files()
        assert len(files) == 1
        assert files[0].name == 'test.jsonl'

    print("  ✓ Local source: PASS")


def test_sink_local():
    """Verify LocalSink basic operation."""
    from pipeline.sinks import LocalSink

    with tempfile.TemporaryDirectory() as tmpdir:
        sink = LocalSink(tmpdir)
        sink.ensure_dir()

        # Create a dummy shard
        shard_path = Path(tmpdir) / 'test-shard.jsonl'
        shard_path.write_text('{"text": "test"}\n')

        sink.write_shard(shard_path, 'test-shard.jsonl')
        sink.finalize()

        stats = sink.stats
        assert stats['shards_written'] == 1
        assert stats['sink_type'] == 'local'

    print("  ✓ Local sink: PASS")


def test_end_to_end_local():
    """
    End-to-end test: create temp input with JSONL data,
    run discovery + dedup + sharder, verify output.
    """
    from pipeline.dataset_discovery import discover_dataset_files, open_data_file
    from pipeline.dedup import ExactDeduplicator
    from pipeline.output_sharder import OutputSharder

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        output_dir = Path(tmpdir) / 'output'
        input_dir.mkdir()
        output_dir.mkdir()

        # Create test data with duplicates
        records = [
            {"text": "Bu bir test cümlesidir."},
            {"text": "Türkçe metin kalite kontrolü."},
            {"text": "Bu bir test cümlesidir."},  # exact duplicate
            {"text": "Farklı bir metin parçası."},
            {"text": "  Bu bir test cümlesidir.  "},  # whitespace-normalized duplicate
        ]
        with open(input_dir / 'test.jsonl', 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        # Discover
        files = discover_dataset_files(input_dir)
        assert len(files) == 1

        # Process with dedup
        dedup = ExactDeduplicator()
        kept = []

        for fpath in files:
            with open_data_file(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = rec.get('text', '')
                    if not dedup.is_duplicate(text):
                        kept.append(rec)

        # Should have removed 2 duplicates
        assert len(kept) == 3, f"Expected 3 unique records, got {len(kept)}"

        # Write to sharder
        sharder = OutputSharder(output_dir=output_dir, target_mb=1)
        for rec in kept:
            sharder.write_record(rec)
        sharder.close()

        # Verify output
        stats = sharder.stats
        assert stats['total_records_written'] == 3
        assert stats['total_shards_produced'] >= 1

        # Verify shard content
        total = 0
        for sf in output_dir.glob('filtered-shard-*.jsonl'):
            with open(sf) as f:
                for line in f:
                    json.loads(line.strip())
                    total += 1
        assert total == 3

    print("  ✓ End-to-end local: PASS")


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 50)
    print("  SMOKE TESTS — dataset-fixer pipeline")
    print("=" * 50 + "\n")

    tests = [
        test_source_type_detection,
        test_dataset_discovery,
        test_local_shard_discovery_supports_parquet,
        test_output_sharder,
        test_manifest,
        test_dedup_integration,
        test_local_source,
        test_sink_local,
        test_end_to_end_local,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: FAIL — {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}\n")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
