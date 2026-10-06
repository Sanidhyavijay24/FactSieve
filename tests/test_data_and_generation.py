"""
@file test_data_and_generation.py
@description Unit tests for data loading, benchmark dataset schema, caching, and prompt construction
@module tests/test_data_and_generation
"""

import tempfile
from pathlib import Path
import pytest
from src.cache import DiskCache
from src.data_loader import create_default_benchmark_dataset, load_benchmark_samples, save_benchmark_samples
from src.models import BenchmarkSample


def test_disk_cache_operations():
    """Verify set, get, persistence, and key determinism in DiskCache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / "test_cache.jsonl"
        cache = DiskCache(str(cache_file))

        payload = {"model": "gemini-2.5-flash", "prompt": "Hello test"}
        key = DiskCache.compute_key(payload)

        # Initially empty
        assert cache.get(key) is None
        assert len(cache) == 0

        # Set value
        cache.set(key, "Generated response text", metadata={"tag": "test"})
        assert cache.get(key) == "Generated response text"
        assert len(cache) == 1

        # Re-instantiate from disk to test persistence
        cache2 = DiskCache(str(cache_file))
        assert cache2.get(key) == "Generated response text"
        assert len(cache2) == 1


def test_benchmark_dataset_creation_and_loading():
    """Verify default benchmark dataset creation, schema validity, and reloading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bench_file = Path(tmpdir) / "benchmark.jsonl"
        samples = create_default_benchmark_dataset(output_path=str(bench_file), samples_per_setting=10)

        assert len(samples) == 30
        settings = {s.setting for s in samples}
        assert settings == {"zero_context", "noisy_context", "accurate_context"}

        # Verify reloading
        loaded_samples = load_benchmark_samples(str(bench_file))
        assert len(loaded_samples) == 30
        assert loaded_samples[0].id == samples[0].id
        assert loaded_samples[0].question == samples[0].question
        assert loaded_samples[0].reference == samples[0].reference


def test_benchmark_sample_integrity():
    """Ensure every sample has non-empty question, reference, and setting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bench_file = Path(tmpdir) / "benchmark.jsonl"
        samples = create_default_benchmark_dataset(output_path=str(bench_file))

        for sample in samples:
            assert isinstance(sample, BenchmarkSample)
            assert len(sample.question.strip()) > 0
            assert len(sample.reference.strip()) > 0
            assert sample.setting in ["zero_context", "noisy_context", "accurate_context"]
            assert sample.ground_truth_label in ["Entailment", "Neutral", "Contradiction"]
