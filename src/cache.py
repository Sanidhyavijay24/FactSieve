"""
@file cache.py
@description Deterministic disk-based caching layer for LLM calls and extraction artifacts
@module src/cache
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class DiskCache:
    """Thread-safe persistent JSONL cache keyed by SHA-256 hash."""

    def __init__(self, cache_file_path: str):
        """Initialize DiskCache with target file path.

        Args:
            cache_file_path (str): Relative or absolute path to the cache JSONL file.
        """
        self.cache_file = Path(cache_file_path)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._load_cache()

    @staticmethod
    def compute_key(payload: Any) -> str:
        """Compute deterministic SHA-256 hash string for payload.

        Args:
            payload (Any): Dict, string, or JSON-serializable structure.

        Returns:
            str: Hex digest SHA-256 hash.
        """
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load existing records from disk into in-memory dictionary."""
        if not self.cache_file.exists():
            return

        with open(self.cache_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry.get("__key__")
                    val = entry.get("__val__")
                    if key is not None:
                        self._memory_cache[key] = val
                except json.JSONDecodeError:
                    continue

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached entry by key.

        Args:
            key (str): SHA-256 key string.

        Returns:
            Optional[Any]: Cached value or None if cache miss.
        """
        return self._memory_cache.get(key)

    def set(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store key-value pair in cache and immediately append to disk.

        Args:
            key (str): SHA-256 key string.
            value (Any): JSON-serializable value.
            metadata (Optional[Dict[str, Any]]): Optional descriptive metadata for inspection.
        """
        self._memory_cache[key] = value
        record = {
            "__key__": key,
            "__val__": value,
            "__metadata__": metadata or {},
        }
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        """Returns total number of cached records."""
        return len(self._memory_cache)
