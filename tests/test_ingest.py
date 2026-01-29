"""Tests for repository ingestion module."""

import json
import tempfile
from pathlib import Path

import pytest

from swarmcode_agent.ingest import IngestConfig, RepositoryIngester


@pytest.fixture
def sample_repo():
    """Create a sample repository structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)

        # Create Python files
        (repo / "main.py").write_text('''
def main():
    """Entry point."""
    print("Hello, World!")

if __name__ == "__main__":
    main()
''')

        # Create subdirectory with modules
        src = repo / "src"
        src.mkdir()

        (src / "__init__.py").write_text("")

        (src / "utils.py").write_text('''
import os

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class Calculator:
    """Simple calculator."""

    def multiply(self, a, b):
        return a * b
''')

        # Create config file
        (repo / "config.yaml").write_text("key: value\n")

        # Create file to exclude
        pycache = repo / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.pyc").write_text("binary")

        yield repo


class TestRepositoryIngester:
    """Tests for RepositoryIngester."""

    def test_ingest_local_repo(self, sample_repo):
        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            assert result.snapshot_path.exists()
            assert result.code_map_path.exists()
            assert result.file_count > 0

    def test_ingest_creates_code_map(self, sample_repo):
        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            # Load and verify code map
            with open(result.code_map_path) as f:
                code_map = json.load(f)

            assert "files" in code_map
            assert "symbols" in code_map
            assert "tree" in code_map

    def test_ingest_extracts_functions(self, sample_repo):
        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            assert result.function_count > 0

            with open(result.code_map_path) as f:
                code_map = json.load(f)

            # Check that functions were extracted
            symbols = code_map.get("symbols", {})
            has_functions = any(
                len(s.get("functions", [])) > 0
                for s in symbols.values()
            )
            assert has_functions

    def test_ingest_extracts_classes(self, sample_repo):
        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            assert result.class_count > 0

    def test_ingest_excludes_pycache(self, sample_repo):
        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            # Check no __pycache__ in snapshot
            pycache_files = list(result.snapshot_path.rglob("__pycache__"))
            assert len(pycache_files) == 0

    def test_ingest_respects_max_file_size(self, sample_repo):
        # Create a large file
        large_file = sample_repo / "large.py"
        large_file.write_text("x" * 1000000)  # ~1MB

        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
                max_file_size_kb=500,  # 500KB limit
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            # Large file should be excluded
            with open(result.code_map_path) as f:
                code_map = json.load(f)

            file_paths = [f["path"] for f in code_map["files"]]
            assert "large.py" not in file_paths

    def test_ingest_filters_extensions(self, sample_repo):
        # Create a file with excluded extension
        (sample_repo / "data.csv").write_text("a,b,c\n1,2,3")

        with tempfile.TemporaryDirectory() as output_dir:
            config = IngestConfig(
                source=str(sample_repo),
                output_dir=Path(output_dir),
            )

            ingester = RepositoryIngester(config)
            result = ingester.ingest()

            with open(result.code_map_path) as f:
                code_map = json.load(f)

            file_paths = [f["path"] for f in code_map["files"]]
            assert "data.csv" not in file_paths

    def test_ingest_nonexistent_path(self):
        config = IngestConfig(
            source="/nonexistent/path",
            output_dir=Path("./output"),
        )

        ingester = RepositoryIngester(config)

        with pytest.raises(ValueError, match="does not exist"):
            ingester.ingest()
