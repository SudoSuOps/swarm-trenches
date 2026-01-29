"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from swarmcode_cli.config import (
    IngestConfig,
    ServeConfig,
    SwarmCodeConfig,
    SynthConfig,
    TrainConfig,
    load_config,
    load_subconfig,
)


class TestIngestConfig:
    """Tests for IngestConfig."""

    def test_default_values(self):
        config = IngestConfig(source="https://github.com/test/repo.git")

        assert config.source == "https://github.com/test/repo.git"
        assert config.branch == "main"
        assert config.depth == 0
        assert ".git" in config.exclude_patterns
        assert ".py" in config.include_extensions

    def test_custom_values(self):
        config = IngestConfig(
            source="/local/path",
            branch="develop",
            depth=10,
            max_file_size_kb=1000,
        )

        assert config.source == "/local/path"
        assert config.branch == "develop"
        assert config.depth == 10
        assert config.max_file_size_kb == 1000


class TestSynthConfig:
    """Tests for SynthConfig."""

    def test_default_values(self):
        config = SynthConfig(snapshot_dir=Path("./snapshots"))

        assert config.snapshot_dir == Path("./snapshots")
        assert config.min_diff_lines == 3
        assert config.max_diff_lines == 500
        assert config.include_tests is True
        assert len(config.task_templates) > 0


class TestTrainConfig:
    """Tests for TrainConfig."""

    def test_default_values(self):
        config = TrainConfig()

        assert "Qwen" in config.model_name
        assert config.lora_r == 64
        assert config.lora_alpha == 16
        assert config.num_epochs == 3
        assert config.load_in_4bit is True

    def test_qwen_14b_config(self):
        config = TrainConfig(
            model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
            batch_size=2,
            gradient_accumulation_steps=8,
        )

        assert "14B" in config.model_name
        assert config.batch_size == 2


class TestServeConfig:
    """Tests for ServeConfig."""

    def test_default_values(self):
        config = ServeConfig()

        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.enable_tools is True
        assert config.adapter_path is None

    def test_custom_adapter(self):
        config = ServeConfig(
            adapter_path=Path("./artifacts/adapter"),
            port=9000,
        )

        assert config.adapter_path == Path("./artifacts/adapter")
        assert config.port == 9000


class TestSwarmCodeConfig:
    """Tests for global SwarmCodeConfig."""

    def test_default_values(self):
        config = SwarmCodeConfig()

        assert config.data_dir == Path("./data")
        assert config.artifacts_dir == Path("./artifacts")
        assert config.log_level == "INFO"


class TestLoadConfig:
    """Tests for config loading functions."""

    def test_load_config_no_file(self):
        config = load_config(None)

        assert isinstance(config, SwarmCodeConfig)

    def test_load_config_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "data_dir": "./custom/data",
                "log_level": "DEBUG",
            }, f)
            f.flush()

            config = load_config(Path(f.name))

            assert config.data_dir == Path("./custom/data")
            assert config.log_level == "DEBUG"

    def test_load_subconfig(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "source": "https://github.com/test/repo.git",
                "branch": "feature",
                "depth": 5,
            }, f)
            f.flush()

            config = load_subconfig(Path(f.name), IngestConfig)

            assert config.source == "https://github.com/test/repo.git"
            assert config.branch == "feature"
            assert config.depth == 5
