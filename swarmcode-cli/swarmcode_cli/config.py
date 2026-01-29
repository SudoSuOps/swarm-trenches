"""Configuration management for SwarmCode."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestConfig(BaseModel):
    """Configuration for repository ingestion."""

    source: str = Field(description="Git URL or local path to ingest")
    output_dir: Path = Field(default=Path("./data/snapshots"))
    branch: str = Field(default="main")
    depth: int = Field(default=0, description="Git clone depth, 0 for full history")
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "*.pyc",
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "*.egg-info",
            "dist",
            "build",
        ]
    )
    max_file_size_kb: int = Field(default=500, description="Skip files larger than this")
    include_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".go",
            ".rs",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".md",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
        ]
    )


class SynthConfig(BaseModel):
    """Configuration for synthetic data generation."""

    snapshot_dir: Path = Field(description="Path to ingested snapshot")
    output_path: Path = Field(default=Path("./data/training.jsonl"))
    min_diff_lines: int = Field(default=3, description="Minimum diff lines to include")
    max_diff_lines: int = Field(default=500, description="Maximum diff lines to include")
    max_context_files: int = Field(default=10, description="Max context files per example")
    include_tests: bool = Field(default=True)
    task_templates: list[str] = Field(
        default_factory=lambda: [
            "Fix the bug in {file}",
            "Implement {function} in {file}",
            "Refactor {function} to improve {aspect}",
            "Add error handling to {function}",
            "Write unit tests for {file}",
        ]
    )


class TrainConfig(BaseModel):
    """Configuration for QLoRA training."""

    model_name: str = Field(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    dataset_path: Path = Field(default=Path("./data/training.jsonl"))
    output_dir: Path = Field(default=Path("./artifacts"))

    # QLoRA parameters
    lora_r: int = Field(default=64)
    lora_alpha: int = Field(default=16)
    lora_dropout: float = Field(default=0.1)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training parameters
    num_epochs: int = Field(default=3)
    batch_size: int = Field(default=4)
    gradient_accumulation_steps: int = Field(default=4)
    learning_rate: float = Field(default=2e-4)
    max_seq_length: int = Field(default=4096)
    warmup_ratio: float = Field(default=0.03)
    weight_decay: float = Field(default=0.01)

    # Quantization
    load_in_4bit: bool = Field(default=True)
    bnb_4bit_compute_dtype: str = Field(default="bfloat16")
    bnb_4bit_quant_type: str = Field(default="nf4")

    # Misc
    seed: int = Field(default=42)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=100)
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="swarmcode")


class EvalConfig(BaseModel):
    """Configuration for evaluation."""

    adapter_path: Path = Field(description="Path to adapter weights")
    test_dataset: Path = Field(default=Path("./data/eval.jsonl"))
    base_model: str = Field(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    run_tests: bool = Field(default=True)
    test_command: str = Field(default="pytest")
    metrics: list[str] = Field(default_factory=lambda: ["pass_rate", "patch_apply_rate"])


class ServeConfig(BaseModel):
    """Configuration for serving the agent."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    adapter_path: Path | None = Field(default=None)
    base_model: str = Field(default="Qwen/Qwen2.5-Coder-7B-Instruct")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1)
    workspace_dir: Path = Field(default=Path("./workspace"))
    enable_tools: bool = Field(default=True)
    tool_timeout: int = Field(default=60)


class PublishConfig(BaseModel):
    """Configuration for publishing agent manifests."""

    adapter_path: Path = Field(description="Path to adapter weights")
    manifest_output: Path = Field(default=Path("./artifacts/manifest.json"))
    name: str = Field(default="swarmcode-agent")
    version: str = Field(default="0.1.0")
    description: str = Field(default="SwarmCode coding agent")
    author: str = Field(default="")
    private_key: str | None = Field(default=None, description="Hex private key for signing")
    ipfs_gateway: str = Field(default="http://localhost:5001")


class SwarmCodeConfig(BaseSettings):
    """Global SwarmCode configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SWARMCODE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("./data"))
    artifacts_dir: Path = Field(default=Path("./artifacts"))
    cache_dir: Path = Field(default=Path("~/.cache/swarmcode").expanduser())
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Sub-configs loaded from files
    ingest: IngestConfig | None = None
    synth: SynthConfig | None = None
    train: TrainConfig | None = None
    eval: EvalConfig | None = None
    serve: ServeConfig | None = None
    publish: PublishConfig | None = None


def load_config(config_path: Path | None = None) -> SwarmCodeConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        return SwarmCodeConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return SwarmCodeConfig(**data)


def load_subconfig(config_path: Path, config_class: type[BaseModel]) -> Any:
    """Load a specific sub-configuration from YAML."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return config_class(**data)
