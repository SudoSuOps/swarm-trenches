# SwarmCode

Local-first specialized coding agent training and runtime system.

Train custom coding agents on your codebase using QLoRA fine-tuning, then deploy them with an OpenAI-compatible API.

## Features

- **Ingest**: Clone repositories and create normalized snapshots with code maps
- **Synth**: Generate training data from git history and synthetic prompts
- **Train**: QLoRA fine-tuning with Qwen2.5-Coder models (7B/14B/32B)
- **Eval**: Evaluate model performance with patch application and test execution
- **Serve**: OpenAI-compatible API with tool execution (read_file, search, apply_patch, run_tests)
- **Publish**: Create signed agent manifests with IPFS upload support

## Quickstart

### Installation

```bash
# Core installation
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With code parsing (tree-sitter)
pip install -e ".[parser]"

# Everything
pip install -e ".[all]"
```

### Basic Workflow

```bash
# 1. Ingest a repository
swarmcode ingest https://github.com/your-org/your-repo.git

# 2. Generate training data
swarmcode synth ./data/snapshots/your-repo

# 3. Fine-tune a model
swarmcode train --dataset ./data/training.jsonl

# 4. Evaluate the model
swarmcode eval ./artifacts/swarmcode_adapter

# 5. Start the API server
swarmcode serve --adapter ./artifacts/swarmcode_adapter

# 6. Publish the agent
swarmcode publish ./artifacts/swarmcode_adapter --name "my-agent"
```

### Using Config Files

```bash
swarmcode ingest --config configs/ingest.yaml https://github.com/org/repo.git
swarmcode train --config configs/train.yaml
swarmcode serve --config configs/serve.yaml
```

## API Usage

Once the server is running, use any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="swarmcode",
    messages=[
        {"role": "user", "content": "Fix the bug in src/utils.py where the function returns None instead of an empty list"}
    ]
)
print(response.choices[0].message.content)
```

### Available Tools

The agent supports these tools (enabled by default):

- `read_file`: Read file contents with line numbers
- `search`: Search for patterns in files using regex
- `apply_patch`: Apply unified diff patches
- `run_tests`: Execute test commands
- `list_files`: List directory contents
- `write_file`: Create or update files

## Configuration

### Training Presets

| Preset | Model | VRAM | Batch Size |
|--------|-------|------|------------|
| `train.yaml` | Qwen2.5-Coder-7B | ~16GB | 4 |
| `train-14b.yaml` | Qwen2.5-Coder-14B | ~24GB | 2 |
| `train-32b.yaml` | Qwen2.5-Coder-32B | ~48GB | 1 |

### Environment Variables

```bash
SWARMCODE_DATA_DIR=./data
SWARMCODE_ARTIFACTS_DIR=./artifacts
SWARMCODE_LOG_LEVEL=INFO
```

## Project Structure

```
swarmcode/
├── swarmcode-cli/          # CLI entry point (Typer)
│   └── swarmcode_cli/
│       ├── main.py         # CLI commands
│       └── config.py       # Pydantic configs
├── swarmcode-agent/        # Runtime agent
│   └── swarmcode_agent/
│       ├── code_parser.py  # Tree-sitter/regex parsing
│       ├── ingest.py       # Repository ingestion
│       ├── synth.py        # Training data synthesis
│       ├── tools.py        # Tool implementations
│       └── server.py       # FastAPI server
├── swarmcode-trainer/      # Training pipeline
│   └── swarmcode_trainer/
│       ├── qlora.py        # QLoRA training
│       └── eval.py         # Evaluation
├── swarmcode-registry/     # Publishing
│   └── swarmcode_registry/
│       └── publish.py      # Manifest & IPFS
├── configs/                # Example configs
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── Makefile
└── pyproject.toml
```

## Development

```bash
# Install dev dependencies
make install-dev

# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run all checks
make check
```

## Training Data Format

The synthesized training data is in JSONL format:

```json
{
  "task": "Fix the null pointer exception in parse_config",
  "context_files": [
    {"path": "src/config.py", "content": "..."},
    {"path": "src/utils.py", "content": "..."}
  ],
  "diff": "--- a/src/config.py\n+++ b/src/config.py\n@@ -10,3 +10,5 @@\n...",
  "tests_command": "pytest tests/test_config.py",
  "metadata": {"source": "git_diff", "commit": "abc123"}
}
```

## Agent Manifest

Published agents include a signed manifest:

```json
{
  "name": "my-coding-agent",
  "version": "1.0.0",
  "description": "Custom agent trained on my codebase",
  "adapter": {
    "hash": "sha256:...",
    "size": 12345678
  },
  "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "capabilities": ["code_generation", "bug_fixing", "refactoring"],
  "tools": ["read_file", "search", "apply_patch", "run_tests"],
  "signature": "0x..."
}
```

## License

MIT
