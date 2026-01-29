# SwarmCode Architecture

## Overview

SwarmCode is a local-first system for training and deploying specialized coding agents. It follows a modular architecture with four main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (swarmcode)                          │
├─────────────────────────────────────────────────────────────────┤
│  ingest  │  synth  │  train  │  eval  │  serve  │  publish      │
└────┬─────┴────┬────┴────┬────┴────┬───┴────┬────┴────┬─────────┘
     │          │         │         │        │         │
     ▼          ▼         ▼         ▼        ▼         ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐
│  Agent  │ │  Agent  │ │ Trainer │ │ Trainer │ │    Registry     │
│ Ingest  │ │  Synth  │ │  QLoRA  │ │  Eval   │ │    Publish      │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘
     │          │         │         │        │         │
     ▼          ▼         ▼         ▼        ▼         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│  snapshots/  │  training.jsonl  │  artifacts/  │  manifests/    │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. CLI (`swarmcode-cli`)

The entry point for all operations. Built with Typer for a modern CLI experience.

- **Config Management**: Pydantic-based configuration with YAML support
- **Command Routing**: Delegates to appropriate modules
- **Progress Reporting**: Rich terminal output

### 2. Agent (`swarmcode-agent`)

Runtime components for ingestion, synthesis, and serving.

#### Code Parser

Extracts symbols from source code:

```python
# With tree-sitter (accurate)
from swarmcode_agent.code_parser import TreeSitterParser

# Fallback regex (no dependencies)
from swarmcode_agent.code_parser import RegexParser
```

Supported languages:
- Python
- JavaScript/TypeScript
- Go
- Rust

#### Repository Ingester

Creates normalized snapshots:

```
input: git repo
output:
  - snapshot/           # Copied source files
  - code_map.json       # File tree + symbols
  - files.txt           # File listing
```

#### Data Synthesizer

Generates training data:

```json
{
  "task": "Fix the bug in parse_config",
  "context_files": [{"path": "...", "content": "..."}],
  "diff": "--- a/file.py\n+++ b/file.py\n...",
  "tests_command": "pytest tests/",
  "metadata": {"source": "git_diff"}
}
```

#### Tools

OpenAI-compatible tool implementations:

| Tool | Description |
|------|-------------|
| `read_file` | Read file with line numbers |
| `search` | Regex search in files |
| `apply_patch` | Apply unified diffs |
| `run_tests` | Execute test commands |
| `list_files` | Directory listing |
| `write_file` | Create/update files |

#### Server

FastAPI server with OpenAI-compatible endpoints:

- `POST /v1/chat/completions` - Chat completions
- `GET /v1/models` - List models
- `GET /v1/tools` - List available tools
- `POST /v1/tools/execute` - Direct tool execution

### 3. Trainer (`swarmcode-trainer`)

Training pipeline using QLoRA.

#### QLoRA Configuration

```python
lora_r: int = 64           # Rank
lora_alpha: int = 16       # Alpha scaling
lora_dropout: float = 0.1  # Dropout
load_in_4bit: bool = True  # 4-bit quantization
```

#### Training Flow

```
Load base model (4-bit quantized)
         │
         ▼
Add LoRA adapters to attention layers
         │
         ▼
Fine-tune on task→diff pairs
         │
         ▼
Save adapter weights (~100MB)
```

#### Evaluation

Metrics computed:
- **Pass rate**: Tests passing after patch
- **Patch apply rate**: Valid unified diffs

### 4. Registry (`swarmcode-registry`)

Publishing and distribution.

#### Agent Manifest

```json
{
  "name": "my-agent",
  "version": "1.0.0",
  "adapter": {
    "hash": "sha256:...",
    "size": 12345678
  },
  "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "capabilities": ["code_generation", "bug_fixing"],
  "tools": ["read_file", "search", "apply_patch"],
  "signature": "0x..."
}
```

#### Signing (EIP-191)

```python
# Create signable message
message = json.dumps(manifest, sort_keys=True)

# Sign with private key
signature = account.sign_message(message)
```

## Data Flow

### Training Pipeline

```
1. INGEST
   git repo → snapshot + code_map.json

2. SYNTH
   snapshot + git history → training.jsonl

3. TRAIN
   training.jsonl + base_model → adapter_weights

4. EVAL
   adapter_weights + eval.jsonl → metrics
```

### Inference Pipeline

```
1. SERVE
   adapter_weights → API server

2. REQUEST
   user message → model + tools → response

3. TOOL CALLS
   tool request → execute → result → model → response
```

## Security Considerations

### Tool Execution

- Workspace isolation: Tools operate within configured directory
- Command allowlist: Only approved test commands
- Timeout enforcement: Prevents runaway processes
- Path validation: Prevents directory traversal

### Signing

- EIP-191 signatures for manifest integrity
- Private key never stored in manifests
- Verification before loading external adapters

## Extension Points

### Custom Tools

```python
from swarmcode_agent.tools import AgentTools

class CustomTools(AgentTools):
    async def my_tool(self, arg: str) -> ToolResult:
        # Implementation
        return ToolResult(success=True, output="...")
```

### Custom Training

```python
from swarmcode_trainer.qlora import QLoRATrainer, TrainConfig

config = TrainConfig(
    model_name="your-model",
    lora_target_modules=["custom_module"],
)
trainer = QLoRATrainer(config)
```
