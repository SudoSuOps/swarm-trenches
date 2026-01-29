"""SwarmCode CLI - Main entry point."""

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from swarmcode_cli.config import (
    EvalConfig,
    IngestConfig,
    PublishConfig,
    ServeConfig,
    SynthConfig,
    TrainConfig,
    load_subconfig,
)

app = typer.Typer(
    name="swarmcode",
    help="Local-first specialized coding agent training and runtime system.",
    no_args_is_help=True,
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """SwarmCode - Train and run specialized coding agents."""
    setup_logging(verbose)


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="Git URL or local path to ingest")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for snapshot")
    ] = Path("./data/snapshots"),
    branch: Annotated[str, typer.Option("--branch", "-b", help="Git branch to clone")] = "main",
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
) -> None:
    """Ingest a repository and create a normalized snapshot with code map."""
    from swarmcode_agent.ingest import RepositoryIngester

    if config:
        cfg = load_subconfig(config, IngestConfig)
    else:
        cfg = IngestConfig(source=source, output_dir=output, branch=branch)

    console.print(f"[bold blue]Ingesting repository:[/] {cfg.source}")

    ingester = RepositoryIngester(cfg)
    result = ingester.ingest()

    console.print(f"[bold green]✓[/] Snapshot created at: {result.snapshot_path}")
    console.print(f"  Files: {result.file_count}")
    console.print(f"  Functions: {result.function_count}")
    console.print(f"  Code map: {result.code_map_path}")


@app.command()
def synth(
    snapshot: Annotated[Path, typer.Argument(help="Path to ingested snapshot directory")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output JSONL path")
    ] = Path("./data/training.jsonl"),
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
) -> None:
    """Generate training examples from git diffs and synthetic prompts."""
    from swarmcode_agent.synth import DataSynthesizer

    if config:
        cfg = load_subconfig(config, SynthConfig)
    else:
        cfg = SynthConfig(snapshot_dir=snapshot, output_path=output)

    console.print(f"[bold blue]Synthesizing training data from:[/] {cfg.snapshot_dir}")

    synthesizer = DataSynthesizer(cfg)
    result = synthesizer.synthesize()

    console.print(f"[bold green]✓[/] Generated {result.example_count} training examples")
    console.print(f"  Output: {result.output_path}")
    console.print(f"  From diffs: {result.diff_examples}")
    console.print(f"  Synthetic: {result.synthetic_examples}")


@app.command()
def train(
    dataset: Annotated[
        Path, typer.Option("--dataset", "-d", help="Path to training JSONL")
    ] = Path("./data/training.jsonl"),
    model: Annotated[
        str, typer.Option("--model", "-m", help="Base model name or path")
    ] = "Qwen/Qwen2.5-Coder-7B-Instruct",
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for adapter")
    ] = Path("./artifacts"),
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of epochs")] = 3,
) -> None:
    """Fine-tune a model using QLoRA on the training dataset."""
    from swarmcode_trainer.qlora import QLoRATrainer

    if config:
        cfg = load_subconfig(config, TrainConfig)
    else:
        cfg = TrainConfig(
            dataset_path=dataset,
            model_name=model,
            output_dir=output,
            num_epochs=epochs,
        )

    console.print(f"[bold blue]Starting QLoRA training:[/]")
    console.print(f"  Model: {cfg.model_name}")
    console.print(f"  Dataset: {cfg.dataset_path}")
    console.print(f"  Epochs: {cfg.num_epochs}")

    trainer = QLoRATrainer(cfg)
    result = trainer.train()

    console.print(f"[bold green]✓[/] Training complete!")
    console.print(f"  Adapter saved to: {result.adapter_path}")
    console.print(f"  Final loss: {result.final_loss:.4f}")


@app.command("eval")
def evaluate(
    adapter: Annotated[Path, typer.Argument(help="Path to adapter weights")],
    dataset: Annotated[
        Path, typer.Option("--dataset", "-d", help="Evaluation dataset path")
    ] = Path("./data/eval.jsonl"),
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
    run_tests: Annotated[bool, typer.Option("--run-tests/--no-tests", help="Run unit tests")] = True,
) -> None:
    """Evaluate the fine-tuned model on test examples."""
    from swarmcode_trainer.eval import Evaluator

    if config:
        cfg = load_subconfig(config, EvalConfig)
    else:
        cfg = EvalConfig(adapter_path=adapter, test_dataset=dataset, run_tests=run_tests)

    console.print(f"[bold blue]Evaluating model:[/]")
    console.print(f"  Adapter: {cfg.adapter_path}")
    console.print(f"  Dataset: {cfg.test_dataset}")

    evaluator = Evaluator(cfg)
    result = evaluator.evaluate()

    console.print(f"[bold green]✓[/] Evaluation complete!")
    console.print(f"  Pass rate: {result.pass_rate:.1%}")
    console.print(f"  Patch apply rate: {result.patch_apply_rate:.1%}")
    if result.test_results:
        console.print(f"  Tests passed: {result.test_results.passed}/{result.test_results.total}")


@app.command()
def serve(
    adapter: Annotated[
        Optional[Path], typer.Option("--adapter", "-a", help="Path to adapter weights")
    ] = None,
    host: Annotated[str, typer.Option("--host", "-h", help="Host to bind")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind")] = 8000,
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
) -> None:
    """Start the OpenAI-compatible API server."""
    import uvicorn

    from swarmcode_agent.server import create_app

    if config:
        cfg = load_subconfig(config, ServeConfig)
    else:
        cfg = ServeConfig(adapter_path=adapter, host=host, port=port)

    console.print(f"[bold blue]Starting SwarmCode server:[/]")
    console.print(f"  Host: {cfg.host}:{cfg.port}")
    console.print(f"  Adapter: {cfg.adapter_path or 'None (base model)'}")
    console.print(f"  Tools enabled: {cfg.enable_tools}")

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port)


@app.command()
def publish(
    adapter: Annotated[Path, typer.Argument(help="Path to adapter weights")],
    name: Annotated[str, typer.Option("--name", "-n", help="Agent name")] = "swarmcode-agent",
    version: Annotated[str, typer.Option("--version", help="Agent version")] = "0.1.0",
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config YAML")
    ] = None,
) -> None:
    """Create a signed agent manifest and publish to IPFS."""
    from swarmcode_registry.publish import Publisher

    if config:
        cfg = load_subconfig(config, PublishConfig)
    else:
        cfg = PublishConfig(adapter_path=adapter, name=name, version=version)

    console.print(f"[bold blue]Publishing agent:[/]")
    console.print(f"  Name: {cfg.name}")
    console.print(f"  Version: {cfg.version}")

    publisher = Publisher(cfg)
    result = publisher.publish()

    console.print(f"[bold green]✓[/] Agent published!")
    console.print(f"  Manifest: {result.manifest_path}")
    console.print(f"  Signature: {result.signature[:16]}...")
    if result.ipfs_cid:
        console.print(f"  IPFS CID: {result.ipfs_cid}")


if __name__ == "__main__":
    app()
