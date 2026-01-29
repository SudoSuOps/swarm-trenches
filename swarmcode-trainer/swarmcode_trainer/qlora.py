"""QLoRA fine-tuning implementation."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    dataset_path: Path = field(default_factory=lambda: Path("./data/training.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("./artifacts"))

    # QLoRA parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "swarmcode"


@dataclass
class TrainResult:
    """Training result."""

    adapter_path: Path
    final_loss: float
    steps: int
    epochs: int


class QLoRATrainer:
    """QLoRA fine-tuning trainer for coding models."""

    def __init__(self, config: TrainConfig):
        self.config = config

    def train(self) -> TrainResult:
        """Run QLoRA fine-tuning."""
        try:
            return self._train_with_transformers()
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.info("Install with: pip install swarmcode[train]")
            raise

    def _train_with_transformers(self) -> TrainResult:
        """Train using transformers + peft + accelerate."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer

        logger.info(f"Loading tokenizer: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure quantization
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"Loading model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)

        # Log trainable parameters
        trainable_params, total_params = model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        # Load dataset
        dataset = self._load_dataset(tokenizer)

        # Configure training
        output_dir = self.config.output_dir / "swarmcode_adapter"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            bf16=compute_dtype == torch.bfloat16,
            fp16=compute_dtype == torch.float16,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            group_by_length=True,
            report_to="wandb" if self.config.use_wandb else "none",
            run_name="swarmcode_train" if self.config.use_wandb else None,
            seed=self.config.seed,
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save adapter
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # Save training config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_name": self.config.model_name,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "final_loss": train_result.training_loss,
            }, f, indent=2)

        logger.info(f"Training complete! Adapter saved to: {output_dir}")

        return TrainResult(
            adapter_path=output_dir,
            final_loss=train_result.training_loss,
            steps=trainer.state.global_step,
            epochs=self.config.num_epochs,
        )

    def _load_dataset(self, tokenizer) -> "Dataset":
        """Load and format training dataset."""
        from datasets import Dataset

        examples = []

        with open(self.config.dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(examples)} training examples")

        # Format as chat conversations
        formatted = []
        for ex in examples:
            text = self._format_example(ex, tokenizer)
            formatted.append({"text": text})

        return Dataset.from_list(formatted)

    def _format_example(self, example: dict, tokenizer) -> str:
        """Format a single example as a chat conversation."""
        task = example.get("task", "")
        context_files = example.get("context_files", [])
        diff = example.get("diff", "")

        # Build context
        context_parts = []
        for cf in context_files[:5]:  # Limit context files
            path = cf.get("path", "unknown")
            content = cf.get("content", "")
            # Truncate long files
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated)"
            context_parts.append(f"### {path}\n```\n{content}\n```")

        context = "\n\n".join(context_parts)

        # Format as messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert coding assistant. Given a task and context files, "
                    "produce a unified diff patch to accomplish the task."
                ),
            },
            {
                "role": "user",
                "content": f"## Task\n{task}\n\n## Context\n{context}",
            },
            {
                "role": "assistant",
                "content": f"```diff\n{diff}\n```",
            },
        ]

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Fallback format
            parts = []
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                parts.append(f"<|{role}|>\n{content}")
            return "\n".join(parts) + tokenizer.eos_token


# Preset configurations for different model sizes
MODEL_PRESETS = {
    "qwen-7b": TrainConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=4096,
    ),
    "qwen-14b": TrainConfig(
        model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
        batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=4096,
    ),
    "qwen-32b": TrainConfig(
        model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
        batch_size=1,
        gradient_accumulation_steps=16,
        max_seq_length=2048,
    ),
}


def get_preset(name: str) -> TrainConfig:
    """Get a preset training configuration."""
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[name]
