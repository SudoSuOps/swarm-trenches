"""Evaluation module for trained models."""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    adapter_path: Path
    test_dataset: Path = field(default_factory=lambda: Path("./data/eval.jsonl"))
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    run_tests: bool = True
    test_command: str = "pytest"
    metrics: list[str] = field(default_factory=lambda: ["pass_rate", "patch_apply_rate"])


@dataclass
class TestResults:
    """Results from running tests."""

    passed: int
    failed: int
    total: int
    output: str


@dataclass
class EvalResult:
    """Evaluation results."""

    pass_rate: float
    patch_apply_rate: float
    test_results: TestResults | None
    examples_evaluated: int
    detailed_results: list[dict]


class Evaluator:
    """Evaluates fine-tuned models on test examples."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def evaluate(self) -> EvalResult:
        """Run evaluation on the test dataset."""
        # Load test examples
        examples = self._load_examples()
        logger.info(f"Loaded {len(examples)} evaluation examples")

        if not examples:
            return EvalResult(
                pass_rate=0.0,
                patch_apply_rate=0.0,
                test_results=None,
                examples_evaluated=0,
                detailed_results=[],
            )

        # Initialize model
        self._load_model()

        # Evaluate each example
        detailed_results = []
        patches_applied = 0
        tests_passed = 0

        for i, example in enumerate(examples):
            logger.info(f"Evaluating example {i + 1}/{len(examples)}")
            result = self._evaluate_example(example)
            detailed_results.append(result)

            if result["patch_applied"]:
                patches_applied += 1
            if result.get("tests_passed"):
                tests_passed += 1

        # Calculate metrics
        pass_rate = tests_passed / len(examples) if examples else 0.0
        patch_apply_rate = patches_applied / len(examples) if examples else 0.0

        # Run overall tests if configured
        test_results = None
        if self.config.run_tests:
            test_results = self._run_tests()

        return EvalResult(
            pass_rate=pass_rate,
            patch_apply_rate=patch_apply_rate,
            test_results=test_results,
            examples_evaluated=len(examples),
            detailed_results=detailed_results,
        )

    def _load_examples(self) -> list[dict]:
        """Load evaluation examples."""
        examples = []

        if not self.config.test_dataset.exists():
            logger.warning(f"Test dataset not found: {self.config.test_dataset}")
            return examples

        with open(self.config.test_dataset) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return examples

    def _load_model(self):
        """Load the model with adapter."""
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {self.config.base_model}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            self._model = PeftModel.from_pretrained(
                base_model,
                self.config.adapter_path,
            )
            self._model.eval()

            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning("transformers not available, using mock evaluation")
            self._model = "mock"
            self._tokenizer = "mock"

    def _evaluate_example(self, example: dict) -> dict:
        """Evaluate a single example."""
        task = example.get("task", "")
        context_files = example.get("context_files", [])
        expected_diff = example.get("diff", "")
        tests_command = example.get("tests_command")

        # Generate prediction
        predicted_diff = self._generate_patch(task, context_files)

        # Try to apply the patch
        patch_applied = self._try_apply_patch(predicted_diff, context_files)

        # Run tests if command provided
        tests_passed = False
        if patch_applied and tests_command and self.config.run_tests:
            tests_passed = self._run_example_tests(tests_command)

        return {
            "task": task,
            "expected_diff": expected_diff[:500],
            "predicted_diff": predicted_diff[:500],
            "patch_applied": patch_applied,
            "tests_passed": tests_passed,
        }

    def _generate_patch(self, task: str, context_files: list[dict]) -> str:
        """Generate a patch using the model."""
        if self._model == "mock":
            return "--- mock patch ---"

        try:
            import torch

            # Build context
            context_parts = []
            for cf in context_files[:5]:
                path = cf.get("path", "unknown")
                content = cf.get("content", "")[:5000]
                context_parts.append(f"### {path}\n```\n{content}\n```")

            context = "\n\n".join(context_parts)

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
            ]

            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = f"Task: {task}\nContext: {context}\nDiff:"

            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Extract diff from response
            return self._extract_diff(response)

        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""

    def _extract_diff(self, response: str) -> str:
        """Extract diff content from model response."""
        # Look for code blocks with diff
        import re

        patterns = [
            r"```diff\n(.*?)```",
            r"```\n(---.*?)```",
            r"(---\s+a/.*?\n\+\+\+.*?(?:\n(?!---)[^\n]*)*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Return raw response if no diff found
        return response.strip()

    def _try_apply_patch(self, diff: str, context_files: list[dict]) -> bool:
        """Try to apply a patch to check if it's valid."""
        if not diff or not context_files:
            return False

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Write context files
                for cf in context_files:
                    path = cf.get("path", "")
                    content = cf.get("content", "")
                    if path:
                        file_path = tmpdir / path
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(content)

                # Try to apply patch
                result = subprocess.run(
                    ["patch", "-p1", "--dry-run"],
                    input=diff.encode(),
                    cwd=tmpdir,
                    capture_output=True,
                    timeout=10,
                )

                return result.returncode == 0

        except FileNotFoundError:
            # patch command not available
            logger.debug("patch command not available")
            return self._validate_diff_syntax(diff)
        except Exception as e:
            logger.debug(f"Patch application failed: {e}")
            return False

    def _validate_diff_syntax(self, diff: str) -> bool:
        """Basic validation of diff syntax."""
        lines = diff.split("\n")

        has_header = False
        has_changes = False

        for line in lines:
            if line.startswith("---") or line.startswith("+++"):
                has_header = True
            if line.startswith("+") or line.startswith("-"):
                if not line.startswith("---") and not line.startswith("+++"):
                    has_changes = True

        return has_header and has_changes

    def _run_example_tests(self, command: str) -> bool:
        """Run tests for a specific example."""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                timeout=60,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_tests(self) -> TestResults:
        """Run the overall test suite."""
        try:
            result = subprocess.run(
                self.config.test_command.split(),
                capture_output=True,
                timeout=300,
            )

            output = result.stdout.decode("utf-8", errors="replace")
            output += result.stderr.decode("utf-8", errors="replace")

            # Parse test results (pytest format)
            passed = 0
            failed = 0
            total = 0

            import re

            # Look for pytest summary
            match = re.search(r"(\d+) passed", output)
            if match:
                passed = int(match.group(1))

            match = re.search(r"(\d+) failed", output)
            if match:
                failed = int(match.group(1))

            total = passed + failed

            return TestResults(
                passed=passed,
                failed=failed,
                total=total,
                output=output[:5000],
            )

        except Exception as e:
            logger.warning(f"Failed to run tests: {e}")
            return TestResults(
                passed=0,
                failed=0,
                total=0,
                output=str(e),
            )
