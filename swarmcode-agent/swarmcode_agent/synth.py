"""Training data synthesis from git diffs and synthetic prompts."""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import git

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """A single training example."""

    task: str
    context_files: list[dict]
    diff: str
    tests_command: str | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "task": self.task,
            "context_files": self.context_files,
            "diff": self.diff,
            "tests_command": self.tests_command,
            "metadata": self.metadata,
        }


@dataclass
class SynthConfig:
    """Configuration for synthesis."""

    snapshot_dir: Path
    output_path: Path = field(default_factory=lambda: Path("./data/training.jsonl"))
    min_diff_lines: int = 3
    max_diff_lines: int = 500
    max_context_files: int = 10
    include_tests: bool = True
    task_templates: list[str] = field(
        default_factory=lambda: [
            "Fix the bug in {file}",
            "Implement {function} in {file}",
            "Refactor {function} to improve readability",
            "Add error handling to {function}",
            "Write unit tests for {file}",
        ]
    )


@dataclass
class SynthResult:
    """Result of data synthesis."""

    output_path: Path
    example_count: int
    diff_examples: int
    synthetic_examples: int


class DataSynthesizer:
    """Generates training examples from repositories."""

    def __init__(self, config: SynthConfig):
        self.config = config
        self._code_map: dict | None = None

    def synthesize(self) -> SynthResult:
        """Generate training examples."""
        self._load_code_map()

        examples: list[TrainingExample] = []

        # Generate from git diffs if available
        diff_examples = self._extract_from_diffs()
        examples.extend(diff_examples)
        logger.info(f"Generated {len(diff_examples)} examples from git diffs")

        # Generate synthetic examples
        synthetic_examples = self._generate_synthetic()
        examples.extend(synthetic_examples)
        logger.info(f"Generated {len(synthetic_examples)} synthetic examples")

        # Write output
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        return SynthResult(
            output_path=self.config.output_path,
            example_count=len(examples),
            diff_examples=len(diff_examples),
            synthetic_examples=len(synthetic_examples),
        )

    def _load_code_map(self) -> None:
        """Load the code map from snapshot."""
        code_map_path = self.config.snapshot_dir / "code_map.json"
        if not code_map_path.exists():
            raise ValueError(f"Code map not found: {code_map_path}")

        with open(code_map_path) as f:
            self._code_map = json.load(f)

    def _extract_from_diffs(self) -> list[TrainingExample]:
        """Extract training examples from git history."""
        examples = []

        # Try to find .git in original repo reference
        git_dir = self.config.snapshot_dir.parent
        repo = None

        # Walk up to find .git
        for _ in range(5):
            if (git_dir / ".git").exists():
                try:
                    repo = git.Repo(git_dir)
                    break
                except git.InvalidGitRepositoryError:
                    pass
            git_dir = git_dir.parent

        if repo is None:
            logger.warning("No git repository found, skipping diff extraction")
            return examples

        try:
            commits = list(repo.iter_commits(max_count=100))
        except Exception as e:
            logger.warning(f"Failed to iterate commits: {e}")
            return examples

        for commit in commits:
            if not commit.parents:
                continue

            try:
                diff = commit.parents[0].diff(commit, create_patch=True)
            except Exception:
                continue

            for d in diff:
                if not d.a_path or not d.diff:
                    continue

                # Get diff content
                try:
                    diff_text = d.diff.decode("utf-8", errors="replace")
                except Exception:
                    continue

                # Filter by size
                diff_lines = len(diff_text.split("\n"))
                if diff_lines < self.config.min_diff_lines:
                    continue
                if diff_lines > self.config.max_diff_lines:
                    continue

                # Create task description from commit message
                task = self._create_task_from_commit(commit.message, d.a_path)

                # Get context files
                context_files = self._get_context_files(d.a_path)

                # Detect test command
                tests_command = self._detect_test_command(d.a_path) if self.config.include_tests else None

                example = TrainingExample(
                    task=task,
                    context_files=context_files,
                    diff=diff_text,
                    tests_command=tests_command,
                    metadata={
                        "source": "git_diff",
                        "commit": str(commit.hexsha)[:8],
                        "file": d.a_path,
                    },
                )
                examples.append(example)

        return examples

    def _generate_synthetic(self) -> list[TrainingExample]:
        """Generate synthetic training examples."""
        examples = []

        if not self._code_map:
            return examples

        symbols = self._code_map.get("symbols", {})

        for file_path, file_symbols in symbols.items():
            functions = file_symbols.get("functions", [])
            classes = file_symbols.get("classes", [])

            # Generate function-level tasks
            for func in functions:
                for template in random.sample(
                    self.config.task_templates,
                    min(2, len(self.config.task_templates))
                ):
                    task = template.format(
                        file=file_path,
                        function=func["name"],
                        aspect="performance",
                    )

                    context_files = self._get_context_files(file_path)

                    # Create synthetic diff placeholder
                    diff = self._create_synthetic_diff(file_path, func)

                    tests_command = self._detect_test_command(file_path)

                    example = TrainingExample(
                        task=task,
                        context_files=context_files,
                        diff=diff,
                        tests_command=tests_command,
                        metadata={
                            "source": "synthetic",
                            "file": file_path,
                            "function": func["name"],
                            "template": template,
                        },
                    )
                    examples.append(example)

            # Generate class-level tasks
            for cls in classes:
                task = f"Add a new method to class {cls['name']} in {file_path}"
                context_files = self._get_context_files(file_path)
                diff = self._create_synthetic_class_diff(file_path, cls)

                example = TrainingExample(
                    task=task,
                    context_files=context_files,
                    diff=diff,
                    tests_command=self._detect_test_command(file_path),
                    metadata={
                        "source": "synthetic",
                        "file": file_path,
                        "class": cls["name"],
                    },
                )
                examples.append(example)

        return examples

    def _create_task_from_commit(self, message: str, file_path: str) -> str:
        """Create a task description from a commit message."""
        # Clean up commit message
        first_line = message.split("\n")[0].strip()

        # Remove common prefixes
        prefixes = ["fix:", "feat:", "refactor:", "chore:", "docs:", "test:"]
        for prefix in prefixes:
            if first_line.lower().startswith(prefix):
                first_line = first_line[len(prefix):].strip()
                break

        if len(first_line) < 10:
            return f"Make changes to {file_path}"

        return first_line

    def _get_context_files(self, target_file: str) -> list[dict]:
        """Get relevant context files for a target file."""
        context = []

        if not self._code_map:
            return context

        files = self._code_map.get("files", [])

        # Find files in same directory
        target_dir = str(Path(target_file).parent)
        related = [f for f in files if f["path"].startswith(target_dir)]

        # Add target file first if exists
        target_data = next((f for f in files if f["path"] == target_file), None)
        if target_data:
            file_content = self._read_file(target_file)
            if file_content:
                context.append({
                    "path": target_file,
                    "content": file_content,
                })

        # Add related files
        for f in related[:self.config.max_context_files - 1]:
            if f["path"] == target_file:
                continue
            file_content = self._read_file(f["path"])
            if file_content:
                context.append({
                    "path": f["path"],
                    "content": file_content,
                })

        return context

    def _read_file(self, rel_path: str) -> str | None:
        """Read a file from the snapshot."""
        # Reconstruct path in snapshot
        parts = rel_path.split("/")
        file_path = self.config.snapshot_dir

        for part in parts:
            file_path = file_path / part

        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def _detect_test_command(self, file_path: str) -> str | None:
        """Detect the appropriate test command for a file."""
        ext = Path(file_path).suffix.lower()

        test_commands = {
            ".py": "pytest",
            ".js": "npm test",
            ".ts": "npm test",
            ".tsx": "npm test",
            ".jsx": "npm test",
            ".go": "go test ./...",
            ".rs": "cargo test",
            ".java": "mvn test",
        }

        return test_commands.get(ext)

    def _create_synthetic_diff(self, file_path: str, func: dict) -> str:
        """Create a synthetic diff for a function."""
        return f"""--- a/{file_path}
+++ b/{file_path}
@@ -{func['start_line']},10 +{func['start_line']},15 @@
 # Function: {func['name']}
 # Signature: {func.get('signature', 'N/A')}
+# [SYNTHETIC PLACEHOLDER - Replace with actual implementation]
+# This diff represents changes to be made to the function.
"""

    def _create_synthetic_class_diff(self, file_path: str, cls: dict) -> str:
        """Create a synthetic diff for a class."""
        return f"""--- a/{file_path}
+++ b/{file_path}
@@ -{cls['start_line']},10 +{cls['start_line']},20 @@
 # Class: {cls['name']}
 # Bases: {', '.join(cls.get('bases', []))}
+# [SYNTHETIC PLACEHOLDER - Replace with actual method implementation]
+    def new_method(self):
+        pass
"""
