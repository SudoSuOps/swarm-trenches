"""Repository ingestion and code map generation."""

import fnmatch
import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import git

from swarmcode_agent.code_parser import FileSymbols, get_parser

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for an ingested file."""

    path: str
    size: int
    hash: str
    language: str
    line_count: int
    last_modified: str | None = None


@dataclass
class CodeMap:
    """Lightweight code map of a repository."""

    repo_name: str
    snapshot_id: str
    created_at: str
    tree: dict
    files: list[FileMetadata]
    symbols: dict[str, FileSymbols]
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "repo_name": self.repo_name,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "tree": self.tree,
            "files": [
                {
                    "path": f.path,
                    "size": f.size,
                    "hash": f.hash,
                    "language": f.language,
                    "line_count": f.line_count,
                }
                for f in self.files
            ],
            "symbols": {
                path: {
                    "language": sym.language,
                    "functions": [
                        {
                            "name": fn.name,
                            "start_line": fn.start_line,
                            "end_line": fn.end_line,
                            "signature": fn.signature,
                        }
                        for fn in sym.functions
                    ],
                    "classes": [
                        {
                            "name": cls.name,
                            "start_line": cls.start_line,
                            "end_line": cls.end_line,
                            "bases": cls.bases,
                        }
                        for cls in sym.classes
                    ],
                    "imports": sym.imports,
                }
                for path, sym in self.symbols.items()
            },
            "total_files": self.total_files,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
        }


@dataclass
class IngestResult:
    """Result of repository ingestion."""

    snapshot_path: Path
    code_map_path: Path
    file_count: int
    function_count: int
    class_count: int


@dataclass
class IngestConfig:
    """Configuration for ingestion (imported from CLI)."""

    source: str
    output_dir: Path = field(default_factory=lambda: Path("./data/snapshots"))
    branch: str = "main"
    depth: int = 0
    exclude_patterns: list[str] = field(
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
    max_file_size_kb: int = 500
    include_extensions: list[str] = field(
        default_factory=lambda: [
            ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
            ".java", ".cpp", ".c", ".h", ".hpp", ".md", ".yaml",
            ".yml", ".json", ".toml",
        ]
    )


class RepositoryIngester:
    """Ingests repositories and creates normalized snapshots with code maps."""

    def __init__(self, config: IngestConfig):
        self.config = config
        self.parser = get_parser()

    def ingest(self) -> IngestResult:
        """Ingest the repository and create snapshot."""
        source = self.config.source
        is_remote = source.startswith(("http://", "https://", "git@"))

        if is_remote:
            repo_path = self._clone_repo(source)
            cleanup_temp = True
        else:
            repo_path = Path(source).resolve()
            if not repo_path.exists():
                raise ValueError(f"Local path does not exist: {repo_path}")
            cleanup_temp = False

        try:
            return self._process_repo(repo_path)
        finally:
            if cleanup_temp:
                shutil.rmtree(repo_path, ignore_errors=True)

    def _clone_repo(self, url: str) -> Path:
        """Clone a remote repository."""
        temp_dir = Path(tempfile.mkdtemp(prefix="swarmcode_"))

        logger.info(f"Cloning {url} to {temp_dir}")

        clone_kwargs = {}
        if self.config.depth > 0:
            clone_kwargs["depth"] = self.config.depth

        # Try with specified branch first, fall back to default branch
        try:
            clone_kwargs["branch"] = self.config.branch
            git.Repo.clone_from(url, temp_dir, **clone_kwargs)
        except git.GitCommandError as e:
            if "not found" in str(e).lower():
                # Branch not found, try without specifying branch (use default)
                logger.info(f"Branch '{self.config.branch}' not found, using default branch")
                shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir = Path(tempfile.mkdtemp(prefix="swarmcode_"))
                clone_kwargs.pop("branch", None)
                git.Repo.clone_from(url, temp_dir, **clone_kwargs)
            else:
                raise

        return temp_dir

    def _process_repo(self, repo_path: Path) -> IngestResult:
        """Process repository and generate snapshot."""
        repo_name = repo_path.name
        snapshot_id = hashlib.sha256(
            f"{repo_name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        snapshot_dir = self.config.output_dir / f"{repo_name}_{snapshot_id}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Copy files and build metadata
        files: list[FileMetadata] = []
        symbols: dict[str, FileSymbols] = {}
        tree = {"name": repo_name, "type": "directory", "children": []}

        self._process_directory(repo_path, snapshot_dir, files, symbols, tree)

        # Build code map
        total_functions = sum(len(s.functions) for s in symbols.values())
        total_classes = sum(len(s.classes) for s in symbols.values())

        code_map = CodeMap(
            repo_name=repo_name,
            snapshot_id=snapshot_id,
            created_at=datetime.now().isoformat(),
            tree=tree,
            files=files,
            symbols=symbols,
            total_files=len(files),
            total_functions=total_functions,
            total_classes=total_classes,
        )

        # Save code map
        code_map_path = snapshot_dir / "code_map.json"
        with open(code_map_path, "w") as f:
            json.dump(code_map.to_dict(), f, indent=2)

        # Save file listing
        with open(snapshot_dir / "files.txt", "w") as f:
            for file_meta in files:
                f.write(f"{file_meta.path}\n")

        return IngestResult(
            snapshot_path=snapshot_dir,
            code_map_path=code_map_path,
            file_count=len(files),
            function_count=total_functions,
            class_count=total_classes,
        )

    def _process_directory(
        self,
        source_dir: Path,
        dest_dir: Path,
        files: list[FileMetadata],
        symbols: dict[str, FileSymbols],
        tree: dict,
        rel_path: str = "",
    ) -> None:
        """Recursively process a directory."""
        try:
            entries = sorted(source_dir.iterdir())
        except PermissionError:
            logger.warning(f"Permission denied: {source_dir}")
            return

        for entry in entries:
            entry_rel = f"{rel_path}/{entry.name}" if rel_path else entry.name

            # Check exclusions
            if self._should_exclude(entry.name, entry_rel):
                continue

            if entry.is_dir():
                child_tree = {"name": entry.name, "type": "directory", "children": []}
                tree["children"].append(child_tree)

                child_dest = dest_dir / entry.name
                child_dest.mkdir(exist_ok=True)

                self._process_directory(
                    entry, child_dest, files, symbols, child_tree, entry_rel
                )

            elif entry.is_file():
                self._process_file(entry, dest_dir, files, symbols, tree, entry_rel)

    def _process_file(
        self,
        source_file: Path,
        dest_dir: Path,
        files: list[FileMetadata],
        symbols: dict[str, FileSymbols],
        tree: dict,
        rel_path: str,
    ) -> None:
        """Process a single file."""
        # Check extension
        if source_file.suffix.lower() not in self.config.include_extensions:
            return

        # Check size
        try:
            size = source_file.stat().st_size
            if size > self.config.max_file_size_kb * 1024:
                logger.debug(f"Skipping large file: {rel_path} ({size} bytes)")
                return
        except OSError:
            return

        # Read and hash content
        try:
            content = source_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to read {rel_path}: {e}")
            return

        file_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        line_count = content.count("\n") + 1

        # Copy file
        dest_file = dest_dir / source_file.name
        dest_file.write_text(content)

        # Detect language
        language = self.parser.parse(content, source_file).language

        # Create metadata
        file_meta = FileMetadata(
            path=rel_path,
            size=size,
            hash=file_hash,
            language=language,
            line_count=line_count,
        )
        files.append(file_meta)

        # Add to tree
        tree["children"].append({
            "name": source_file.name,
            "type": "file",
            "size": size,
            "language": language,
        })

        # Parse symbols
        file_symbols = self.parser.parse(content, source_file)
        if file_symbols.functions or file_symbols.classes:
            symbols[rel_path] = file_symbols

    def _should_exclude(self, name: str, rel_path: str) -> bool:
        """Check if a path should be excluded."""
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False
