"""Tool implementations for the coding agent."""

import asyncio
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    error: str | None = None


class AgentTools:
    """Tool implementations for the SwarmCode agent."""

    def __init__(self, workspace_dir: Path, timeout: int = 60):
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool_map = {
            "read_file": self.read_file,
            "search": self.search,
            "apply_patch": self.apply_patch,
            "run_tests": self.run_tests,
            "list_files": self.list_files,
            "write_file": self.write_file,
        }

        tool_func = tool_map.get(tool_name)
        if not tool_func:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        try:
            return await tool_func(**arguments)
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed")
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    async def read_file(self, path: str, start_line: int = 0, end_line: int = -1) -> ToolResult:
        """Read contents of a file."""
        file_path = self._resolve_path(path)

        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        if not file_path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a file: {path}",
            )

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")

            if end_line == -1:
                end_line = len(lines)

            selected_lines = lines[start_line:end_line]

            # Add line numbers
            numbered_content = "\n".join(
                f"{i + start_line + 1:4d} | {line}"
                for i, line in enumerate(selected_lines)
            )

            return ToolResult(
                success=True,
                output=numbered_content,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}",
            )

    async def search(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        max_results: int = 50,
    ) -> ToolResult:
        """Search for a pattern in files."""
        search_path = self._resolve_path(path)

        if not search_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {path}",
            )

        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid regex pattern: {e}",
            )

        try:
            for file_path in search_path.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                if len(results) >= max_results:
                    break

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.split("\n")):
                        if regex.search(line):
                            rel_path = file_path.relative_to(self.workspace_dir)
                            results.append(f"{rel_path}:{i + 1}: {line.strip()}")
                            if len(results) >= max_results:
                                break
                except Exception:
                    continue

            output = "\n".join(results)
            if len(results) >= max_results:
                output += f"\n\n(truncated at {max_results} results)"

            return ToolResult(
                success=True,
                output=output or "No matches found",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {e}",
            )

    async def apply_patch(self, patch: str, path: str | None = None) -> ToolResult:
        """Apply a unified diff patch to files."""
        if path:
            # Apply to specific file
            file_path = self._resolve_path(path)
            return await self._apply_patch_to_file(patch, file_path)
        else:
            # Apply as unified diff
            return await self._apply_unified_patch(patch)

    async def _apply_patch_to_file(self, patch: str, file_path: Path) -> ToolResult:
        """Apply a patch to a specific file."""
        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path}",
            )

        # Write patch to temp file
        patch_file = self.workspace_dir / ".tmp_patch"
        patch_file.write_text(patch)

        try:
            proc = await asyncio.create_subprocess_exec(
                "patch",
                "-p0",
                str(file_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(patch.encode()),
                timeout=self.timeout,
            )

            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    output=stdout.decode("utf-8", errors="replace"),
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout.decode("utf-8", errors="replace"),
                    error=stderr.decode("utf-8", errors="replace"),
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error="Patch operation timed out",
            )
        except FileNotFoundError:
            # patch command not available, try manual application
            return await self._apply_patch_manual(patch, file_path)
        finally:
            patch_file.unlink(missing_ok=True)

    async def _apply_unified_patch(self, patch: str) -> ToolResult:
        """Apply a unified diff patch."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "patch",
                "-p1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(patch.encode()),
                timeout=self.timeout,
            )

            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    output=stdout.decode("utf-8", errors="replace"),
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout.decode("utf-8", errors="replace"),
                    error=stderr.decode("utf-8", errors="replace"),
                )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error="Patch operation timed out",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="patch command not available",
            )

    async def _apply_patch_manual(self, patch: str, file_path: Path) -> ToolResult:
        """Manually apply a simple patch (fallback when patch command unavailable)."""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Parse patch for additions/deletions
            changes_made = 0
            patch_lines = patch.split("\n")

            for patch_line in patch_lines:
                if patch_line.startswith("+") and not patch_line.startswith("+++"):
                    # This is a simplified implementation
                    changes_made += 1

            if changes_made == 0:
                return ToolResult(
                    success=False,
                    output="",
                    error="Could not parse patch format",
                )

            return ToolResult(
                success=True,
                output=f"Manual patch application not fully implemented. Detected {changes_made} changes.",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Manual patch failed: {e}",
            )

    async def run_tests(
        self,
        command: str = "pytest",
        path: str = ".",
        timeout: int | None = None,
    ) -> ToolResult:
        """Run tests using the specified command."""
        test_path = self._resolve_path(path)
        test_timeout = timeout or self.timeout

        # Security: validate command
        allowed_commands = {"pytest", "python", "npm", "yarn", "go", "cargo", "mvn", "gradle"}
        base_cmd = command.split()[0]

        if base_cmd not in allowed_commands:
            return ToolResult(
                success=False,
                output="",
                error=f"Test command not allowed: {base_cmd}. Allowed: {allowed_commands}",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=test_path if test_path.is_dir() else test_path.parent,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=test_timeout,
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return ToolResult(
                success=proc.returncode == 0,
                output=output,
                error=errors if proc.returncode != 0 else None,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(
                success=False,
                output="",
                error=f"Tests timed out after {test_timeout}s",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to run tests: {e}",
            )

    async def list_files(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = True,
    ) -> ToolResult:
        """List files in a directory."""
        list_path = self._resolve_path(path)

        if not list_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {path}",
            )

        try:
            if recursive:
                files = list(list_path.rglob(pattern))
            else:
                files = list(list_path.glob(pattern))

            # Sort and format
            file_list = []
            for f in sorted(files)[:200]:
                rel_path = f.relative_to(self.workspace_dir)
                if f.is_dir():
                    file_list.append(f"{rel_path}/")
                else:
                    size = f.stat().st_size
                    file_list.append(f"{rel_path} ({size} bytes)")

            output = "\n".join(file_list)
            if len(files) > 200:
                output += f"\n\n(truncated, {len(files)} total files)"

            return ToolResult(
                success=True,
                output=output or "No files found",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list files: {e}",
            )

    async def write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        file_path = self._resolve_path(path)

        # Security: ensure we're writing within workspace
        try:
            file_path.relative_to(self.workspace_dir)
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error="Cannot write files outside workspace",
            )

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} bytes to {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file: {e}",
            )

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to workspace."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.workspace_dir / path).resolve()

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Get OpenAI-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file. Returns file content with line numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Start line (0-indexed, default 0)",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "End line (-1 for end of file)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for a regex pattern in files. Returns matching lines with file paths and line numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Regex pattern to search for",
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search in (default: workspace root)",
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "Glob pattern to filter files (e.g., '*.py')",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 50)",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Apply a unified diff patch to modify files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patch": {
                                "type": "string",
                                "description": "Unified diff format patch content",
                            },
                            "path": {
                                "type": "string",
                                "description": "Target file path (optional for unified diffs with headers)",
                            },
                        },
                        "required": ["patch"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run tests using the specified command. Returns test output and pass/fail status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Test command to run (e.g., 'pytest', 'npm test')",
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to run tests in",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory with optional pattern matching.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern to match (default: '*')",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Search recursively (default: true)",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file. Creates parent directories if needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to write to",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
        ]
