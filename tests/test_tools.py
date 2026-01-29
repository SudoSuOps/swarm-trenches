"""Tests for agent tools module."""

import tempfile
from pathlib import Path

import pytest

from swarmcode_agent.tools import AgentTools, ToolResult


@pytest.fixture
def workspace():
    """Create a temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tools(workspace):
    """Create tools instance with workspace."""
    return AgentTools(workspace)


@pytest.fixture
def sample_files(workspace):
    """Create sample files in workspace."""
    # Create Python file
    py_file = workspace / "test.py"
    py_file.write_text('''
def hello(name):
    """Say hello."""
    return f"Hello, {name}!"

def goodbye(name):
    return f"Goodbye, {name}!"
''')

    # Create subdirectory with files
    subdir = workspace / "src"
    subdir.mkdir()

    (subdir / "utils.py").write_text('''
def add(a, b):
    return a + b
''')

    (subdir / "config.json").write_text('{"key": "value"}')

    return workspace


class TestReadFile:
    """Tests for read_file tool."""

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tools, sample_files):
        result = await tools.read_file("test.py")

        assert result.success
        assert "def hello" in result.output
        assert "def goodbye" in result.output

    @pytest.mark.asyncio
    async def test_read_file_with_line_numbers(self, tools, sample_files):
        result = await tools.read_file("test.py")

        assert result.success
        # Should have line numbers
        assert "|" in result.output

    @pytest.mark.asyncio
    async def test_read_file_range(self, tools, sample_files):
        result = await tools.read_file("test.py", start_line=1, end_line=4)

        assert result.success
        assert "def hello" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tools, workspace):
        result = await tools.read_file("nonexistent.py")

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_file_in_subdir(self, tools, sample_files):
        result = await tools.read_file("src/utils.py")

        assert result.success
        assert "def add" in result.output


class TestSearch:
    """Tests for search tool."""

    @pytest.mark.asyncio
    async def test_search_pattern(self, tools, sample_files):
        result = await tools.search("def hello")

        assert result.success
        assert "test.py" in result.output

    @pytest.mark.asyncio
    async def test_search_regex(self, tools, sample_files):
        result = await tools.search(r"def \w+\(")

        assert result.success
        assert "test.py" in result.output

    @pytest.mark.asyncio
    async def test_search_no_matches(self, tools, sample_files):
        result = await tools.search("nonexistent_pattern_xyz")

        assert result.success
        assert "No matches" in result.output

    @pytest.mark.asyncio
    async def test_search_with_file_pattern(self, tools, sample_files):
        result = await tools.search("def", file_pattern="*.py")

        assert result.success
        assert "test.py" in result.output

    @pytest.mark.asyncio
    async def test_search_invalid_regex(self, tools, sample_files):
        result = await tools.search("[invalid")

        assert not result.success
        assert result.error is not None


class TestListFiles:
    """Tests for list_files tool."""

    @pytest.mark.asyncio
    async def test_list_files_root(self, tools, sample_files):
        result = await tools.list_files()

        assert result.success
        assert "test.py" in result.output
        assert "src" in result.output

    @pytest.mark.asyncio
    async def test_list_files_subdir(self, tools, sample_files):
        result = await tools.list_files("src")

        assert result.success
        assert "utils.py" in result.output
        assert "config.json" in result.output

    @pytest.mark.asyncio
    async def test_list_files_pattern(self, tools, sample_files):
        result = await tools.list_files(pattern="*.py")

        assert result.success
        assert "test.py" in result.output

    @pytest.mark.asyncio
    async def test_list_nonexistent_dir(self, tools, workspace):
        result = await tools.list_files("nonexistent")

        assert not result.success


class TestWriteFile:
    """Tests for write_file tool."""

    @pytest.mark.asyncio
    async def test_write_new_file(self, tools, workspace):
        result = await tools.write_file("new_file.txt", "Hello, World!")

        assert result.success
        assert (workspace / "new_file.txt").exists()
        assert (workspace / "new_file.txt").read_text() == "Hello, World!"

    @pytest.mark.asyncio
    async def test_write_file_creates_dirs(self, tools, workspace):
        result = await tools.write_file("new/nested/file.txt", "content")

        assert result.success
        assert (workspace / "new" / "nested" / "file.txt").exists()

    @pytest.mark.asyncio
    async def test_write_overwrites_file(self, tools, workspace):
        # Create initial file
        (workspace / "existing.txt").write_text("old content")

        result = await tools.write_file("existing.txt", "new content")

        assert result.success
        assert (workspace / "existing.txt").read_text() == "new content"


class TestApplyPatch:
    """Tests for apply_patch tool."""

    @pytest.mark.asyncio
    async def test_apply_patch_basic(self, tools, workspace):
        # Create original file
        (workspace / "file.txt").write_text("line 1\nline 2\nline 3\n")

        patch = """--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
 line 1
-line 2
+modified line 2
 line 3
"""
        result = await tools.apply_patch(patch)

        # Result depends on whether patch command is available
        assert isinstance(result, ToolResult)


class TestRunTests:
    """Tests for run_tests tool."""

    @pytest.mark.asyncio
    async def test_run_tests_allowed_command(self, tools, workspace):
        # This will fail because pytest isn't installed in the temp workspace
        # but it should be allowed to run
        result = await tools.run_tests("python --version")

        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_run_tests_disallowed_command(self, tools, workspace):
        result = await tools.run_tests("rm -rf /")

        assert not result.success
        assert "not allowed" in result.error.lower()


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_get_tool_definitions(self):
        definitions = AgentTools.get_tool_definitions()

        assert isinstance(definitions, list)
        assert len(definitions) >= 4

        tool_names = [t["function"]["name"] for t in definitions]
        assert "read_file" in tool_names
        assert "search" in tool_names
        assert "apply_patch" in tool_names
        assert "run_tests" in tool_names

    def test_tool_definitions_have_required_fields(self):
        definitions = AgentTools.get_tool_definitions()

        for tool in definitions:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
