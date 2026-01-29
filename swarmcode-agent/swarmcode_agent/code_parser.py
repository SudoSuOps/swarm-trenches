"""Code parsing utilities using tree-sitter with regex fallback."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    pass


@dataclass
class FunctionInfo:
    """Information about a function/method."""

    name: str
    start_line: int
    end_line: int
    signature: str
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    is_method: bool = False
    class_name: str | None = None


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    start_line: int
    end_line: int
    methods: list[FunctionInfo] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class FileSymbols:
    """Symbols extracted from a file."""

    path: Path
    language: str
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


class CodeParser(Protocol):
    """Protocol for code parsers."""

    def parse(self, content: str, path: Path) -> FileSymbols:
        """Parse source code and extract symbols."""
        ...


class RegexParser:
    """Fallback regex-based parser for multiple languages."""

    PATTERNS = {
        "python": {
            "function": r"^(?P<indent>\s*)(?P<decorators>(?:@\w+.*\n\s*)*)?def\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)(?:\s*->\s*(?P<return>[^:]+))?\s*:",
            "class": r"^(?P<indent>\s*)class\s+(?P<name>\w+)\s*(?:\((?P<bases>[^)]*)\))?\s*:",
            "import": r"^(?:from\s+[\w.]+\s+)?import\s+.+$",
        },
        "javascript": {
            "function": r"(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
            "class": r"(?:export\s+)?class\s+(?P<name>\w+)(?:\s+extends\s+(?P<bases>\w+))?",
            "import": r"^import\s+.+$",
            "arrow": r"(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
        },
        "typescript": {
            "function": r"(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)\s*(?:<[^>]+>)?\s*\((?P<params>[^)]*)\)",
            "class": r"(?:export\s+)?class\s+(?P<name>\w+)(?:<[^>]+>)?(?:\s+extends\s+(?P<bases>\w+))?",
            "import": r"^import\s+.+$",
            "interface": r"(?:export\s+)?interface\s+(?P<name>\w+)",
        },
        "go": {
            "function": r"func\s+(?:\([^)]+\)\s+)?(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
            "struct": r"type\s+(?P<name>\w+)\s+struct\s*\{",
            "import": r"^import\s+.+$",
        },
        "rust": {
            "function": r"(?:pub\s+)?(?:async\s+)?fn\s+(?P<name>\w+)\s*(?:<[^>]+>)?\s*\((?P<params>[^)]*)\)",
            "struct": r"(?:pub\s+)?struct\s+(?P<name>\w+)",
            "impl": r"impl(?:<[^>]+>)?\s+(?P<name>\w+)",
            "import": r"^use\s+.+;$",
        },
    }

    EXTENSION_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
    }

    def detect_language(self, path: Path) -> str:
        """Detect language from file extension."""
        return self.EXTENSION_TO_LANG.get(path.suffix.lower(), "unknown")

    def parse(self, content: str, path: Path) -> FileSymbols:
        """Parse source code using regex patterns."""
        language = self.detect_language(path)
        symbols = FileSymbols(path=path, language=language)

        if language not in self.PATTERNS:
            return symbols

        patterns = self.PATTERNS[language]
        lines = content.split("\n")

        # Extract imports
        if "import" in patterns:
            for match in re.finditer(patterns["import"], content, re.MULTILINE):
                symbols.imports.append(match.group(0).strip())

        # Extract functions
        if "function" in patterns:
            for match in re.finditer(patterns["function"], content, re.MULTILINE):
                start_line = content[: match.start()].count("\n") + 1
                end_line = self._find_block_end(lines, start_line - 1, language)

                func_info = FunctionInfo(
                    name=match.group("name"),
                    start_line=start_line,
                    end_line=end_line,
                    signature=match.group(0).strip(),
                )
                symbols.functions.append(func_info)

        # Extract arrow functions (JS/TS)
        if "arrow" in patterns:
            for match in re.finditer(patterns["arrow"], content, re.MULTILINE):
                start_line = content[: match.start()].count("\n") + 1
                end_line = self._find_block_end(lines, start_line - 1, language)

                func_info = FunctionInfo(
                    name=match.group("name"),
                    start_line=start_line,
                    end_line=end_line,
                    signature=match.group(0).strip(),
                )
                symbols.functions.append(func_info)

        # Extract classes
        if "class" in patterns:
            for match in re.finditer(patterns["class"], content, re.MULTILINE):
                start_line = content[: match.start()].count("\n") + 1
                end_line = self._find_block_end(lines, start_line - 1, language)

                bases = []
                if match.lastgroup and "bases" in match.groupdict() and match.group("bases"):
                    bases = [b.strip() for b in match.group("bases").split(",")]

                class_info = ClassInfo(
                    name=match.group("name"),
                    start_line=start_line,
                    end_line=end_line,
                    bases=bases,
                )
                symbols.classes.append(class_info)

        # Extract structs (Go/Rust)
        if "struct" in patterns:
            for match in re.finditer(patterns["struct"], content, re.MULTILINE):
                start_line = content[: match.start()].count("\n") + 1
                end_line = self._find_block_end(lines, start_line - 1, language)

                class_info = ClassInfo(
                    name=match.group("name"),
                    start_line=start_line,
                    end_line=end_line,
                )
                symbols.classes.append(class_info)

        return symbols

    def _find_block_end(self, lines: list[str], start_idx: int, language: str) -> int:
        """Find the end of a code block."""
        if language == "python":
            return self._find_python_block_end(lines, start_idx)
        else:
            return self._find_brace_block_end(lines, start_idx)

    def _find_python_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find end of Python indented block."""
        if start_idx >= len(lines):
            return start_idx + 1

        start_line = lines[start_idx]
        base_indent = len(start_line) - len(start_line.lstrip())

        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                return i
        return len(lines)

    def _find_brace_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find end of brace-delimited block."""
        brace_count = 0
        started = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    started = True
                elif char == "}":
                    brace_count -= 1
                    if started and brace_count == 0:
                        return i + 1
        return len(lines)


class TreeSitterParser:
    """Tree-sitter based parser for accurate AST extraction."""

    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise RuntimeError("tree-sitter not available")

        self._parsers: dict[str, tree_sitter.Parser] = {}
        self._init_parsers()

    def _init_parsers(self):
        """Initialize tree-sitter parsers for supported languages."""
        try:
            py_lang = tree_sitter.Language(tree_sitter_python.language())
            py_parser = tree_sitter.Parser(py_lang)
            self._parsers["python"] = py_parser
        except Exception:
            pass

        try:
            js_lang = tree_sitter.Language(tree_sitter_javascript.language())
            js_parser = tree_sitter.Parser(js_lang)
            self._parsers["javascript"] = js_parser
        except Exception:
            pass

        try:
            ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
            ts_parser = tree_sitter.Parser(ts_lang)
            self._parsers["typescript"] = ts_parser
        except Exception:
            pass

    def parse(self, content: str, path: Path) -> FileSymbols:
        """Parse using tree-sitter AST."""
        language = RegexParser.EXTENSION_TO_LANG.get(path.suffix.lower(), "unknown")
        symbols = FileSymbols(path=path, language=language)

        if language not in self._parsers:
            # Fallback to regex
            return RegexParser().parse(content, path)

        parser = self._parsers[language]
        tree = parser.parse(bytes(content, "utf8"))

        if language == "python":
            self._extract_python_symbols(tree, content, symbols)
        elif language in ("javascript", "typescript"):
            self._extract_js_symbols(tree, content, symbols)

        return symbols

    def _extract_python_symbols(self, tree, content: str, symbols: FileSymbols):
        """Extract symbols from Python AST."""
        lines = content.split("\n")

        def visit(node):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_info = FunctionInfo(
                        name=name_node.text.decode("utf8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        signature=lines[node.start_point[0]].strip(),
                    )
                    symbols.functions.append(func_info)

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_info = ClassInfo(
                        name=name_node.text.decode("utf8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    )
                    symbols.classes.append(class_info)

            elif node.type in ("import_statement", "import_from_statement"):
                symbols.imports.append(node.text.decode("utf8"))

            for child in node.children:
                visit(child)

        visit(tree.root_node)

    def _extract_js_symbols(self, tree, content: str, symbols: FileSymbols):
        """Extract symbols from JavaScript/TypeScript AST."""
        lines = content.split("\n")

        def visit(node):
            if node.type in ("function_declaration", "method_definition"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_info = FunctionInfo(
                        name=name_node.text.decode("utf8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        signature=lines[node.start_point[0]].strip(),
                    )
                    symbols.functions.append(func_info)

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    class_info = ClassInfo(
                        name=name_node.text.decode("utf8"),
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    )
                    symbols.classes.append(class_info)

            elif node.type == "import_statement":
                symbols.imports.append(node.text.decode("utf8"))

            for child in node.children:
                visit(child)

        visit(tree.root_node)


def get_parser() -> CodeParser:
    """Get the best available parser."""
    if TREE_SITTER_AVAILABLE:
        try:
            return TreeSitterParser()
        except Exception:
            pass
    return RegexParser()
