"""Tests for code parser module."""

from pathlib import Path

import pytest

from swarmcode_agent.code_parser import RegexParser, get_parser


class TestRegexParser:
    """Tests for the regex-based parser."""

    @pytest.fixture
    def parser(self):
        return RegexParser()

    def test_detect_python(self, parser):
        assert parser.detect_language(Path("test.py")) == "python"
        assert parser.detect_language(Path("module/test.py")) == "python"

    def test_detect_javascript(self, parser):
        assert parser.detect_language(Path("test.js")) == "javascript"
        assert parser.detect_language(Path("test.jsx")) == "javascript"

    def test_detect_typescript(self, parser):
        assert parser.detect_language(Path("test.ts")) == "typescript"
        assert parser.detect_language(Path("test.tsx")) == "typescript"

    def test_detect_unknown(self, parser):
        assert parser.detect_language(Path("test.xyz")) == "unknown"

    def test_parse_python_function(self, parser):
        code = '''
def hello_world(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

def another_func():
    pass
'''
        symbols = parser.parse(code, Path("test.py"))

        assert symbols.language == "python"
        assert len(symbols.functions) == 2
        assert symbols.functions[0].name == "hello_world"
        assert symbols.functions[1].name == "another_func"

    def test_parse_python_class(self, parser):
        code = '''
class MyClass(BaseClass):
    """A test class."""

    def method(self):
        pass
'''
        symbols = parser.parse(code, Path("test.py"))

        assert len(symbols.classes) == 1
        assert symbols.classes[0].name == "MyClass"
        assert "BaseClass" in symbols.classes[0].bases

    def test_parse_python_imports(self, parser):
        code = '''
import os
from pathlib import Path
from typing import Optional, List
'''
        symbols = parser.parse(code, Path("test.py"))

        assert len(symbols.imports) == 3
        assert "import os" in symbols.imports

    def test_parse_javascript_function(self, parser):
        code = '''
function greet(name) {
    return `Hello, ${name}!`;
}

async function fetchData() {
    return await fetch('/api');
}
'''
        symbols = parser.parse(code, Path("test.js"))

        assert symbols.language == "javascript"
        assert len(symbols.functions) >= 2

    def test_parse_javascript_arrow_function(self, parser):
        code = '''
const add = (a, b) => a + b;

export const multiply = (a, b) => {
    return a * b;
};
'''
        symbols = parser.parse(code, Path("test.js"))

        # Arrow functions should be detected
        assert len(symbols.functions) >= 1

    def test_parse_typescript_interface(self, parser):
        code = '''
export interface User {
    id: number;
    name: string;
}

function getUser(id: number): User {
    return { id, name: 'test' };
}
'''
        symbols = parser.parse(code, Path("test.ts"))

        assert symbols.language == "typescript"
        assert len(symbols.functions) >= 1

    def test_parse_go_function(self, parser):
        code = '''
func main() {
    fmt.Println("Hello")
}

func (s *Server) Handle(w http.ResponseWriter, r *http.Request) {
    // handler
}
'''
        symbols = parser.parse(code, Path("test.go"))

        assert symbols.language == "go"
        assert len(symbols.functions) == 2

    def test_parse_rust_function(self, parser):
        code = '''
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

async fn fetch_data() -> Result<Data, Error> {
    Ok(Data::new())
}
'''
        symbols = parser.parse(code, Path("test.rs"))

        assert symbols.language == "rust"
        assert len(symbols.functions) == 2

    def test_parse_empty_file(self, parser):
        symbols = parser.parse("", Path("test.py"))

        assert symbols.language == "python"
        assert len(symbols.functions) == 0
        assert len(symbols.classes) == 0

    def test_parse_unknown_language(self, parser):
        symbols = parser.parse("some content", Path("test.xyz"))

        assert symbols.language == "unknown"
        assert len(symbols.functions) == 0


class TestGetParser:
    """Tests for parser factory."""

    def test_get_parser_returns_parser(self):
        parser = get_parser()
        assert parser is not None

    def test_parser_can_parse_python(self):
        parser = get_parser()
        code = "def test(): pass"
        symbols = parser.parse(code, Path("test.py"))
        assert symbols.language == "python"
