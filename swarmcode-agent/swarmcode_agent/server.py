"""FastAPI server with OpenAI-compatible API."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from swarmcode_agent.tools import AgentTools

logger = logging.getLogger(__name__)


# Request/Response Models
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "swarmcode"
    messages: list[Message]
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int | None = Field(default=4096, gt=0)
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class HealthResponse(BaseModel):
    status: str
    model: str
    tools_enabled: bool


@dataclass
class ServeConfig:
    """Configuration for the server."""

    host: str = "0.0.0.0"
    port: int = 8000
    adapter_path: Path | None = None
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_tokens: int = 4096
    temperature: float = 0.1
    workspace_dir: Path = field(default_factory=lambda: Path("./workspace"))
    enable_tools: bool = True
    tool_timeout: int = 60


class CodeAgent:
    """Coding agent with tool execution capabilities."""

    def __init__(self, config: ServeConfig):
        self.config = config
        self.tools = AgentTools(config.workspace_dir, config.tool_timeout)
        self._model = None
        self._tokenizer = None

    async def initialize(self):
        """Initialize the model (lazy loading)."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"Loading model: {self.config.base_model}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
            )

            # Check for adapter
            if self.config.adapter_path and self.config.adapter_path.exists():
                from peft import PeftModel

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
                logger.info(f"Loaded adapter from {self.config.adapter_path}")
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning("transformers not available, using mock responses")
            self._model = "mock"
            self._tokenizer = "mock"

    async def generate(
        self,
        messages: list[Message],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[dict] | None]:
        """Generate a response from the model."""
        await self.initialize()

        # Format messages for the model
        formatted_messages = self._format_messages(messages)

        if self._model == "mock":
            # Mock response for testing without model
            return await self._generate_mock(formatted_messages, tools)

        try:
            import torch

            # Build prompt
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self._build_prompt(formatted_messages)

            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode
            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Check for tool calls
            tool_calls = self._parse_tool_calls(response, tools) if tools else None

            return response, tool_calls

        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e))

    async def _generate_mock(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> tuple[str, list[dict] | None]:
        """Generate mock response for testing."""
        last_message = messages[-1]["content"] if messages else ""

        # Check if tools are requested
        if tools and "read" in last_message.lower():
            return "", [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "README.md"}),
                    },
                }
            ]

        return f"Mock response to: {last_message[:100]}...", None

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Format messages for the model."""
        formatted = []
        for msg in messages:
            entry = {"role": msg.role}

            if msg.content:
                entry["content"] = msg.content
            elif msg.tool_calls:
                entry["content"] = json.dumps(msg.tool_calls)

            if msg.name:
                entry["name"] = msg.name

            formatted.append(entry)

        return formatted

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build a simple prompt from messages."""
        parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|ASSISTANT|>\n")
        return "\n".join(parts)

    def _parse_tool_calls(
        self,
        response: str,
        tools: list[dict],
    ) -> list[dict] | None:
        """Parse tool calls from response."""
        tool_names = {t["function"]["name"] for t in tools if "function" in t}

        # Look for JSON tool call patterns
        patterns = [
            r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}',
            r'<tool_call>\s*(\w+)\s*\(([^)]+)\)\s*</tool_call>',
        ]

        import re

        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                tool_calls = []
                for match in matches:
                    name = match[0]
                    if name in tool_names:
                        try:
                            args = match[1] if match[1].startswith("{") else f'{{{match[1]}}}'
                            tool_calls.append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args,
                                },
                            })
                        except Exception:
                            pass

                if tool_calls:
                    return tool_calls

        return None

    async def execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        func = tool_call.get("function", {})
        name = func.get("name")
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments: {args_str}"

        result = await self.tools.execute(name, args)

        if result.success:
            return result.output
        else:
            return f"Error: {result.error}"


def create_app(config: ServeConfig) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="SwarmCode Agent API",
        description="OpenAI-compatible API for coding agent",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    agent = CodeAgent(config)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model=config.base_model,
            tools_enabled=config.enable_tools,
        )

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "swarmcode",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "swarmcode",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""
        # Add tool definitions if tools are enabled and not provided
        tools = request.tools
        if config.enable_tools and tools is None:
            tools = AgentTools.get_tool_definitions()

        # Generate response
        content, tool_calls = await agent.generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens or config.max_tokens,
            tools=tools,
        )

        # Build response message
        response_message = Message(
            role="assistant",
            content=content if not tool_calls else None,
            tool_calls=tool_calls,
        )

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else "stop"

        # Token counting (approximate)
        prompt_tokens = sum(len(m.content or "") // 4 for m in request.messages)
        completion_tokens = len(content) // 4

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.post("/v1/tools/execute")
    async def execute_tool(tool_call: dict):
        """Execute a tool call directly."""
        result = await agent.execute_tool(tool_call)
        return {"result": result}

    @app.get("/v1/tools")
    async def list_tools():
        """List available tools."""
        return {"tools": AgentTools.get_tool_definitions()}

    return app
