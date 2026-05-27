"""Native Anthropic Claude API backend for DeepEGB.

Provides a proper agentic tool-use loop using the anthropic SDK directly,
bypassing Agno.  Use this when DEEPEGB_PROVIDER=anthropic and you want
full control over the API call — prompt caching, streaming, tool execution,
and access to tools from both DeepEGB and CosmoRAG.

Key advantages over Agno's Claude wrapper
------------------------------------------
* Prompt caching: system prompt and tool definitions are marked
  ``cache_control: ephemeral`` so repeated queries within the same session
  hit the Anthropic cache and cost ~10× less to process.
* Direct streaming: text tokens stream to stdout as they arrive; tool-call
  events are shown inline.
* No version-skew: calls the Anthropic API schema directly — no Agno
  compatibility layer to break.
* Full tool set: accepts any Python callable with type-annotated signatures,
  from both DeepEGB and CosmoRAG, without extra wiring.

Usage
-----
  Set ``DEEPEGB_PROVIDER=anthropic`` and ``DEEPEGB_USE_NATIVE_CLAUDE=1``
  then run ``deepegb chat`` as usual.  Or call ``run_chat_native()`` directly.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import typing
from collections.abc import Callable
from typing import Any, Optional, get_type_hints

try:
    import anthropic as _anthropic_module
except ImportError:
    _anthropic_module = None  # type: ignore


# ---------------------------------------------------------------------------
# Python type annotation → JSON Schema
# ---------------------------------------------------------------------------

def _type_to_json_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a minimal JSON Schema dict."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    # Primitives
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    # list[X]
    if origin is list:
        item_schema = _type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # Optional[X] == Union[X, None]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])
        return {"type": "string"}

    # X | None  (Python ≥ 3.10 union syntax)
    cls_name = type(annotation).__name__
    if cls_name == "UnionType":
        non_none = [a for a in annotation.__args__ if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])
        return {"type": "string"}

    return {"type": "string"}


# ---------------------------------------------------------------------------
# NumPy-style docstring parameter parser
# ---------------------------------------------------------------------------

def _parse_numpy_params(docstring: str) -> dict[str, str]:
    """Return {param_name: description} from a NumPy-style docstring.

    Handles two common forms:
      - ``name : type_hint``  followed by indented description lines
      - ``name : description text``  (inline description, no separate type line)
    """
    params: dict[str, str] = {}
    if not docstring:
        return params

    in_params = False
    current: str | None = None
    SECTION_HEADERS = {
        "Returns", "Notes", "Examples", "Raises",
        "See Also", "References", "Attributes", "Yields",
    }

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped == "Parameters":
            in_params = True
            continue
        if stripped == "----------":
            continue
        if in_params and stripped in SECTION_HEADERS:
            break
        if not in_params:
            continue

        # New parameter: "name : ..."  (unindented)
        if " : " in stripped and not line.startswith("    ") and not line.startswith("\t"):
            name, rest = stripped.split(" : ", 1)
            name = name.strip()
            rest = rest.strip()
            if name.replace("_", "").isalnum():
                current = name
                # Capture inline description if `rest` looks descriptive
                # (i.e. not just a bare type name like "int", "str", "float")
                bare_types = {"int", "float", "str", "bool", "list", "dict",
                              "tuple", "Any", "None", "Optional"}
                first_word = rest.split()[0].rstrip(".,") if rest else ""
                if rest and first_word not in bare_types:
                    params[current] = rest
                else:
                    params[current] = ""
        elif current is not None and stripped and (line.startswith("    ") or line.startswith("\t")):
            sep = " " if params.get(current) else ""
            params[current] = params.get(current, "") + sep + stripped

    return params


# ---------------------------------------------------------------------------
# Tool spec builder
# ---------------------------------------------------------------------------

def build_tool_spec(func: Callable) -> dict:
    """Build an Anthropic-compatible tool definition from a Python callable.

    Extracts the tool name, a one-paragraph description, and a JSON Schema
    for all parameters from the function's type annotations and NumPy-style
    docstring.
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # First paragraph → description
    description = doc.split("\n\n")[0].strip().replace("\n", " ") or func.__name__

    param_docs = _parse_numpy_params(doc)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = hints.get(name, str)
        schema: dict = dict(_type_to_json_schema(ann))
        desc = param_docs.get(name, "").strip()
        if desc:
            schema["description"] = desc
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": func.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# Native Claude API agentic backend
# ---------------------------------------------------------------------------

class ClaudeAPIBackend:
    """Stateful agentic tool-use loop over the Anthropic Claude API.

    Manages conversation history, executes tool calls in a loop until
    ``stop_reason == "end_turn"``, streams text to stdout, and applies
    ``cache_control`` to the system prompt and tool definitions for lower
    latency on repeated queries within the same session.

    Parameters
    ----------
    tools : List of plain Python callables (DeepEGB + CosmoRAG tools).
    system : System prompt text.
    model : Claude model ID (default: ``claude-sonnet-4-6``).
    api_key : Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
    max_tokens : Max tokens per API call (default 8192).
    temperature : Sampling temperature (default 0.25).
    max_tool_iterations : Safety cap on the tool-use loop (default 20).
    """

    def __init__(
        self,
        tools: list[Callable],
        system: str,
        *,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.25,
        max_tool_iterations: int = 20,
    ) -> None:
        if _anthropic_module is None:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            )
        self.client = _anthropic_module.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_tool_iterations = max_tool_iterations
        self.system = system

        self._func_map: dict[str, Callable] = {f.__name__: f for f in tools}
        self._tool_specs: list[dict] = [build_tool_spec(f) for f in tools]
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_blocks(self) -> list[dict]:
        """Return system prompt as a list of typed blocks with cache_control."""
        return [
            {
                "type": "text",
                "text": self.system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _tools_for_api(self) -> list[dict]:
        """Return tool specs with cache_control on the last entry."""
        if not self._tool_specs:
            return []
        specs = [dict(s) for s in self._tool_specs]
        # Cache the full tool list — only the last item needs the marker.
        specs[-1] = dict(specs[-1])
        specs[-1]["cache_control"] = {"type": "ephemeral"}
        return specs

    def _run_tool(self, name: str, input_: dict) -> str:
        """Execute a registered tool and return its string result."""
        func = self._func_map.get(name)
        if func is None:
            return json.dumps(
                {
                    "error": "TOOL_ERROR:unknown_tool",
                    "tool": name,
                    "available": list(self._func_map),
                }
            )
        try:
            result = func(**input_)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except TypeError as exc:
            return json.dumps({"error": "TOOL_ERROR:bad_arguments", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": "TOOL_ERROR:execution_failed", "message": str(exc)})

    @staticmethod
    def _blocks_to_dicts(content: list) -> list[dict]:
        """Convert Anthropic SDK content block objects to plain dicts."""
        result = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    result.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    result.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            else:
                result.append(block)
        return result

    # ------------------------------------------------------------------
    # One API turn — streaming
    # ------------------------------------------------------------------

    def _stream_turn(self) -> tuple[str, list[dict]]:
        """Stream one API call; returns (final_text, tool_use_list)."""
        text_parts: list[str] = []
        current_tool_name = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self._system_blocks(),
            tools=self._tools_for_api(),
            messages=self.history,
        ) as stream:
            for event in stream:
                etype = getattr(event, "type", "")

                if etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", "") == "tool_use":
                        current_tool_name = getattr(block, "name", "?")
                        print(
                            f"\n[tool_call: {current_tool_name}] ",
                            end="",
                            flush=True,
                        )

                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        txt = getattr(delta, "text", "")
                        print(txt, end="", flush=True)
                        text_parts.append(txt)

                elif etype == "content_block_stop":
                    if current_tool_name:
                        current_tool_name = ""

            final_msg = stream.get_final_message()

        print()  # newline after streaming

        # Record assistant turn in history
        content_dicts = self._blocks_to_dicts(list(final_msg.content))
        if content_dicts:
            self.history.append({"role": "assistant", "content": content_dicts})

        # Extract tool-use blocks
        tool_uses = [b for b in content_dicts if b.get("type") == "tool_use"]
        return "".join(text_parts), tool_uses

    # ------------------------------------------------------------------
    # One API turn — blocking (no streaming)
    # ------------------------------------------------------------------

    def _blocking_turn(self) -> tuple[str, list[dict]]:
        """Non-streaming API call; returns (final_text, tool_use_list)."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self._system_blocks(),
            tools=self._tools_for_api(),
            messages=self.history,
        )
        content_dicts = self._blocks_to_dicts(list(response.content))
        if content_dicts:
            self.history.append({"role": "assistant", "content": content_dicts})

        text = " ".join(b["text"] for b in content_dicts if b.get("type") == "text")
        tool_uses = [b for b in content_dicts if b.get("type") == "tool_use"]
        return text, tool_uses

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, message: str, *, stream: bool = True) -> str:
        """Send a user message and run the agentic tool-use loop.

        Streams text to stdout when ``stream=True``.  Executes all tool calls
        in sequence until the model returns ``stop_reason == "end_turn"``.

        Parameters
        ----------
        message : User message text.
        stream : Stream tokens to stdout as they arrive (default True).

        Returns
        -------
        Final text response from the model.
        """
        self.history.append({"role": "user", "content": message})

        final_text = ""
        for _iteration in range(self.max_tool_iterations):
            if stream:
                text, tool_uses = self._stream_turn()
            else:
                text, tool_uses = self._blocking_turn()
            final_text = text

            if not tool_uses:
                break

            # Execute tool calls and add results to history
            tool_results = []
            for tu in tool_uses:
                name = tu["name"]
                input_ = tu.get("input", {})
                print(f"\n[executing: {name}({', '.join(f'{k}={v!r}' for k, v in input_.items())})] ", flush=True)
                result = self._run_tool(name, input_)
                # Truncated preview
                preview = result[:200] + "…" if len(result) > 200 else result
                print(f"→ {preview}", flush=True)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result,
                    }
                )
            self.history.append({"role": "user", "content": tool_results})

        return final_text

    def reset(self) -> None:
        """Clear conversation history (start a new session)."""
        self.history = []

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._func_map)


# ---------------------------------------------------------------------------
# Interactive REPL using the native backend
# ---------------------------------------------------------------------------

def run_chat_native(
    initial_message: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    tools: Optional[list[Callable]] = None,
    system: str = "",
    stream: bool = True,
) -> None:
    """Interactive REPL using the native Anthropic Claude API backend.

    Parameters
    ----------
    initial_message : Optional first user message (non-interactive single-shot
                      when stdin is not a tty).
    provider : Provider name — only "anthropic" is valid here.
    tools : List of callable tools (DeepEGB + CosmoRAG functions).
    system : System prompt text.
    stream : Stream tokens to stdout (default True).
    """
    from .llm import resolve_provider, _env  # noqa: PLC0415

    cfg = resolve_provider(provider or "anthropic")
    model = cfg.model
    api_key = cfg.api_key
    max_tokens = int(_env("DEEPEGB_LLM_MAX_TOKENS", default="8192"))
    temperature = float(_env("DEEPEGB_LLM_TEMPERATURE", default="0.25"))

    backend = ClaudeAPIBackend(
        tools=tools or [],
        system=system,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    tool_list = ", ".join(backend.tool_names) or "(none)"
    print(
        f"[DeepEGB native Claude API] model={model}  "
        f"tools={len(backend.tool_names)}: {tool_list}"
    )

    if initial_message:
        backend.chat(initial_message, stream=stream)
        if not sys.stdin.isatty():
            return

    print("\nDeepEGB (native Claude API) ready — type your request, /exit to quit.\n")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg in {"/exit", "/quit", ":q"}:
            break
        if msg == "/reset":
            backend.reset()
            print("[Conversation history cleared]")
            continue
        if msg == "/tools":
            print(f"Registered tools: {', '.join(backend.tool_names)}")
            continue
        backend.chat(msg, stream=stream)
