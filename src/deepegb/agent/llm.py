"""LLM provider abstraction.

Maps a `provider` string to an Agno model object. We support:

* **local**: any OpenAI-compatible endpoint (llama.cpp `llama-server`,
  LM Studio, vLLM, …) reached via the OpenAI client.
* **anthropic**: Claude via the Anthropic SDK.
* **openai**: OpenAI's API.
* **zai**: Z.AI GLM via their OpenAI-compatible endpoint.

Configuration is read from environment variables (see `.env.example`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    from agno.models.openai import OpenAIChat
except ImportError:  # pragma: no cover
    OpenAIChat = None  # type: ignore
try:
    from agno.models.anthropic import Claude
except ImportError:  # pragma: no cover
    Claude = None  # type: ignore


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float | None = None


def _env(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


def resolve_provider(name: str | None = None) -> ProviderConfig:
    """Return a ProviderConfig given a provider name (env-driven)."""
    name = (name or _env("DEEPEGB_PROVIDER") or "local").lower()

    if name == "local":
        return ProviderConfig(
            name="local",
            model=_env("DEEPEGB_LLM_MODEL", default="local"),
            base_url=_env("DEEPEGB_LLM_BASE_URL", default="http://127.0.0.1:8001/v1"),
            api_key=_env("DEEPEGB_LLM_API_KEY", default="sk-no-key-needed"),
            timeout=float(_env("DEEPEGB_LLM_TIMEOUT", default="120")),
        )
    if name == "anthropic":
        return ProviderConfig(
            name="anthropic",
            model=_env("ANTHROPIC_MODEL", default="claude-sonnet-4-6"),
            api_key=_env("ANTHROPIC_API_KEY"),
        )
    if name == "openai":
        return ProviderConfig(
            name="openai",
            model=_env("OPENAI_MODEL", default="gpt-5"),
            api_key=_env("OPENAI_API_KEY"),
        )
    if name == "zai":
        return ProviderConfig(
            name="zai",
            model=_env("ZAI_MODEL", default="glm-4.6"),
            base_url=_env("ZAI_BASE_URL",
                          default="https://api.z.ai/api/paas/v4/"),
            api_key=_env("ZAI_API_KEY"),
        )
    raise ValueError(f"Unknown provider: {name}")


def get_model(provider: str | None = None) -> Any:
    """Build an Agno model for the requested provider.

    Notes
    -----
    Local LLMs served by `llama.cpp` typically default to a tiny
    `n_predict` (~128 tokens) when `max_tokens` is not specified, which
    cuts responses off mid-sentence. We force a generous default of
    `DEEPEGB_LLM_MAX_TOKENS` (env, default 8192). API providers also
    benefit but at higher cost per call; override via env if needed.
    """
    cfg = resolve_provider(provider)
    max_tokens = int(_env("DEEPEGB_LLM_MAX_TOKENS", default="8192"))
    temperature = float(_env("DEEPEGB_LLM_TEMPERATURE", default="0.4"))

    if cfg.name in ("local", "openai", "zai"):
        if OpenAIChat is None:
            raise RuntimeError(
                "Agno's OpenAIChat is not available. `pip install agno openai`."
            )
        kwargs: dict[str, Any] = {
            "id": cfg.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        if cfg.timeout is not None:
            kwargs["timeout"] = cfg.timeout
        # Some Agno versions name the kwarg `max_completion_tokens` instead.
        try:
            return OpenAIChat(**kwargs)
        except TypeError:
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            try:
                return OpenAIChat(**kwargs)
            except TypeError:
                kwargs.pop("max_completion_tokens", None)
                return OpenAIChat(**kwargs)

    if cfg.name == "anthropic":
        if Claude is None:
            raise RuntimeError(
                "Agno's Claude model is not available. `pip install agno anthropic`."
            )
        kwargs = {
            "id": cfg.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        try:
            return Claude(**kwargs)
        except TypeError:
            kwargs.pop("temperature", None)
            return Claude(**kwargs)

    raise ValueError(f"Unknown provider: {cfg.name}")
