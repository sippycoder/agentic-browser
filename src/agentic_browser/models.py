"""Model-agnostic LLM router.

Mirrors Polar's disclosed stance: no single model — an orchestrator, worker, and
micro tier, each mapped to whatever frontier model is best for the job right now.
Provider and model are chosen per role via environment variables:

    AGENTIC_ORCHESTRATOR_MODEL="anthropic:claude-opus-4-6"
    AGENTIC_WORKER_MODEL="moonshot:kimi-k3"
    AGENTIC_MICRO_MODEL="moonshot:kimi-k2.7-code-highspeed"   (fast/cheap)
    AGENTIC_JUDGE_MODEL="moonshot:kimi-k3"                   (for eval rubric judging)

Keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY / MOONSHOT_API_KEY, or from a
.env file in the working directory / project root (loaded automatically, never
overrides real env vars). No key -> clear error, never a silent fallback.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ModelMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    images: list[str] = field(default_factory=list)  # base64 PNGs, user msgs only
    tool_call_id: str = ""
    tool_name: str = ""
    usage: dict = field(default_factory=dict)  # prompt/completion/cached tokens (assistant msgs)


def load_dotenv() -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite).

    Looks in the current working directory, then in the project root
    (two levels above this file: <root>/src/agentic_browser/models.py).
    Keeps API keys out of shell history and process args.
    Set AGENTIC_NO_DOTENV=1 to disable (used by tests).
    """
    if os.environ.get("AGENTIC_NO_DOTENV"):
        return
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        break


def _make_http_client():
    """httpx client that honors the egress proxy without parsing the rest of
    the environment. The sandbox's no_proxy contains bracketed IPv6 literals
    that break httpx's env parsing, so we set the proxy explicitly and disable
    trust_env. No proxy env -> direct client."""
    import httpx2

    proxy_url = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )
    kwargs: dict = {"timeout": 120.0, "trust_env": False}
    if proxy_url:
        kwargs["proxy"] = proxy_url
    return httpx2.Client(**kwargs)


def _model_for(role: str) -> str:
    env = {
        "orchestrator": "AGENTIC_ORCHESTRATOR_MODEL",
        "worker": "AGENTIC_WORKER_MODEL",
        "micro": "AGENTIC_MICRO_MODEL",
        "judge": "AGENTIC_JUDGE_MODEL",
    }[role]
    model = os.environ.get(env, "").strip()
    if not model:
        raise RuntimeError(
            f"No model configured for role '{role}'. Set {env} to e.g. "
            f"'anthropic:claude-opus-4-6' or 'openai:gpt-5.4'."
        )
    if ":" not in model:
        raise RuntimeError(
            f"Bad model spec {model!r} in {env}: expected 'provider:model-id'."
        )
    return model


class Provider:
    def generate(
        self,
        model_id: str,
        messages: list[ModelMessage],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> ModelMessage:
        raise NotImplementedError


class OpenAIProvider(Provider):
    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key, http_client=_make_http_client())

    def generate(self, model_id, messages, tools=None, max_tokens=4096) -> ModelMessage:
        payload: list[dict] = []
        for m in messages:
            if m.role == "system":
                payload.append({"role": "system", "content": m.text})
            elif m.role == "user":
                content: list[dict] = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                for img in m.images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img}",
                                "detail": "high",
                            },
                        }
                    )
                payload.append({"role": "user", "content": content})
            elif m.role == "assistant":
                entry: dict = {"role": "assistant", "content": m.text or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                payload.append(entry)
            elif m.role == "tool":
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id,
                        "content": m.text,
                    }
                )
        kwargs: dict = {"model": model_id, "messages": payload, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object"}),
                    },
                }
                for t in tools
            ]
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (choice.tool_calls or [])
        ]
        usage = _extract_openai_usage(resp)
        return ModelMessage(role="assistant", text=choice.content or "", tool_calls=calls, usage=usage)


class AnthropicProvider(Provider):
    def __init__(self) -> None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=api_key, http_client=_make_http_client())

    def generate(self, model_id, messages, tools=None, max_tokens=4096) -> ModelMessage:
        system_parts: list[str] = []
        payload: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.text)
            elif m.role == "user":
                content: list[dict] = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                for img in m.images:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img,
                            },
                        }
                    )
                payload.append({"role": "user", "content": content})
            elif m.role == "assistant":
                content = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                payload.append({"role": "assistant", "content": content})
            elif m.role == "tool":
                payload.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.text,
                            }
                        ],
                    }
                )
        tool_defs = None
        if tools:
            tool_defs = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object"}),
                }
                for t in tools
            ]
        resp = self.client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system="\n\n".join(system_parts) or None,
            messages=payload,
            tools=tool_defs or [],
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        usage = _extract_anthropic_usage(resp)
        return ModelMessage(
            role="assistant", text="\n".join(text_parts), tool_calls=calls, usage=usage
        )


class MoonshotProvider(OpenAIProvider):
    """Moonshot AI (Kimi) — OpenAI-compatible API at api.moonshot.ai.

    Key comes from MOONSHOT_API_KEY (or the project's .env file).
    """

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("MOONSHOT_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("MOONSHOT_API_KEY is not set.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
            http_client=_make_http_client(),
        )


def _extract_openai_usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {}
    details = getattr(u, "prompt_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "cached_tokens": int(getattr(details, "cached_tokens", 0) or 0) if details else 0,
    }


def _extract_anthropic_usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    if not u:
        return {}
    return {
        "prompt_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "cached_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
    }


class MockProvider(Provider):
    """Deterministic provider for smoke tests — no network, no key."""

    def __init__(self, script: list[ModelMessage]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def generate(self, model_id, messages, tools=None, max_tokens=4096) -> ModelMessage:
        self.calls.append({"model_id": model_id, "n_messages": len(messages)})
        if not self.script:
            return ModelMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="mock-1", name="finish", arguments={"answer": "done"})
                ],
            )
        return self.script.pop(0)


class Router:
    """Routes each agent role to its configured provider/model. Lazy providers.

    With a UsageLedger attached, every call is budget-checked first and
    metered after (M4 credit metering).
    """

    def __init__(self, provider: Provider | None = None, ledger=None) -> None:
        load_dotenv()
        self._override = provider
        self._providers: dict[str, Provider] = {}
        if ledger is None:
            from .usage import UsageLedger

            ledger = UsageLedger()
        self.ledger = ledger

    def _provider_for(self, provider_name: str) -> Provider:
        if self._override is not None:
            return self._override
        if provider_name not in self._providers:
            if provider_name == "openai":
                self._providers[provider_name] = OpenAIProvider()
            elif provider_name == "anthropic":
                self._providers[provider_name] = AnthropicProvider()
            elif provider_name == "moonshot":
                self._providers[provider_name] = MoonshotProvider()
            else:
                raise RuntimeError(f"Unknown provider {provider_name!r}.")
        return self._providers[provider_name]

    def generate(
        self,
        role: str,
        messages: list[ModelMessage],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        tag: str = "",
    ) -> ModelMessage:
        if self._override is not None:
            return self._override.generate("mock:model", messages, tools, max_tokens)
        if self.ledger is not None:
            self.ledger.check_budget()
        model = _model_for(role)
        provider_name, model_id = model.split(":", 1)
        resp = self._provider_for(provider_name).generate(
            model_id, messages, tools, max_tokens
        )
        if self.ledger is not None and resp.usage:
            u = resp.usage
            self.ledger.record(
                role,
                model,
                int(u.get("prompt_tokens", 0)),
                int(u.get("completion_tokens", 0)),
                int(u.get("cached_tokens", 0)),
                tag=tag,
            )
        return resp


def encode_image_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
