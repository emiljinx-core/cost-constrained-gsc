"""Provider-neutral chat interface and an Ollama implementation.

The returned tuple intentionally matches the existing GPT.call contract:
response text, raw completion data, and a provider-neutral log dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import numbers
import os
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


Message = dict[str, Any]
MessageList = list[Message]

SAFE_LOG_OPTION_KEYS = {
    "temperature",
    "num_predict",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "seed",
    "num_ctx",
}


class ChatProvider(Protocol):
    def call(
        self, messages: MessageList, **params: Any
    ) -> tuple[str, Mapping[str, Any], dict[str, Any]]:
        """Return response text, raw completion data, and a structured log."""


class OllamaProviderError(RuntimeError):
    """Base error for Ollama configuration, transport, or response failures."""


class OllamaHTTPError(OllamaProviderError):
    """Raised when Ollama returns an HTTP error response."""


class OllamaResponseError(OllamaProviderError):
    """Raised when Ollama returns invalid or incomplete JSON."""


@dataclass(frozen=True)
class OllamaConfig:
    endpoint: str
    model: str
    timeout: float
    think: bool | None
    log_prompts: bool
    log_responses: bool

    @classmethod
    def from_environment(
        cls,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        think: bool | None = None,
        log_prompts: bool | None = None,
        log_responses: bool | None = None,
    ) -> "OllamaConfig":
        configured_timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT", "120"))
        )
        if configured_timeout <= 0:
            raise ValueError("Ollama timeout must be greater than zero")

        configured_endpoint = endpoint or os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
        configured_model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        if not configured_endpoint:
            raise ValueError("Ollama endpoint must not be empty")
        if not configured_model:
            raise ValueError("Ollama model must not be empty")

        configured_think = (
            think
            if think is not None
            else _optional_bool_from_environment("OLLAMA_THINK")
        )
        configured_log_prompts = (
            log_prompts
            if log_prompts is not None
            else _bool_from_environment("OLLAMA_LOG_PROMPTS", default=False)
        )
        configured_log_responses = (
            log_responses
            if log_responses is not None
            else _bool_from_environment("OLLAMA_LOG_RESPONSES", default=False)
        )

        return cls(
            endpoint=configured_endpoint.rstrip("/"),
            model=configured_model,
            timeout=configured_timeout,
            think=configured_think,
            log_prompts=configured_log_prompts,
            log_responses=configured_log_responses,
        )


def _bool_from_environment(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be 'true' or 'false'")
    return normalized == "true"


def _optional_bool_from_environment(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return _bool_from_environment(name, default=False)


class OllamaProvider:
    """Call Ollama's non-streaming ``/api/chat`` endpoint."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        think: bool | None = None,
        log_prompts: bool | None = None,
        log_responses: bool | None = None,
    ) -> None:
        self.config = OllamaConfig.from_environment(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            think=think,
            log_prompts=log_prompts,
            log_responses=log_responses,
        )

    def call(
        self,
        messages: MessageList,
        **params: Any,
    ) -> tuple[str, Mapping[str, Any], dict[str, Any]]:
        if not messages:
            raise ValueError("At least one chat message is required")

        temperature = params.pop("temperature", None)
        max_tokens = params.pop("max_tokens", None)
        response_format = params.pop("response_format", None)
        options = dict(params.pop("options", {}))
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        request_params: dict[str, Any] = {}
        if self.config.think is not None:
            request_params["think"] = self.config.think
        if options:
            request_params["options"] = options
        if response_format == {"type": "json_object"}:
            request_params["format"] = "json"
        elif response_format is not None:
            request_params["format"] = response_format
        request_params.update(params)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        payload.update(request_params)

        request = Request(
            f"{self.config.endpoint}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw_body = response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaHTTPError(
                f"Ollama returned HTTP {exc.code}: {body}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaProviderError(
                f"Could not reach Ollama at {self.config.endpoint}: {exc}"
            ) from exc

        try:
            completion = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaResponseError("Ollama returned invalid JSON") from exc

        if not isinstance(completion, dict):
            raise OllamaResponseError("Ollama response must be a JSON object")
        message = completion.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ) or not message["content"].strip():
            raise OllamaResponseError(
                "Ollama response is missing non-empty message.content text"
            )

        response_text = message["content"]
        log = self._build_log(
            messages=messages,
            params=self._safe_log_params(request_params),
            response=response_text,
            completion=completion,
            log_prompts=self.config.log_prompts,
            log_responses=self.config.log_responses,
        )
        return response_text, completion, log

    @staticmethod
    def _safe_log_params(params: Mapping[str, Any]) -> dict[str, Any]:
        safe_params = {}
        if isinstance(params.get("think"), bool):
            safe_params["think"] = params["think"]

        options = params.get("options")
        if isinstance(options, Mapping):
            for key in SAFE_LOG_OPTION_KEYS:
                value = options.get(key)
                if key in {"temperature", "top_p", "min_p", "repeat_penalty"}:
                    if _is_finite_number(value) and not isinstance(value, bool):
                        safe_params[key] = value
                elif key in {"num_predict", "top_k", "seed", "num_ctx"}:
                    if isinstance(value, int) and not isinstance(value, bool):
                        safe_params[key] = value
        return safe_params

    @staticmethod
    def _safe_log_endpoint(endpoint: str) -> str:
        try:
            parsed = urlsplit(endpoint)
            if not parsed.scheme or not parsed.netloc:
                return "<redacted-endpoint>"
            hostname = parsed.hostname
            if not hostname:
                return "<redacted-endpoint>"
            if ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"
            port = f":{parsed.port}" if parsed.port is not None else ""
            return urlunsplit((parsed.scheme, f"{hostname}{port}", "", "", ""))
        except ValueError:
            return "<redacted-endpoint>"

    def _build_log(
        self,
        *,
        messages: Sequence[Message],
        params: Mapping[str, Any],
        response: str,
        completion: Mapping[str, Any],
        log_prompts: bool,
        log_responses: bool,
    ) -> dict[str, Any]:
        system_prompt = None
        user_prompt = None
        logged_messages = None
        if log_prompts:
            system_prompt = "\n".join(
                message["content"]
                for message in messages
                if message.get("role") == "system"
            )
            user_prompt = "\n".join(
                message["content"]
                for message in messages
                if message.get("role") == "user"
            )
            logged_messages = list(messages)
        metadata_keys = {
            "model",
            "created_at",
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        }
        metadata = {
            key: completion[key]
            for key in metadata_keys
            if key in completion
        }
        return {
            "provider": self.provider_name,
            "model": self.config.model,
            "endpoint": self._safe_log_endpoint(self.config.endpoint),
            "params": dict(params),
            "prompt": user_prompt,
            "system_prompt": system_prompt,
            "messages": logged_messages,
            "response": response if log_responses else None,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and math.isfinite(value)
