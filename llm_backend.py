"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


DEFAULT_TEMPERATURE = 0.2         # scoring: ranks windows, numbers in strict JSON
DEFAULT_DETAIL_TEMPERATURE = 0.7  # detail: writes the hook, title and captions


def _temperature_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return min(max(value, 0.0), 2.0)


def scoring_temperature() -> float:
    """Sampling temperature for the scoring pass (``LLM_TEMPERATURE``)."""
    return _temperature_env("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)


def detail_temperature() -> float:
    """Sampling temperature for the detail pass (``LLM_DETAIL_TEMPERATURE``).

    The detail pass is copywriting. At the scoring pass's 0.2 a local model
    plays safe and copies the prompt's example hooks word for word ("Stop
    doing this. Seriously." on a clip about an outage), so it runs warmer.
    """
    return _temperature_env("LLM_DETAIL_TEMPERATURE", DEFAULT_DETAIL_TEMPERATURE)


REASONING_EFFORTS = ("none", "low", "medium", "high")


def _effort_env(name: str) -> Optional[str]:
    value = (os.environ.get(name) or "").strip().lower()
    return value if value in REASONING_EFFORTS else None


def reasoning_effort() -> Optional[str]:
    """Scoring pass ``reasoning_effort`` (``LLM_REASONING_EFFORT``); None = not sent.

    Thinking models (Gemma 4, Qwen 3.x, gpt-oss) reason before answering by
    default. On a scoring call Gemma 4 12B wrote 5-10k characters of hidden
    reasoning (25-48 s) and sometimes ran on until the answer came back empty;
    with "none" the same call took 4-5 s. gpt-oss cannot switch it off and
    treats "none" as its lowest level.
    """
    return _effort_env("LLM_REASONING_EFFORT")


def detail_reasoning_effort() -> Optional[str]:
    """Detail pass ``reasoning_effort`` (``LLM_DETAIL_REASONING_EFFORT``).

    Kept separate because thinking is worth it here: with it off, Gemma 4 12B
    wrote vaguer hooks ("This is actually insane.") than with it on ("The
    scandal that ended Jensen's 33-year reign"), and the detail pass is one or
    two calls per job where scoring is many.
    """
    return _effort_env("LLM_DETAIL_REASONING_EFFORT")


def _is_reasoning_rejection(resp: httpx.Response) -> bool:
    return resp.status_code in (400, 422) and "reasoning" in resp.text.lower()


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  temperature: Optional[float] = None,
                  detail: bool = False) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    ``detail`` selects the detail pass's sampling (warmer, its own reasoning
    effort) over the scoring pass's; ``temperature`` overrides either.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    if temperature is None:
        temperature = detail_temperature() if detail else scoring_temperature()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    effort = detail_reasoning_effort() if detail else reasoning_effort()
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": temperature,
                    "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            if effort:
                body["reasoning_effort"] = effort
            resp = client.post(url, json=body, headers=_headers())
            if effort and _is_reasoning_rejection(resp):
                # A server without the field: ask again without it, and do
                # not send it for the looser formats either.
                effort = None
                body.pop("reasoning_effort")
                resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost
