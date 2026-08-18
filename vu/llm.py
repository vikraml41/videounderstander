"""Claude API wrapper: one entry point for every LLM/vision call in the
pipeline, with a content-addressed disk cache.

Caching matters here more than usual: the verification loop re-runs pipeline
stages, and vision calls are the cost center. Every request is keyed by a
hash of (model, effort, system, messages) — image blocks hash by file
content — so an unchanged call is never re-billed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_MODEL = os.environ.get("VU_MODEL", "claude-opus-5")
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
               ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}


def image_block(path: Path) -> dict:
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported image type: {path}")
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _hash_content(content) -> object:
    """Replace base64 image payloads with their sha256 so cache keys stay small."""
    if isinstance(content, str):
        return content
    hashed = []
    for block in content:
        if block.get("type") == "image":
            digest = hashlib.sha256(
                block["source"]["data"].encode()).hexdigest()
            hashed.append({"type": "image", "sha256": digest})
        else:
            hashed.append(block)
    return hashed


class LLM:
    """Cached Claude client. `cache_dir=None` disables the disk cache."""

    def __init__(self, cache_dir: Path | None = None,
                 model: str = DEFAULT_MODEL):
        self.model = model
        self.cache_dir = cache_dir
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "the 'anthropic' package is not installed — "
                    "`pip install anthropic` (and set ANTHROPIC_API_KEY or "
                    "run `ant auth login`)") from e
            self._client = anthropic.Anthropic()
        return self._client

    def _cache_key(self, payload: dict) -> str:
        canon = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canon.encode()).hexdigest()

    def call(self, messages: list[dict], system: str | None = None,
             max_tokens: int = 16000, effort: str | None = None) -> str:
        """Run one request; return the concatenated text of the response."""
        key_payload = {
            "model": self.model,
            "system": system,
            "effort": effort,
            "max_tokens": max_tokens,
            "messages": [{"role": m["role"],
                          "content": _hash_content(m["content"])}
                         for m in messages],
        }
        cache_path = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{self._cache_key(key_payload)}.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text())["text"]

        kwargs: dict = {"model": self.model, "max_tokens": max_tokens,
                        "messages": messages}
        if system is not None:
            kwargs["system"] = system
        if effort is not None:
            kwargs["output_config"] = {"effort": effort}

        # stream + get_final_message so large max_tokens never hits HTTP timeouts
        with self.client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(
                "model declined the request"
                + (f" ({details.category}: {details.explanation})" if details else ""))
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"response truncated at max_tokens={max_tokens} — raise the limit")

        text = "".join(b.text for b in response.content if b.type == "text")
        if cache_path is not None:
            cache_path.write_text(json.dumps(
                {"text": text, "model": response.model,
                 "usage": response.usage.to_dict()}, ensure_ascii=False))
        return text

    def ask(self, prompt: str, images: list[Path] = (), **kwargs) -> str:
        """Single user turn: optional images followed by a text prompt."""
        content: list[dict] = [image_block(p) for p in images]
        content.append(text_block(prompt))
        return self.call([{"role": "user", "content": content}], **kwargs)


_FENCE = re.compile(r"```(?:json|yaml)?\s*\n(.*?)\n```", re.DOTALL)


def extract_json(text: str):
    """Parse a JSON object/array from a model response, tolerating code fences
    and surrounding prose."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to the outermost bracketed span
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON found in model response:\n{text[:500]}")
