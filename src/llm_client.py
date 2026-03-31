"""Unified LLM access for Gemini (google.genai) and Ollama backends."""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

from src.config import GEMINI_API_KEY, OLLAMA_HOST, OLLAMA_MODEL


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


class LLMClient:
    def __init__(self, backend: str = "gemini", api_key: str | None = None) -> None:
        self.backend = (backend or "gemini").lower().strip()
        self._api_key = api_key if api_key is not None else GEMINI_API_KEY
        self._gemini_client: Any = None
        self._ollama_client: Any = None

    def get_gemini_client(self) -> Any:
        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client(api_key=self._api_key)
        return self._gemini_client

    def _get_ollama_client(self) -> Any:
        if self._ollama_client is None:
            from ollama import Client as OllamaClient

            self._ollama_client = OllamaClient(host=OLLAMA_HOST.rstrip("/"))
        return self._ollama_client

    def _gemini_generate_config(
        self,
        types: Any,
        *,
        system: str,
        temperature: float,
        max_output_tokens: int,
        disable_thinking: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if disable_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def _finish_reason_from_gemini_chunk(self, chunk: Any) -> str | None:
        if not chunk.candidates:
            return None
        fr = chunk.candidates[0].finish_reason
        if fr is None:
            return None
        return str(fr.name) if hasattr(fr, "name") else str(fr)

    def generate_stream(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        disable_thinking: bool = True,
    ) -> Iterator[tuple[str, str | None]]:
        if self.backend == "gemini":
            from google.genai import types

            client = self.get_gemini_client()
            cfg = self._gemini_generate_config(
                types,
                system=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=user,
                config=cfg,
            ):
                piece = chunk.text or ""
                fr = self._finish_reason_from_gemini_chunk(chunk)
                if piece:
                    yield piece, fr
                elif fr is not None:
                    yield "", fr
            return

        if self.backend == "ollama":
            client = self._get_ollama_client()
            opts: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
            stream = client.chat(
                model=model or OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options=opts,
                stream=True,
            )
            last_fr: str | None = None
            for part in stream:
                msg = getattr(part, "message", None)
                content = (getattr(msg, "content", None) or "") if msg else ""
                if content:
                    yield content, None
                if getattr(part, "done", False):
                    last_fr = "stop"
            if last_fr:
                yield "", last_fr
            return

        raise ValueError(f"Unknown backend: {self.backend}")

    def generate(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        disable_thinking: bool = True,
    ) -> tuple[str, str | None]:
        if self.backend == "gemini":
            from google.genai import types

            client = self.get_gemini_client()
            cfg = self._gemini_generate_config(
                types,
                system=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=cfg,
            )
            fr: str | None = None
            if response.candidates:
                r = response.candidates[0].finish_reason
                if r is not None:
                    fr = str(r.name) if hasattr(r, "name") else str(r)
            return (response.text or "").strip(), fr

        if self.backend == "ollama":
            client = self._get_ollama_client()
            opts: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
            r = client.chat(
                model=model or OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options=opts,
            )
            text = (r.message.content or "").strip() if r.message else ""
            return text, "stop"

        raise ValueError(f"Unknown backend: {self.backend}")

    def generate_json(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        raw, _ = self.generate(
            model,
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=True,
        )
        cleaned = _strip_json_fences(raw)
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
