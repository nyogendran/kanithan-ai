"""Validation for chapter plugin naming and placement."""

from __future__ import annotations

import re

from .base import ChapterPlugin


_CHAPTER_CODE_RE = re.compile(r"^[a-z0-9_]+$")


def validate_plugin(plugin: ChapterPlugin) -> None:
    ident = plugin.topic_pack.identity
    if ident.chapter_number != plugin.chapter:
        raise ValueError(
            f"Chapter mismatch: plugin.chapter={plugin.chapter} "
            f"but identity.chapter_number={ident.chapter_number}"
        )
    if not _CHAPTER_CODE_RE.match(ident.chapter_code):
        raise ValueError(
            f"Invalid chapter_code={ident.chapter_code!r}. "
            "Use lowercase snake_case [a-z0-9_]."
        )
    expected_module_suffix = ident.canonical_path.replace("/", ".")
    module_name = plugin.plugin_module or ""
    if not module_name.endswith(expected_module_suffix):
        raise ValueError(
            "Plugin module path does not match canonical structure. "
            f"Expected suffix '{expected_module_suffix}', got '{module_name}'."
        )
