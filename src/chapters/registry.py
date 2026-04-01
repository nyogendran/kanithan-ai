"""Registry for chapter plugins."""

from __future__ import annotations

from .base import ChapterPlugin
from .grade_07.mathematics.part_1.chapter_04_factors_hcf_lcm import CHAPTER4_PLUGIN
from .validator import validate_plugin

validate_plugin(CHAPTER4_PLUGIN)

_PLUGINS: dict[int, ChapterPlugin] = {
    CHAPTER4_PLUGIN.chapter: CHAPTER4_PLUGIN,
}


def register_chapter_plugin(plugin: ChapterPlugin) -> None:
    validate_plugin(plugin)
    _PLUGINS[plugin.chapter] = plugin


def get_chapter_plugin(chapter: int) -> ChapterPlugin:
    return _PLUGINS.get(chapter, CHAPTER4_PLUGIN)
