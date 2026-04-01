"""Chapter plugin contracts for topic packs and diagram behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ChapterIdentity:
    grade: int
    subject: str
    part: int
    chapter_number: int
    chapter_code: str
    chapter_name: str

    @property
    def canonical_path(self) -> str:
        return (
            f"grade_{self.grade:02d}/"
            f"{self.subject.lower().replace(' ', '_')}/"
            f"part_{self.part}/"
            f"chapter_{self.chapter_number:02d}_{self.chapter_code}"
        )


@dataclass(frozen=True)
class ChapterTopicPack:
    identity: ChapterIdentity
    intent_priority: list[str]
    intent_keywords: dict[str, list[str]]
    topic_keywords: dict[str, list[str]]
    section_topic_map: dict[str, str]
    topic_detect_keywords: dict[str, str] = field(default_factory=dict)
    method_topic_map: dict[str, tuple[int, str]] = field(default_factory=dict)
    corpus: list[dict[str, Any]] = field(default_factory=list)
    prerequisite_graph: dict[str, list[str]] = field(default_factory=dict)
    topic_to_skill_map: dict[str, str] = field(default_factory=dict)
    default_topic: str = "factor_listing"
    hcf_word_problem_hints: tuple[str, ...] = ()
    lcm_word_problem_hints: tuple[str, ...] = ()
    skill_to_graph_entry: dict[str, str] = field(default_factory=dict)
    skill_labels_ta: dict[str, str] = field(default_factory=dict)
    diagrammable_topics: set[str] = field(default_factory=set)
    lcm_topics: set[str] = field(default_factory=set)

    def topic_to_skill(self, topic: str) -> str:
        return self.topic_to_skill_map.get(topic, "divisibility_rules")


class DiagramAdapter(Protocol):
    def normalize_numbers(self, numbers: list[int], topic: str | None) -> list[int]:
        """Clean/shape extracted numbers before diagram generation."""


@dataclass(frozen=True)
class ChapterPlugin:
    chapter: int
    topic_pack: ChapterTopicPack
    diagram_adapter: DiagramAdapter
    plugin_module: str = ""
