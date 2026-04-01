"""
DiagnosticAgent — Socratic prerequisite probing before teaching advanced topics.

Walks the PREREQUISITE_GRAPH, identifies weak skills, generates simple diagnostic
questions, evaluates answers programmatically, and decides the next conversational step.
"""

from __future__ import annotations

import random
import re
from collections import deque
from functools import reduce
from math import gcd
from typing import Any, Optional

from src.chapters.registry import get_chapter_plugin
from src.agents.math_verifier import MathVerifierAgent
from src.models.student import StudentProfile

SKILL_WEAK_THRESHOLD = 0.4
MAX_DIAGNOSTIC_QUESTIONS = 3
DELTA_CORRECT = 0.15
DELTA_CORRECT_RETRY = 0.10
DELTA_INCORRECT = -0.05

class DiagnosticAgent:
    def __init__(
        self,
        *,
        chapter: int = 4,
        prerequisite_graph: dict[str, list[str]] | None = None,
        topic_to_skill=None,
        skill_to_graph_entry: dict[str, str] | None = None,
        skill_labels_ta: dict[str, str] | None = None,
    ):
        plugin = get_chapter_plugin(chapter)
        pack = plugin.topic_pack
        self.prerequisite_graph = prerequisite_graph if prerequisite_graph is not None else pack.prerequisite_graph
        self.topic_to_skill = topic_to_skill if topic_to_skill is not None else pack.topic_to_skill
        self.skill_to_graph_entry = (
            skill_to_graph_entry if skill_to_graph_entry is not None else pack.skill_to_graph_entry
        )
        self.skill_labels_ta = skill_labels_ta if skill_labels_ta is not None else pack.skill_labels_ta

    def build_diagnostic_queue(
        self, target_topic: str, student: StudentProfile
    ) -> list[str]:
        """
        BFS the prerequisite graph from the target topic, collect all prerequisite
        skill keys where the student is weak (< 0.4), deduplicate, and cap at 3.
        Returns an ordered list of skill keys to probe (most foundational first).
        """
        entry = self.skill_to_graph_entry.get(target_topic)
        if not entry:
            entry = target_topic
        if entry not in self.prerequisite_graph:
            for k in self.prerequisite_graph:
                if self.topic_to_skill(k) == target_topic:
                    entry = k
                    break

        visited: set[str] = set()
        queue: deque[str] = deque()
        prereq_skills_ordered: list[str] = []
        seen_skills: set[str] = set()

        if entry in self.prerequisite_graph:
            for p in self.prerequisite_graph[entry]:
                queue.append(p)

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            if node in self.prerequisite_graph:
                for p in self.prerequisite_graph[node]:
                    if p not in visited:
                        queue.append(p)

            skill_key = self.topic_to_skill(node)
            if skill_key in seen_skills:
                continue
            if skill_key == target_topic:
                continue
            skill_val = student.skills.get(skill_key, 0.0)
            if skill_val < SKILL_WEAK_THRESHOLD:
                seen_skills.add(skill_key)
                prereq_skills_ordered.append(skill_key)

        prereq_skills_ordered.reverse()
        return prereq_skills_ordered[:MAX_DIAGNOSTIC_QUESTIONS]

    def generate_probe_question(self, skill_key: str) -> dict[str, Any]:
        """
        Generate a simple diagnostic question for the given skill key.
        Returns {skill, number, question_ta, answer, answer_type}.
        """
        generators = {
            "divisibility_rules": self._probe_divisibility,
            "digit_sum": self._probe_digit_sum,
            "factor_listing": self._probe_factors,
            "prime_factorization": self._probe_prime_factors,
            "hcf": self._probe_hcf,
            "lcm": self._probe_lcm,
        }
        gen = generators.get(skill_key, self._probe_factors)
        probe = gen()
        probe["skill"] = skill_key
        return probe

    def evaluate_probe_answer(
        self, probe: dict[str, Any], student_answer: str
    ) -> dict[str, Any]:
        """
        Evaluate the student's answer to a diagnostic probe programmatically.
        Returns {correct, missing, extra, hint_ta}.
        """
        answer_type = probe.get("answer_type", "factor_list")
        evaluators = {
            "factor_list": self._eval_factor_list,
            "prime_factors": self._eval_prime_factors,
            "digit_sum": self._eval_single_number,
            "divisibility_yn": self._eval_divisibility,
            "single_number": self._eval_single_number,
        }
        evaluator = evaluators.get(answer_type, self._eval_factor_list)
        return evaluator(probe, student_answer)

    @staticmethod
    def compute_skill_delta(correct: bool, was_retry: bool) -> float:
        if correct:
            return DELTA_CORRECT_RETRY if was_retry else DELTA_CORRECT
        return DELTA_INCORRECT

    def skill_label_ta(self, skill_key: str) -> str:
        return self.skill_labels_ta.get(skill_key, skill_key)

    # ── Probe generators ──

    def _probe_factors(self) -> dict[str, Any]:
        n = random.choice([12, 15, 18, 20, 24, 28, 30, 36])
        factors = MathVerifierAgent.positive_divisors(n)
        return {
            "number": n,
            "question_ta": f"{n}-இன் அனைத்து காரணிகளையும் சொல்லுங்கள்.",
            "answer": factors,
            "answer_type": "factor_list",
            "hint_on_fail_ta": (
                f"குறிப்பு: 1 × {n}, 2 × ? ... என்று ஜோடி பெருக்கம் மூலம் காண்க. "
                f"மொத்தம் {len(factors)} காரணிகள் உள்ளன."
            ),
        }

    def _probe_prime_factors(self) -> dict[str, Any]:
        n = random.choice([12, 18, 20, 24, 28, 30, 42, 45])
        factors = _prime_factorize(n)
        expr = " × ".join(str(f) for f in factors)
        return {
            "number": n,
            "question_ta": f"{n}-ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுங்கள்.",
            "answer": factors,
            "answer_type": "prime_factors",
            "hint_on_fail_ta": (
                "குறிப்பு: 2 இலிருந்து தொடங்கி வகுத்தல் ஏணி வரையுங்கள். "
                f"விடை: {n} = {expr}"
            ),
        }

    def _probe_digit_sum(self) -> dict[str, Any]:
        n = random.choice([47, 63, 156, 289, 738, 1234])
        ds = _digit_sum(n)
        return {
            "number": n,
            "question_ta": f"{n}-இன் இலக்கச் சுட்டி என்ன?",
            "answer": ds,
            "answer_type": "digit_sum",
            "hint_on_fail_ta": (
                f"குறிப்பு: {n}-இன் இலக்கங்களைக் கூட்டுங்கள். "
                "தனி இலக்கம் வரும் வரை மீண்டும் கூட்டவும்."
            ),
        }

    def _probe_divisibility(self) -> dict[str, Any]:
        divisor = random.choice([2, 3, 9])
        n = random.randint(20, 200)
        is_div = n % divisor == 0
        return {
            "number": n,
            "question_ta": f"{n} என்ற எண் {divisor} ஆல் வகுபடுமா?",
            "answer": is_div,
            "answer_type": "divisibility_yn",
            "divisor": divisor,
            "hint_on_fail_ta": self._divisibility_hint(n, divisor, is_div),
        }

    def _probe_hcf(self) -> dict[str, Any]:
        a, b = random.choice([(12, 18), (24, 36), (15, 25), (20, 30)])
        hcf_val = gcd(a, b)
        return {
            "number": [a, b],
            "question_ta": f"{a}, {b} ஆகிய எண்களின் பொ.கா.பெ. என்ன?",
            "answer": hcf_val,
            "answer_type": "single_number",
            "hint_on_fail_ta": (
                f"குறிப்பு: {a} மற்றும் {b}-இன் காரணிகளைப் பட்டியலிட்டு, "
                f"பொதுவானவற்றில் பெரியதைக் காணுங்கள்."
            ),
        }

    def _probe_lcm(self) -> dict[str, Any]:
        a, b = random.choice([(4, 6), (3, 5), (6, 8), (4, 10)])
        lcm_val = (a * b) // gcd(a, b)
        return {
            "number": [a, b],
            "question_ta": f"{a}, {b} ஆகிய எண்களின் பொ.ம.சி. என்ன?",
            "answer": lcm_val,
            "answer_type": "single_number",
            "hint_on_fail_ta": (
                f"குறிப்பு: {a} மற்றும் {b}-இன் மடங்குகளை எழுதி, "
                "முதல் பொது மடங்கைக் காணுங்கள்."
            ),
        }

    @staticmethod
    def _divisibility_hint(n: int, d: int, is_div: bool) -> str:
        answer_ta = "ஆம், வகுபடும்" if is_div else "இல்லை, வகுபடாது"
        rules = {
            2: f"{n}-இன் ஒன்றினிட இலக்கம் {'இரட்டை' if n % 2 == 0 else 'ஒற்றை'}.",
            3: f"{n}-இன் இலக்கச் சுட்டி = {_digit_sum(n)}. 3 ஆல் {'வகுபடும்' if _digit_sum(n) % 3 == 0 else 'வகுபடாது'}.",
            9: f"{n}-இன் இலக்கச் சுட்டி = {_digit_sum(n)}. 9 ஆல் {'வகுபடும்' if _digit_sum(n) % 9 == 0 else 'வகுபடாது'}.",
        }
        return f"{answer_ta}. {rules.get(d, '')}"

    # ── Answer evaluators ──

    def _eval_factor_list(
        self, probe: dict[str, Any], student_answer: str
    ) -> dict[str, Any]:
        expected = set(probe["answer"])
        given = set(_extract_numbers(student_answer))
        missing = expected - given
        extra = given - expected
        correct = (missing == set() and extra == set())
        hint = ""
        if not correct:
            if missing:
                hint = f"விடுபட்டவை: {len(missing)} காரணி(கள்). "
            if extra:
                hint += f"தவறானவை: {', '.join(str(x) for x in sorted(extra))}. "
            hint += probe.get("hint_on_fail_ta", "")
        return {
            "correct": correct,
            "missing": sorted(missing),
            "extra": sorted(extra),
            "hint_ta": hint,
        }

    def _eval_prime_factors(
        self, probe: dict[str, Any], student_answer: str
    ) -> dict[str, Any]:
        expected = sorted(probe["answer"])
        given = sorted(_extract_numbers(student_answer))
        product_ok = (
            len(given) > 0
            and reduce(lambda a, b: a * b, given, 1) == probe["number"]
        )
        all_prime = all(_is_prime(x) for x in given)
        correct = product_ok and all_prime and given == expected
        hint = ""
        if not correct:
            if not all_prime:
                hint = "சில எண்கள் முதன்மை எண்கள் அல்ல. "
            elif not product_ok:
                hint = "பெருக்கல் சரியாக இல்லை. "
            hint += probe.get("hint_on_fail_ta", "")
        return {"correct": correct, "missing": [], "extra": [], "hint_ta": hint}

    def _eval_single_number(
        self, probe: dict[str, Any], student_answer: str
    ) -> dict[str, Any]:
        expected = probe["answer"]
        nums = _extract_numbers(student_answer)
        correct = expected in nums
        hint = "" if correct else probe.get("hint_on_fail_ta", "")
        return {"correct": correct, "missing": [], "extra": [], "hint_ta": hint}

    def _eval_divisibility(
        self, probe: dict[str, Any], student_answer: str
    ) -> dict[str, Any]:
        expected = probe["answer"]
        ans_lower = student_answer.lower().strip()
        yes_markers = ("ஆம்", "ஆமா", "ஆமாம்", "வகுபடும்", "yes", "aam", "true", "சரி")
        no_markers = ("இல்லை", "வகுபடாது", "no", "illai", "false", "தவறு")
        student_says_yes = any(m in ans_lower for m in yes_markers)
        student_says_no = any(m in ans_lower for m in no_markers)
        if not student_says_yes and not student_says_no:
            nums = _extract_numbers(student_answer)
            if nums:
                student_says_yes = any(n == probe["number"] for n in nums)

        if student_says_yes and not student_says_no:
            correct = expected is True
        elif student_says_no and not student_says_yes:
            correct = expected is False
        else:
            correct = False

        hint = "" if correct else probe.get("hint_on_fail_ta", "")
        return {"correct": correct, "missing": [], "extra": [], "hint_ta": hint}


# ── Utility functions ──

def _extract_numbers(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def _digit_sum(n: int) -> int:
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n


def _prime_factorize(n: int) -> list[int]:
    factors, d = [], 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    return factors


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
