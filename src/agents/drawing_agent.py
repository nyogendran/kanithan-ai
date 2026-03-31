"""
Diagram specs for the NIE Tamil math tutor — Chapter 4 (Factors, HCF, LCM).

Produces structured JSON consumed by the web blackboard renderer.
Diagram types:
  factor_tree          — prime factorization tree
  division_ladder      — HCF (all must divide) or single-number prime factorization
  lcm_division_ladder  — LCM (any divides, others carry forward)
  factor_pairs         — pair method listing
  multiples_line       — number line showing common multiples
"""

from __future__ import annotations

import re
from functools import reduce
from math import gcd, lcm
from typing import Optional

from ..models import DiagramSpec, QueryContext, RetrievedContext


class DrawingAgent:
    _DIAGRAMMABLE_TOPICS: set[str] = {
        "factor_listing", "factor_listing_pair_method",
        "prime_factorization", "prime_factorization_tree", "prime_factorization_division",
        "hcf", "hcf_method_1_list", "hcf_method_2_prime", "hcf_method_3_division",
        "lcm", "lcm_prime_method", "lcm_division_method",
        "divisibility_rules", "digit_sum",
    }

    _LCM_TOPICS: set[str] = {"lcm", "lcm_prime_method", "lcm_division_method"}

    def should_draw(
        self,
        intent: str,
        retrieved_chunks: list,
        query: str,
        *,
        expected_method_number: int | None = None,
        topic: str | None = None,
    ) -> bool:
        if intent in {"DIAGRAM_REQUEST", "SHOW_METHOD"}:
            return True
        for chunk in retrieved_chunks:
            if chunk.get("diagram_trigger") is not None:
                keywords = ["காட்டு", "எப்படி", "வரை", "show", "draw", "explain"]
                if any(kw in query.lower() for kw in keywords):
                    return True
        if expected_method_number in (1, 2, 3):
            if (topic or "").lower() in self._DIAGRAMMABLE_TOPICS:
                return True
            for chunk in retrieved_chunks:
                if (chunk.get("topic") or "").lower() in self._DIAGRAMMABLE_TOPICS:
                    return True
            if intent in {"EXPLAIN", "SHOW_METHOD", "EXERCISE_REQUEST"}:
                return True
        return False

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def generate(
        self,
        ctx: QueryContext,
        retrieved: RetrievedContext,
        expected_method_number: int | None = None,
    ) -> Optional[DiagramSpec]:
        intent_val = ctx.intent.value if hasattr(ctx.intent, "value") else str(ctx.intent)
        query = ctx.raw_query
        topic = ctx.topic if hasattr(ctx, "topic") else None
        if not self.should_draw(
            intent_val, retrieved.chunks, query,
            expected_method_number=expected_method_number, topic=topic,
        ):
            return None
        nums = ctx.numbers or self._extract_numbers(query)
        if not nums:
            return None

        is_lcm = (topic or "").lower() in self._LCM_TOPICS

        desired_triggers: set[str] = set()
        if is_lcm:
            desired_triggers = {"factor_tree"} if expected_method_number == 1 else {"division_ladder"}
        elif expected_method_number == 1:
            desired_triggers = {"factor_pairs"}
        elif expected_method_number == 2:
            desired_triggers = {"factor_tree"}
        elif expected_method_number == 3:
            desired_triggers = {"division_ladder"}

        candidates = [c for c in retrieved.chunks if c.get("diagram_trigger")]

        # Priority 1: chunk matches the enforced method trigger.
        if desired_triggers:
            for chunk in candidates:
                if chunk.get("diagram_trigger") in desired_triggers:
                    spec_dict = self.generate_spec(chunk, nums, is_lcm=is_lcm)
                    if spec_dict and not spec_dict.get("error"):
                        return self._wrap(spec_dict)

        # Priority 2: any chunk with a diagram trigger (legacy).
        for chunk in candidates:
            spec_dict = self.generate_spec(chunk, nums, is_lcm=is_lcm)
            if spec_dict and not spec_dict.get("error"):
                return self._wrap(spec_dict)

        # Priority 3: deterministic generation from numbers + method.
        spec_dict = self._deterministic_spec(nums, expected_method_number, is_lcm)
        return self._wrap(spec_dict) if spec_dict else None

    def _wrap(self, spec_dict: dict) -> DiagramSpec:
        animate = spec_dict.get("animate", spec_dict.get("animate_step_by_step", True))
        return DiagramSpec(
            diagram_type=spec_dict.get("diagram", ""),
            spec=spec_dict,
            caption_ta=spec_dict.get("label_ta", ""),
            animate=bool(animate),
        )

    def _deterministic_spec(
        self, nums: list[int], method: int | None, is_lcm: bool,
    ) -> dict:
        if is_lcm:
            if method == 1:
                return self._factor_tree_spec(nums[0])
            return self._lcm_division_ladder_spec(nums)
        if method == 1:
            return self._factor_pairs_spec(nums[0])
        if method == 2:
            return self._factor_tree_spec(nums[0])
        if method == 3:
            return self._division_ladder_spec(nums)
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numbers(query: str) -> list[int]:
        return [int(n) for n in re.findall(r"\b\d+\b", query)]

    def generate_spec(
        self, chunk: dict, numbers: list | None = None, *, is_lcm: bool = False,
    ) -> dict:
        dt = chunk.get("diagram_trigger")
        if not dt:
            return {}
        if dt == "factor_tree" and numbers:
            return self._factor_tree_spec(numbers[0])
        if dt == "division_ladder" and numbers:
            if is_lcm:
                return self._lcm_division_ladder_spec(numbers)
            return self._division_ladder_spec(numbers)
        if dt == "factor_pairs" and numbers:
            return self._factor_pairs_spec(numbers[0])
        if dt == "multiples_line" and numbers:
            return self._multiples_line_spec(numbers)
        return {"diagram": dt, "error": "numbers_not_provided"}

    # ------------------------------------------------------------------
    # Factor tree
    # ------------------------------------------------------------------

    def _factor_tree_spec(self, n: int) -> dict:
        primes = self._prime_factors(n)
        tree = self._build_factor_tree(n)
        return {
            "diagram": "factor_tree",
            "root": n,
            "tree": tree,
            "prime_factors": primes,
            "result_label_ta": f"{n} = " + " × ".join(map(str, primes)),
            "highlight_primes": True,
            "animate_step_by_step": True,
            "label_ta": f"{n} இன் காரணி மரம்",
        }

    # ------------------------------------------------------------------
    # Division ladder — HCF (all must divide) / single-number factorization
    # ------------------------------------------------------------------

    def _division_ladder_spec(self, numbers: list) -> dict:
        """
        HCF ladder: divide only when ALL remaining values are divisible.
        For a single number this produces a complete prime factorization.
        Uses dynamic trial division (no fixed prime list).
        """
        steps: list[dict] = []
        remaining = list(numbers)
        divisors_used: list[int] = []

        d = 2
        while any(r > 1 for r in remaining):
            if all(r % d == 0 for r in remaining):
                after = [r // d for r in remaining]
                steps.append({"divisor": d, "before": list(remaining), "after": after})
                divisors_used.append(d)
                remaining = after
            else:
                d += 1
                if d > max(remaining):
                    break

        product = reduce(lambda a, b: a * b, divisors_used, 1)
        actual_hcf = reduce(gcd, numbers)
        is_single = len(numbers) == 1

        if is_single:
            label = f"{numbers[0]} இன் முதன்மைக் காரணிகளாக்கம் (வகுத்தல் முறை)"
            result = f"{numbers[0]} = " + " × ".join(map(str, divisors_used)) if divisors_used else ""
        else:
            label = f"{', '.join(map(str, numbers))} இன் பொ.கா.பெ. = {actual_hcf}"
            result = (
                "பொ.கா.பெ. = " + " × ".join(map(str, divisors_used)) + f" = {actual_hcf}"
                if divisors_used else f"பொ.கா.பெ. = 1"
            )

        return {
            "diagram": "division_ladder",
            "numbers": numbers,
            "steps": steps,
            "hcf_value": numbers[0] if is_single else actual_hcf,
            "hcf_product_shown": (
                " × ".join(map(str, divisors_used)) + f" = {product}"
                if divisors_used else "1"
            ),
            "result_label_ta": result,
            "animate": True,
            "label_ta": label,
        }

    # ------------------------------------------------------------------
    # LCM division ladder (any divides, others carry forward)
    # ------------------------------------------------------------------

    def _lcm_division_ladder_spec(self, numbers: list) -> dict:
        """
        NIE LCM ladder: divide by smallest prime that divides at least one value.
        Undivisible values carry forward unchanged.
        """
        steps: list[dict] = []
        remaining = list(numbers)
        divisors_used: list[int] = []

        d = 2
        while any(r > 1 for r in remaining):
            active = [r for r in remaining if r > 1]
            if any(r % d == 0 for r in active):
                after = [r // d if r % d == 0 else r for r in remaining]
                steps.append({"divisor": d, "before": list(remaining), "after": after})
                divisors_used.append(d)
                remaining = after
            else:
                d += 1
                if d > max(active):
                    break

        actual_lcm = reduce(lcm, numbers)
        product = reduce(lambda a, b: a * b, divisors_used, 1)

        return {
            "diagram": "lcm_division_ladder",
            "numbers": numbers,
            "steps": steps,
            "lcm_value": actual_lcm,
            "lcm_product_shown": (
                " × ".join(map(str, divisors_used)) + f" = {product}"
                if divisors_used else "1"
            ),
            "result_label_ta": (
                "பொ.ம.சி. = " + " × ".join(map(str, divisors_used)) + f" = {actual_lcm}"
                if divisors_used else ""
            ),
            "animate": True,
            "label_ta": f"{', '.join(map(str, numbers))} இன் பொ.ம.சி. = {actual_lcm}",
        }

    # ------------------------------------------------------------------
    # Factor pairs
    # ------------------------------------------------------------------

    def _factor_pairs_spec(self, n: int) -> dict:
        pairs: list[list[int]] = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                pairs.append([i, n // i])
        all_factors = sorted({f for pair in pairs for f in pair})
        return {
            "diagram": "factor_pairs",
            "number": n,
            "pairs": pairs,
            "all_factors": all_factors,
            "result_label_ta": f"{n} இன் காரணிகள்: {', '.join(map(str, all_factors))}",
            "label_ta": f"{n} இன் காரணி ஜோடிகள் (முறை I)",
            "show_multiplication": True,
            "animate": True,
        }

    # ------------------------------------------------------------------
    # Multiples line
    # ------------------------------------------------------------------

    def _multiples_line_spec(self, numbers: list, show_up_to: int = 48) -> dict:
        lcm_val = reduce(lcm, numbers)
        common = [lcm_val * i for i in range(1, 4) if lcm_val * i <= show_up_to * 2]
        colors = ["blue", "purple", "teal", "amber"]
        color_map = {str(n): colors[i % len(colors)] for i, n in enumerate(numbers)}
        return {
            "diagram": "multiples_line",
            "numbers": numbers,
            "show_up_to": show_up_to,
            "highlight_common": common[:3],
            "lcm_value": lcm_val,
            "result_label_ta": f"பொது மடங்குகளுட் சிறியது = {lcm_val}",
            "label_ta": f"{', '.join(map(str, numbers))} இன் மடங்குகள்",
            "color_map": color_map,
        }

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prime_factors(n: int) -> list[int]:
        factors: list[int] = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        return factors

    def _build_factor_tree(self, n: int) -> dict:
        def spf(k: int) -> int:
            if k < 2:
                return k
            d = 2
            while d * d <= k:
                if k % d == 0:
                    return d
                d += 1
            return k

        def build(k: int) -> dict:
            if k < 2:
                return {"value": k}
            p = spf(k)
            if p == k:
                return {"value": k, "is_prime": True}
            return {
                "value": k,
                "left": {"value": p, "is_prime": True},
                "right": build(k // p),
            }

        return build(n)
