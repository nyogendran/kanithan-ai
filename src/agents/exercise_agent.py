"""Curriculum-style exercise generation — logic from adaptive_rag_chapter4 ExerciseGenerator + factor/word generators."""

from __future__ import annotations

import random
from functools import reduce
from math import gcd, lcm
from typing import Optional

from src.chapters.base import ChapterTopicPack
from src.chapters.registry import get_chapter_plugin
from ..models.messages import ExerciseBundle, QueryContext
from ..models.student import StudentProfile


class ExerciseAgent:
    def __init__(self, topic_pack: ChapterTopicPack | None = None, chapter: int = 4):
        pack = topic_pack or get_chapter_plugin(chapter).topic_pack
        self.default_topic = pack.default_topic
        self.topic_to_skill = pack.topic_to_skill

    """
    Generate exercises modelled on NIE பயிற்சி structure.
    Calibrated to student's current skill level.
    """

    @staticmethod
    def _digit_sum(n: int) -> int:
        """Canonical curriculum digit-sum: repeatedly sum digits until single digit (fix F)."""
        s = sum(int(d) for d in str(abs(n)))
        while s >= 10:
            s = sum(int(d) for d in str(s))
        return s

    def generate(self, ctx: QueryContext, student: StudentProfile) -> Optional[ExerciseBundle]:
        topic = ctx.topic if ctx.topic and ctx.topic != "unknown" else student.last_topic
        if not topic:
            topic = self.default_topic
        topic = self.topic_to_skill(topic)

        difficulty = student.get_difficulty_ceiling()

        if topic in ("divisibility_9", "divisibility_rules") and difficulty == 1:
            return self._gen_divisibility_9_pool(difficulty, topic)
        if topic in ("divisibility_9", "divisibility_rules"):
            return self._gen_divisibility_rules(difficulty)

        if topic == "digit_sum":
            return self._gen_digit_sum(difficulty)

        if topic == "prime_factorization" and difficulty == 2:
            return self._gen_prime_factorization_curriculum(difficulty)
        if topic in ("prime_factorization", "factors_via_prime"):
            return self._gen_prime_factors_pool(difficulty)

        if topic == "hcf" and difficulty == 3:
            return self._gen_hcf_curriculum(difficulty)
        if topic == "hcf":
            return self._gen_hcf_pool(difficulty)

        if topic == "lcm" and difficulty == 3:
            return self._gen_lcm_curriculum(difficulty)
        if topic == "lcm":
            return self._gen_lcm_pool(difficulty)

        if topic in ("word_problem", "word_problems"):
            return self._gen_word_problem(difficulty)

        if topic in (
            "factor_listing",
            "factor_definition",
            "factor_pairs",
            "factor_listing_pair_method",
        ):
            return self._gen_factors(difficulty)

        return self._gen_factors(difficulty)

    def _gen_divisibility_9_pool(self, difficulty: int, topic: str) -> ExerciseBundle:
        pool = [
            504,
            207,
            135,
            81,
            333,
            441,
            108,
            252,
            999,
            1008,
            362,
            415,
            700,
            921,
            234,
            567,
            100,
        ]
        numbers = random.sample(pool, 5)
        answers = [n for n in numbers if self._digit_sum(n) == 9]
        return ExerciseBundle(
            question_ta=(
                f"பின்வரும் எண்களில் 9 ஆல் மீதியின்றி வகுபடும் எண்களை "
                f"இலக்கச் சுட்டி மூலம் தீர்மானிக்கவும்:\n"
                f"{', '.join(map(str, numbers))}"
            ),
            numbers=numbers,
            difficulty=difficulty,
            topic=topic,
            hint_ta="ஒவ்வொரு எண்ணின் இலக்கங்களையும் கூட்டி இலக்கச் சுட்டி காண்க; ஒரே இலக்கம் 9 ஆகும் வரை மீண்டும் கூட்டுக",
            expected_steps=[
                "ஒவ்வொரு எண்ணின் இலக்கங்களையும் கூட்டி இலக்கச் சுட்டி காண்க",
                "இலக்கச் சுட்டி 9 ஆன எண்களைத் தேர்வு செய்க",
            ],
            answer=answers,
            method_expected="digit_sum",
        )

    def _gen_divisibility_rules(self, difficulty: int) -> ExerciseBundle:
        divisor = random.choice([2, 3, 6, 9, 4] if difficulty > 1 else [2, 3, 9])
        numbers = random.sample(range(100, 5000), 6)
        correct = [n for n in numbers if n % divisor == 0]
        return ExerciseBundle(
            question_ta=(
                f"பின்வரும் எண்களில் {divisor} ஆல் மீதியின்றி வகுபடும் எண்களை "
                f"வகுக்காமல் தெரிவு செய்க:\n{', '.join(map(str, numbers))}"
            ),
            numbers=numbers,
            difficulty=difficulty,
            topic="divisibility_rules",
            hint_ta=f"வகுபடும் விதி: {self._divisibility_rule(divisor)}",
            expected_steps=[f"ஒவ்வொரு எண்ணுக்கும் {divisor} ஆல் வகுபடும் விதியை பயன்படுத்துக"],
            answer=correct,
            method_expected="divisibility_rule",
        )

    def _divisibility_rule(self, d: int) -> str:
        rules = {
            2: "ஒன்றினிட இலக்கம் இரட்டை எண் ஆயின் 2 ஆல் வகுபடும்",
            3: "இலக்கச் சுட்டி 3 ஆல் வகுபடும் ஆயின் 3 ஆல் வகுபடும்",
            4: "கடைசி இரண்டு இலக்கங்கள் 4 ஆல் வகுபடும் ஆயின் 4 ஆல் வகுபடும்",
            5: "ஒன்றினிட இலக்கம் 0 அல்லது 5 ஆயின் 5 ஆல் வகுபடும்",
            6: "2 ஆலும் 3 ஆலும் வகுபடும் ஆயின் 6 ஆல் வகுபடும்",
            9: "இலக்கச் சுட்டி 9 ஆயின் 9 ஆல் வகுபடும்",
        }
        return rules.get(d, "")

    def _gen_digit_sum(self, difficulty: int) -> ExerciseBundle:
        numbers = random.sample(range(10, 9999), 5)
        return ExerciseBundle(
            question_ta=f"பின்வரும் எண்களின் இலக்கச் சுட்டியைக் காண்க:\n{', '.join(map(str, numbers))}",
            numbers=numbers,
            difficulty=1,
            topic="digit_sum",
            hint_ta="ஒவ்வொரு எண்ணின் இலக்கங்களையும் கூட்டுக. தனி இலக்கம் வரும் வரை திரும்பவும் கூட்டவும்.",
            expected_steps=[
                "ஒவ்வொரு இலக்கத்தையும் தனியாக எழுதுக",
                "அனைத்தையும் கூட்டுக",
                "விடை இரண்டு இலக்கமாக இருந்தால் மீண்டும் கூட்டுக",
            ],
            answer={n: self._digit_sum(n) for n in numbers},
            method_expected="digit_sum",
        )

    def _gen_prime_factorization_curriculum(self, difficulty: int) -> ExerciseBundle:
        n = random.choice([36, 48, 60, 72, 84, 90, 96, 120, 144, 180, 210, 252])
        primes = self._prime_factors_list(n)
        prod_str = " × ".join(map(str, primes))
        return ExerciseBundle(
            question_ta=f"{n} ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுக.",
            numbers=[n],
            difficulty=difficulty,
            topic="prime_factorization",
            hint_ta="மிகச் சிறிய முதன்மை எண்ணான 2 இலிருந்து தொடங்கி வகுத்தல் ஏணி முறையில் காண்க",
            expected_steps=[
                f"{n} ÷ {primes[0]} = {n // primes[0]}",
                "விடை 1 ஆகும் வரை தொடரவும்",
                f"{n} = {prod_str}",
            ],
            answer=prod_str,
            method_expected="division_ladder",
        )

    def _gen_prime_factors_pool(self, difficulty: int) -> ExerciseBundle:
        pool = {1: [12, 18, 30], 2: [48, 60, 84, 90], 3: [120, 168, 210, 252]}
        n = random.choice(pool.get(difficulty, pool[2]))
        primes = self._prime_factors_list(n)
        prod_str = " × ".join(map(str, primes))
        return ExerciseBundle(
            question_ta=f"{n} ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுக.",
            numbers=[n],
            difficulty=difficulty,
            topic="prime_factorization",
            hint_ta="மிகச் சிறிய முதன்மை எண்ணான 2 இலிருந்து தொடங்கி வகுத்தல் ஏணி வரையுங்கள்",
            expected_steps=[
                f"{n} ÷ {primes[0]} = {n // primes[0]}",
                "விடை 1 ஆகும் வரை தொடரவும்",
                f"{n} = {prod_str}",
            ],
            answer=prod_str,
            method_expected="division_ladder",
        )

    @staticmethod
    def _prime_factors_list(x: int) -> list[int]:
        factors, d = [], 2
        while x > 1:
            while x % d == 0:
                factors.append(d)
                x //= d
            d += 1
        return factors

    def _gen_hcf_curriculum(self, difficulty: int) -> ExerciseBundle:
        pairs = [
            (12, 18),
            (24, 36),
            (48, 72),
            (36, 54),
            (60, 90),
            (84, 108),
            (45, 75),
            (16, 24),
            (30, 45),
            (72, 108),
        ]
        a, b = random.choice(pairs)
        ans = gcd(a, b)
        methods_ta = [
            "முறை I: காரணிகளை பட்டியலிட்டு",
            "முறை II: முதன்மைக் காரணிகள் மூலம்",
            "முறை III: வகுத்தல் முறை மூலம்",
        ]
        return ExerciseBundle(
            question_ta=f"{a} உம் {b} உம் ஆகிய எண்களின் பொ.கா.பெ. மூன்று முறைகளில் காண்க.",
            numbers=[a, b],
            difficulty=difficulty,
            topic="hcf",
            hint_ta="மூன்று முறைகளிலும் ஒரே பொ.கா.பெ. கிடைக்க வேண்டும்",
            expected_steps=methods_ta + [f"பொ.கா.பெ. = {ans}"],
            answer=ans,
            method_expected="three_methods",
        )

    def _gen_hcf_pool(self, difficulty: int) -> ExerciseBundle:
        pairs = {
            1: [(12, 18), (24, 36)],
            2: [(48, 72), (60, 90), (84, 108)],
            3: [(72, 108, 144), (36, 54, 90)],
        }
        nums = random.choice(pairs.get(difficulty, pairs[2]))
        hcf_val = nums[0]
        for n in nums[1:]:
            hcf_val = gcd(hcf_val, n)
        return ExerciseBundle(
            question_ta=f"{', '.join(map(str, nums))} ஆகிய எண்களின் பொ.கா.பெ. காண்க.",
            numbers=list(nums),
            difficulty=difficulty,
            topic="hcf",
            hint_ta="வகுத்தல் முறை அல்லது முதன்மைக் காரணிகள் மூலம் காண்க",
            expected_steps=[
                "ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                "பொதுவான முதன்மைக் காரணிகளைக் காண்க",
                "அவற்றை பெருக்குக",
            ],
            answer=hcf_val,
            method_expected="hcf",
        )

    def _gen_lcm_curriculum(self, difficulty: int) -> ExerciseBundle:
        triples = [
            (2, 3, 4),
            (4, 6, 8),
            (3, 4, 6),
            (6, 8, 12),
            (4, 5, 10),
            (3, 5, 9),
            (2, 5, 6),
            (6, 9, 12),
        ]
        nums = random.choice(triples)
        answer = reduce(lcm, nums)
        method_ta = "வகுத்தல் முறை அல்லது முதன்மைக் காரணிகளின் உயர் வலு மூலம் காண்க"
        return ExerciseBundle(
            question_ta=f"{nums[0]}, {nums[1]}, {nums[2]} ஆகிய எண்களின் பொ.ம.சி. காண்க.",
            numbers=list(nums),
            difficulty=difficulty,
            topic="lcm",
            hint_ta=method_ta,
            expected_steps=[
                "ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                "ஒவ்வொரு முதன்மை எண்ணின் உயர் வலுவைத் தேர்வு செய்க",
                f"பொ.ம.சி. = {answer}",
            ],
            answer=answer,
            method_expected="lcm_prime_or_division",
        )

    def _gen_lcm_pool(self, difficulty: int) -> ExerciseBundle:
        pairs = {
            1: [(2, 3), (4, 6)],
            2: [(6, 8, 12), (4, 9, 12)],
            3: [(8, 12, 18), (6, 10, 15)],
        }
        nums = random.choice(pairs.get(difficulty, pairs[2]))
        lcm_val = reduce(lcm, nums)
        return ExerciseBundle(
            question_ta=f"{', '.join(map(str, nums))} ஆகிய எண்களின் பொ.ம.சி. காண்க.",
            numbers=list(nums),
            difficulty=difficulty,
            topic="lcm",
            hint_ta="முதன்மைக் காரணிகளின் உயர் வலுவைப் பெருக்குக",
            expected_steps=[
                "ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளாக பகுக்கவும்",
                "ஒவ்வொரு முதன்மை எண்ணின் உயர் வலுவைத் தேர்வு செய்க",
                "அவற்றை பெருக்குக",
            ],
            answer=lcm_val,
            method_expected="lcm",
        )

    def _gen_factors(self, difficulty: int) -> ExerciseBundle:
        pool = {1: [12, 18, 24, 36], 2: [48, 60, 72, 84, 96], 3: [120, 150, 180, 204]}
        n = random.choice(pool.get(difficulty, pool[2]))
        factors = sorted([i for i in range(1, n + 1) if n % i == 0])
        return ExerciseBundle(
            question_ta=f"{n} இன் அனைத்து காரணிகளையும் காண்க.",
            numbers=[n],
            difficulty=difficulty,
            topic="factor_listing",
            hint_ta="ஜோடி பெருக்க முறை: 1 × ?, 2 × ?, 3 × ? ... என்று காண்க",
            expected_steps=[
                f"{n} = 1 × {n}",
                "ஒவ்வொரு ஜோடியும் எழுதுக",
                "அனைத்து காரணிகளை ஏறுவரிசையில் எழுதுக",
            ],
            answer=factors,
            method_expected="factor_pairs",
        )

    def _gen_word_problem(self, difficulty: int) -> ExerciseBundle:
        problems = [
            {
                "q": "ஒரு கூடையில் 96 அப்பிள்களும் 60 ஆரஞ்சு பழங்களும் உள்ளன. இரு வகைப் பழங்களும் சம எண்ணிக்கையில் இருக்கும் வகையில் பொதிகளில் இடப்பட்டால் பெறக்கூடிய அதிகூடிய பொதிகளின் எண்ணிக்கை யாது?",
                "nums": [96, 60],
                "answer": 12,
                "topic": "hcf",
                "hint": "சம பகிர்வு → பொ.கா.பெ. பயன்படுத்தவும்",
            },
            {
                "q": "இரண்டு மணிகள் முறையே 6 நிமிடங்கள், 8 நிமிடங்களுக்கு ஒரு முறை ஒலிக்கின்றன. காலை 8.00 மணிக்கு ஒருமித்து ஒலித்தால், அவை மீண்டும் எத்தனை மணிக்கு ஒருமித்து ஒலிக்கும்?",
                "nums": [6, 8],
                "answer": "8.24 மணி",
                "topic": "lcm",
                "hint": "முதல் சந்திப்பு → பொ.ம.சி. பயன்படுத்தவும்",
            },
        ]
        prob = random.choice(problems)
        return ExerciseBundle(
            question_ta=prob["q"],
            numbers=prob["nums"],
            difficulty=3,
            topic=prob["topic"],
            hint_ta=prob["hint"],
            expected_steps=[
                "தேவையான தகவல்களை எழுதுக",
                "பொ.கா.பெ. அல்லது பொ.ம.சி. தீர்மானி",
                "கணக்கிட்டு விடை எழுதுக",
            ],
            answer=prob["answer"],
            method_expected="word_problem",
        )
