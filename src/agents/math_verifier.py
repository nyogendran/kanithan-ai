"""Deterministic math verification blocks for Tamil curriculum prompts."""

from __future__ import annotations

import re
from functools import reduce
from math import gcd
from typing import Optional

_PRIME_TRIAL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def _lcm_two(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def lcm_many(nums: list[int]) -> int:
    return reduce(_lcm_two, nums)


def _curriculum_lcm_division_steps(nums: list[int]) -> list[tuple[int, list[int]]]:
    """
    Curriculum-style LCM division ladder: repeat until all rows are 1.
    Each step: prefer smallest prime p that divides at least *two* numbers.
    If none, use smallest p that divides exactly one number > 1 (e.g. 4 in 1,4,1).
    """
    n = list(nums)
    steps: list[tuple[int, list[int]]] = []
    while any(x > 1 for x in n):
        chosen_p: Optional[int] = None
        for p in _PRIME_TRIAL:
            if p > max(n):
                break
            div_idx = [i for i, x in enumerate(n) if x % p == 0]
            if len(div_idx) >= 2:
                chosen_p = p
                break
        if chosen_p is None:
            for p in _PRIME_TRIAL:
                if p > max(n):
                    break
                div_idx = [i for i, x in enumerate(n) if x % p == 0]
                if len(div_idx) == 1 and n[div_idx[0]] > 1:
                    chosen_p = p
                    break
        if chosen_p is None:
            break
        p = chosen_p
        n = [x // p if x % p == 0 else x for x in n]
        steps.append((p, list(n)))
    return steps


class MathVerifierAgent:
    """Static helpers for factor-list and HCF verification strings (prompt injection)."""

    @staticmethod
    def positive_divisors(n: int) -> list[int]:
        """All positive divisors of n, sorted (for factor-list questions)."""
        if n <= 0 or n > 1_000_000:
            return []
        divs: set[int] = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)

    @staticmethod
    def factor_verification_block(query: str) -> Optional[str]:
        """
        When the student asks (in Tamil) for the factors of a single whole number,
        inject the exact divisor list so the LLM cannot invent wrong factors.
        Triggers on 'காரணி' + exactly one Arabic numeral in the query.
        """
        if "காரணி" not in query:
            return None
        nums = [int(x) for x in re.findall(r"\b\d+\b", query)]
        if len(nums) != 1:
            return None
        n = nums[0]
        if n <= 0:
            return None
        divs = MathVerifierAgent.positive_divisors(n)
        ta = ", ".join(str(x) for x in divs)
        return (
            "சரிபார்ப்பு (கணிதத்தில் காரணிகள் — இதை மட்டுமே சரியான முழுப் பட்டியலாகக் கொள்ளவும்):\n"
            f"எண் {n} இன் அனைத்து காரணிகள் (ஏறுவரிசையில்): {ta}\n"
            "இந்த எண்களையே காரணிகளாகப் பட்டியலிட்டு விளக்கவும்; வேறு எண்களைச் சேர்க்க வேண்டாம்; "
            "இவற்றுள் ஒன்றையும் விட்டுவிட வேண்டாம்.\n"
            "ஜோடி முறை (1×…, 2×…) போன்ற NIE முறையில் தமிழில் விளக்கவும்."
        )

    @staticmethod
    def hcf_verification_block(query: str) -> Optional[str]:
        """
        When the student asks for HCF (பொ.கா.பெ.) of two or more positive integers,
        inject gcd(...) so the model must end with the correct value and full steps.
        """
        hcf_markers = (
            "பொ.கா.பெ",
            "பொதுக் காரணி",
            "பொது காரணி",
            "பொதுக் காரணிகளுட் பெரிய",
            "பொதுக் காரணிகளுள் பெரிய",
            "பொது காரணிகளின் பெரிய",
            "பொதுக் காரணிகளின் பெரிய",
        )
        if not any(m in query for m in hcf_markers):
            return None
        nums = [int(x) for x in re.findall(r"\b\d+\b", query)]
        if len(nums) < 2:
            return None
        if any(n <= 0 for n in nums):
            return None
        if any(n > 1_000_000 for n in nums):
            return None

        g = reduce(gcd, nums)
        nums_str = ", ".join(str(x) for x in nums)
        return (
            "சரிபார்ப்பு (பொ.கா.பெ. — இறுதி எண் இதுவே; வேறு எண் எழுத வேண்டாம்):\n"
            f"எண்கள்: {nums_str}\n"
            f"பொதுக் காரணிகளுட் பெரியது (பொ.கா.பெ.) = {g}\n"
            "மேலே உள்ள NIE முறைகளில் (காரணிப் பட்டியல் / முதன்மைக் காரணி மரம் / வகுத்தல் ஏணி) "
            "படிப்படியாக விளக்கி, இறுதியில் இந்தச் சரிபார்ப்பு எண்ணை உறுதிப்படுத்தவும்.\n"
            "மாணவர் 'காண்க' என்று கேட்டுள்ளார்: இறுதி விடையை மறைக்காமல் முழு தீர்வைத் தமிழில் தரவும்."
        )

    @staticmethod
    def lcm_verification_block(query: str) -> Optional[str]:
        """
        When the student asks for LCM (பொ.ம.சி.), inject exact lcm value and a correct
        Curriculum-style division ladder so the model cannot stop early at e.g. 1,4,1.
        """
        lcm_markers = (
            "பொ.ம.சி",
            "பொது மடங்குகளுட் சிறிய",
            "பொது மடங்குகளுள் சிறிய",
            "பொது மடங்குகளுள் சிறிய",
            "மீச்சிறு பொது மடங்கு",
        )
        if not any(m in query for m in lcm_markers):
            return None
        nums = [int(x) for x in re.findall(r"\b\d+\b", query)]
        if len(nums) < 2:
            return None
        if any(n <= 0 for n in nums):
            return None
        if any(n > 1_000_000 for n in nums):
            return None

        lcm_val = lcm_many(nums)
        steps = _curriculum_lcm_division_steps(nums)
        ladder_lines = []
        for p, row in steps:
            row_s = ", ".join(str(x) for x in row)
            ladder_lines.append(f"{p} | {row_s}")
        ladder_ta = "\n".join(ladder_lines)
        nums_str = ", ".join(str(x) for x in nums)

        return (
            "சரிபார்ப்பு (பொ.ம.சி. — NIE வகுத்தல் ஏணி; இதை அப்படியே பின்பற்றவும்):\n"
            f"எண்கள்: {nums_str}\n"
            f"பொ.ம.சி. = {lcm_val} (இறுதி விடை இதுவே; வேறு எண் எழுத வேண்டாம்).\n"
            "வகுத்தல் ஏணி விதி: ஒவ்வொரு வரியிலும் ஒரு முதன்மை எண்ணால் வகுக்கவும். "
            "முதலில் இரண்டு அல்லது அதற்கு மேற்பட்ட எண்களை வகுக்கக்கூடிய சிறிய முதன்மை எண்ணைத் தேர்ந்தெடுக்கவும்; "
            "இரண்டு எண்களும் ஒரே முதன்மையால் வகுபடாத போது மட்டும் ஒரு எண்ணை (எ.கா. 4) தொடர்ந்து வகுக்கவும். "
            "இறுதி வரி அனைத்து இடங்களிலும் 1 ஆக வரும் வரை தொடரவும் — '1, 4, 1' இல் நிறுத்துவது தவறு.\n\n"
            "சரியான வகுத்தல் ஏணி (படிவரிசை):\n"
            f"{ladder_ta}\n\n"
            f"வகுத்த முதன்மை எண்களின் பெருக்கல் = பொ.ம.சி. = {lcm_val}."
        )

    @staticmethod
    def get_verification_blocks(query: str) -> tuple[str, str, str]:
        """Return (factor_note, hcf_note, lcm_note) for system-prompt injection."""
        factor_note = ""
        factor_anchor = MathVerifierAgent.factor_verification_block(query)
        if factor_anchor:
            factor_note = (
                "\n\nஇக்கேள்வி வகை: காரணிகள் முழுப் பட்டியல் — கீழுள்ள 'சரிபார்ப்பு' பட்டியலை "
                "தமிழில் விளக்கிக் கொடுக்கலாம் (சோக்ரட்டிக் 'விடை மறைத்தல்' இங்கு பொருந்தாது).\n"
                f"{factor_anchor}\n"
            )

        hcf_note = ""
        hcf_anchor = MathVerifierAgent.hcf_verification_block(query)
        if hcf_anchor:
            hcf_note = (
                "\n\nஇக்கேள்வி வகை: பொ.கா.பெ. கணக்கீடு — கீழுள்ள 'சரிபார்ப்பு' இறுதி எண்ணை "
                "நிச்சயமாகக் கொண்டு முழுப் படிநிலையையும் தமிழில் எழுதவும் (விடையை மட்டும் கேள்வியாக "
                "மாற்றி முடிக்க வேண்டாம்).\n"
                "விளக்கப் பாதை: ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிப்படுத்தி (வகுத்தல் ஏணி அல்லது காரணி மரம்) "
                "பொதுவான முதன்மைக் காரணிகளின் பெருக்கம் = பொ.கா.பெ. எனக் காட்டவும்; அல்லது NIE முறை I "
                "போல குறுகிய காரணிப் பட்டியல் + பொதுக் காரணிகள். 1 முதல் n வரை தொடர் வகுத்தல் சோதனை வேண்டாம்.\n"
                f"{hcf_anchor}\n"
            )

        lcm_note = ""
        lcm_anchor = MathVerifierAgent.lcm_verification_block(query)
        if lcm_anchor:
            lcm_note = (
                "\n\nஇக்கேள்வி வகை: பொ.ம.சி. — கீழுள்ள 'சரிபார்ப்பு' வகுத்தல் ஏணியையும் இறுதி எண்ணையும் "
                "கட்டாயமாகப் பின்பற்றவும். ஏணியை நடுவில் நிறுத்தி தவறான பெருக்கல் செய்ய வேண்டாம்.\n"
                f"{lcm_anchor}\n"
            )

        return factor_note, hcf_note, lcm_note
