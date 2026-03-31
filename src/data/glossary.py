"""
NIE Tamil mathematical terminology with regional variants.
Maps NIE formal terms to regional spoken equivalents and English.
"""

from __future__ import annotations

NIE_TERM_GLOSSARY: dict[str, str] = {
    "காரணி": "factor",
    "மடங்கு": "multiple",
    "இலக்கச் சுட்டி": "digit_sum",
    "முதன்மை எண்": "prime_number",
    "முதன்மைக் காரணி": "prime_factor",
    "பொ.கா.பெ.": "hcf",
    "பொதுக் காரணிகளுட் பெரியது": "hcf",
    "பொ.ம.சி.": "lcm",
    "பொது மடங்குகளுட் சிறியது": "lcm",
    "வகுபடும்": "divisible",
    "மீதி": "remainder",
    "பொதுக் காரணி": "common_factor",
    "முழுவெண்": "whole_number",
    "உயர் வலு": "highest_power",
    "பெருக்கல்": "multiplication",
    "வகுத்தல்": "division",
    "வகுத்தல் ஏணி": "division_ladder",
    "காரணி மரம்": "factor_tree",
}

DIALECT_NORMALIZER: dict[str, str] = {
    "வகுத்தல்க்கு": "வகுத்தல் மூலம்",
    "பண்ணுவது": "செய்வது",
    "சொல்லுங்க": "சொல்லுங்கள்",
    "இதுக்கு": "இதற்கு",
    "அதுக்கு": "அதற்கு",
    "எப்படி பண்றது": "எப்படி செய்வது",
    "factor காண்க": "காரணி காண்க",
    "HCF காண்க": "பொ.கா.பெ. காண்க",
    "LCM காண்க": "பொ.ம.சி. காண்க",
    "find பண்றது": "காண்பது",
    "calculate பண்க": "கணக்கிட்டு காண்க",
    "answer என்ன": "விடை என்ன",
    "வகுத்தல்னா": "வகுத்தல் என்றால்",
    "போடு": "எழுதுக",
    "இது என்னன்னு": "இது என்னவென்று",
    "வகுதல்": "வகுத்தல்",
}

SL_TAMIL_MATH_NORMALIZATION: dict[str, str] = {
    "வகுதல்": "வகுத்தல்",
    "factor ஆனது": "காரணி ஆகும்",
    "HCF காண்க": "பொ.கா.பெ. காண்க",
    "LCM காண்க": "பொ.ம.சி. காண்க",
    "கூட்டுறவு காரணி": "பொதுக் காரணி",
}

# Tamil number words → digits (ordered longest-first to avoid partial matches).
_TAMIL_NUMBER_WORDS: list[tuple[str, str]] = [
    ("பூஜ்ஜியம்", "0"),
    ("ஒன்று", "1"), ("இரண்டு", "2"), ("மூன்று", "3"),
    ("நான்கு", "4"), ("ஐந்து", "5"), ("ஆறு", "6"),
    ("ஏழு", "7"), ("எட்டு", "8"), ("ஒன்பது", "9"),
    ("பத்து", "10"), ("பதினொன்று", "11"), ("பன்னிரண்டு", "12"),
    ("பதிமூன்று", "13"), ("பதினான்கு", "14"), ("பதினைந்து", "15"),
    ("பதினாறு", "16"), ("பதினேழு", "17"), ("பதினெட்டு", "18"),
    ("பத்தொன்பது", "19"),
    ("இருபது", "20"), ("இருபத்தொன்று", "21"), ("இருபத்திரண்டு", "22"),
    ("இருபத்திமூன்று", "23"), ("இருபத்தினான்கு", "24"), ("இருபத்தைந்து", "25"),
    ("இருபத்தாறு", "26"), ("இருபத்தேழு", "27"), ("இருபத்தெட்டு", "28"),
    ("இருபத்தொன்பது", "29"),
    ("முப்பது", "30"), ("முப்பத்தொன்று", "31"), ("முப்பத்திரண்டு", "32"),
    ("முப்பத்திமூன்று", "33"), ("முப்பத்தினான்கு", "34"), ("முப்பத்தைந்து", "35"),
    ("முப்பத்தாறு", "36"), ("முப்பத்தேழு", "37"), ("முப்பத்தெட்டு", "38"),
    ("முப்பத்தொன்பது", "39"),
    ("நாற்பது", "40"), ("ஐம்பது", "50"), ("அறுபது", "60"),
    ("எழுபது", "70"), ("எண்பது", "80"), ("தொண்ணூறு", "90"),
    ("நூறு", "100"), ("ஆயிரம்", "1000"),
    # Common composite forms heard in speech
    ("நாற்பத்தெட்டு", "48"), ("நாற்பத்தாறு", "46"),
    ("ஐம்பத்தாறு", "56"), ("அறுபத்தாறு", "66"),
    ("எழுபத்திரண்டு", "72"), ("தொண்ணூற்றாறு", "96"),
    ("நூற்றிப்பதினான்கு", "114"), ("நூற்றியிருபது", "120"),
    ("இருநூறு", "200"), ("முந்நூறு", "300"),
]

# Sort longest-first so "பன்னிரண்டு" matches before "இரண்டு".
_TAMIL_NUMBER_WORDS.sort(key=lambda pair: len(pair[0]), reverse=True)


def normalize_tamil_numbers(text: str) -> str:
    """Replace Tamil number words with digit equivalents in a transcript."""
    import re

    result = text

    # Compound spoken forms (ASR often splits): "இருபத்தி நான்கு", "இருபத்தி 4"
    _COMPOUND_TENS = [
        ("இருபத்தி", 20),
        ("முப்பத்தி", 30),
        ("நாற்பத்தி", 40),
        ("ஐம்பத்தி", 50),
        ("அறுபத்தி", 60),
        ("எழுபத்தி", 70),
        ("எண்பத்தி", 80),
        ("தொண்ணூற்றி", 90),
    ]
    _UNITS_TO_DIGIT = {
        "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5,
        "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9,
    }

    for tens_word, tens_val in _COMPOUND_TENS:
        # "இருபத்தி நான்கு" / "இருபத்தி நான்கு" variants
        for unit_word, u in _UNITS_TO_DIGIT.items():
            combo = f"{tens_word}\\s*{unit_word}"
            result = re.sub(combo, str(tens_val + u), result)
        # "இருபத்தி 4" (digit after space)
        result = re.sub(
            rf"{re.escape(tens_word)}\s*(\d)",
            lambda m: str(tens_val + int(m.group(1))),
            result,
        )

    # Longest-first word replacement
    for word, digit in _TAMIL_NUMBER_WORDS:
        result = result.replace(word, digit)

    # Fix garbled splits: "15 இருபத்தி 4" → often meant "15, 24" (15 + twenty-four)
    result = re.sub(
        r"(\d+)\s+இருபத்தி\s*(\d)(?!\d)",
        lambda m: f"{m.group(1)}, {20 + int(m.group(2))}",
        result,
    )

    for word, digit in _TAMIL_NUMBER_WORDS:
        suffixed = word.rstrip("ு") + "ின்"
        if suffixed in result:
            result = result.replace(suffixed, digit + "-இன்")

    return result
