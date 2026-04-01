"""Chapter 4 plugin (Factors / Prime Factors / HCF / LCM)."""

from __future__ import annotations

from src.data.curriculum_corpus import CURRICULUM_CORPUS
from src.data.prerequisite_graph import PREREQUISITE_GRAPH, TOPIC_TO_SKILL

from src.chapters.base import ChapterIdentity, ChapterPlugin, ChapterTopicPack


class Chapter4DiagramAdapter:
    def normalize_numbers(self, numbers: list[int], topic: str | None) -> list[int]:
        if not numbers:
            return []
        nums = [int(n) for n in numbers if int(n) > 0]
        t = (topic or "").lower()
        if t == "word_problem":
            large = [n for n in nums if n >= 10]
            if len(large) >= 2:
                nums = large
            if len(nums) > 3:
                nums = nums[:3]
        return nums


CHAPTER4_TOPIC_PACK = ChapterTopicPack(
    identity=ChapterIdentity(
        grade=7,
        subject="mathematics",
        part=1,
        chapter_number=4,
        chapter_code="factors_hcf_lcm",
        chapter_name="காரணிகளும் மடங்குகளும் II",
    ),
    intent_priority=[
        "CHECK_ANSWER",
        "DIAGRAM_REQUEST",
        "SHOW_METHOD",
        "EXERCISE_REQUEST",
        "WORD_PROBLEM",
        "EXPLAIN",
    ],
    intent_keywords={
        "EXPLAIN": [
            "என்றால் என்ன", "வரையறை", "விளக்கு", "புரியவில்லை", "கூறு", "எப்படி",
            "கூறுங்கள்", "விளக்குங்கள்", "விளக்கவும்", "கூறவும்", "என்ன",
            "கற்றுக்கொடு", "சொல்லுங்கள்", "what is", "explain", "teach", "define",
        ],
        "SHOW_METHOD": [
            "முறை", "எப்படி காண்பது", "எப்படி கணக்கிடுவது", "காட்டு", "காட்டுங்கள்",
            "காட்டவும்", "steps", "படிகள்", "method", "வகுத்தல் முறை", "ஏணி முறை",
            "காரணி மரம் முறை", "show method", "step by step", "எப்படி",
        ],
        "EXERCISE_REQUEST": [
            "பயிற்சி", "கேள்வி கொடு", "கணக்கு கொடு", "சோதனை", "practice", "exercise",
            "question", "problem", "கொடு", "கொடுங்கள்", "கொடுக்கவும்", "தரவும்", "தாருங்கள்",
        ],
        "CHECK_ANSWER": [
            "சரியா", "இது சரியா", "என் பதில்", "விடை", "விடை சரிதானா", "என் பதில் சரியா",
            "நான் கண்டேன்", "check", "correct", "answer", "= ",
        ],
        "DIAGRAM_REQUEST": [
            "வரை", "படம்", "draw", "diagram", "காரணி மரம்", "factor tree",
            "வகுத்தல் ஏணி", "division ladder", "number line", "மடங்கு கோடு", "காட்டு", "chart",
        ],
        "WORD_PROBLEM": [
            "கதை கணக்கு", "சிந்தனைக்கு", "பென்சில்", "மணி", "பழம்", "பொதி", "பகிர்",
            "word problem", "real life", "நிமிடம்", "நேரம்", "பூக்கள்", "மரம்",
        ],
    },
    topic_keywords={
        "divisibility_rules": ["வகுபடும்", "வகுபடாது", "÷2", "÷3", "÷9", "÷6", "÷4", "÷5"],
        "digit_sum": ["இலக்கச் சுட்டி", "digit sum"],
        "factor_listing": ["காரணி", "காரணிகள்", "factor"],
        "prime_factorization": ["முதன்மை", "prime", "காரணி மரம்", "ஏணி"],
        "hcf": ["பொ.கா.பெ.", "பொதுக் காரணி", "HCF", "GCD", "பெரியது"],
        "lcm": ["பொ.ம.சி.", "பொது மடங்கு", "LCM", "சிறியது", "மடங்கு"],
        "factors_via_prime": ["முதன்மைக் காரணி", "prime factor", "காரணிப்படுத்தல்"],
        "word_problem": ["பொதி", "மணி", "பகிர்", "சம", "பழம்", "கதை கணக்கு"],
    },
    section_topic_map={
        "divisibility_rules": "4.1",
        "digit_sum": "4.1",
        "factor_listing": "4.2",
        "prime_factorization": "4.3",
        "factors_via_prime": "4.4",
        "hcf": "4.5",
        "lcm": "4.6",
        "word_problem": "4.6",
    },
    topic_detect_keywords={
        "மீ.பொ.கா.": "hcf",
        "பொ.கா.பெ": "hcf",
        "பொதுக் காரணி": "hcf",
        "மீ.பொ.ம.": "lcm",
        "பொ.ம.சி": "lcm",
        "பொது மடங்கு": "lcm",
        "முதன்மைக் காரணி": "prime_factorization",
        "காரணி மரம்": "prime_factorization",
        "வகுத்தல் ஏணி": "prime_factorization",
        "காரணி": "factor_listing",
        "வகுபடும்": "divisibility_rules",
        "இலக்கச் சுட்டி": "digit_sum",
    },
    corpus=list(CURRICULUM_CORPUS),
    prerequisite_graph=PREREQUISITE_GRAPH,
    topic_to_skill_map=TOPIC_TO_SKILL,
    method_topic_map={
        "factor_listing_pair_method": (1, "முறை I (ஜோடி பெருக்கம் / காரணிப் பட்டியல்)"),
        "prime_factorization_tree": (2, "முறை II (காரணி மரம் / முதன்மைக் காரணிகள்)"),
        "prime_factorization_division": (3, "முறை III (வகுத்தல் ஏணி)"),
        "hcf_method_1_list": (1, "முறை I (காரணிப் பட்டியல் மூலம் பொ.கா.பெ.)"),
        "hcf_method_2_prime": (2, "முறை II (முதன்மைக் காரணிகள் மூலம் பொ.கா.பெ.)"),
        "hcf_method_3_division": (3, "முறை III (வகுத்தல் முறை மூலம் பொ.கா.பெ.)"),
        "lcm_prime_method": (1, "முறை I (முதன்மைக் காரணிகள் மூலம் பொ.ம.சி.)"),
        "lcm_division_method": (2, "முறை II (வகுத்தல் ஏணி மூலம் பொ.ம.சி.)"),
    },
    default_topic="factor_listing",
    hcf_word_problem_hints=(
        "அதி கூடிய", "அதிகபட்ச", "பொதி", "சமமான", "பகிர்", "பிரிக்க",
        "maximum", "greatest", "equal groups",
    ),
    lcm_word_problem_hints=(
        "சிறியது", "குறைந்த", "முதல் முறை", "ஒரே நேரத்தில்",
        "least", "minimum", "same time", "together again",
    ),
    skill_to_graph_entry={
        "hcf": "hcf_definition",
        "lcm": "lcm_definition",
        "factor_listing": "factor_listing",
        "prime_factorization": "prime_factorization_tree",
        "divisibility_rules": "divisibility_rules",
        "digit_sum": "digit_sum",
        "word_problems": "word_problems",
    },
    skill_labels_ta={
        "divisibility_rules": "வகுபடும் விதிகள்",
        "digit_sum": "இலக்கச் சுட்டி",
        "factor_listing": "காரணிகள்",
        "prime_factorization": "முதன்மைக் காரணிகள்",
        "hcf": "பொ.கா.பெ.",
        "lcm": "பொ.ம.சி.",
        "word_problems": "சொல் கணக்குகள்",
    },
    diagrammable_topics={
        "factor_listing", "factor_listing_pair_method",
        "prime_factorization", "prime_factorization_tree", "prime_factorization_division",
        "hcf", "hcf_method_1_list", "hcf_method_2_prime", "hcf_method_3_division",
        "lcm", "lcm_prime_method", "lcm_division_method",
        "divisibility_rules", "digit_sum",
    },
    lcm_topics={"lcm", "lcm_prime_method", "lcm_division_method"},
)


CHAPTER4_PLUGIN = ChapterPlugin(
    chapter=4,
    topic_pack=CHAPTER4_TOPIC_PACK,
    diagram_adapter=Chapter4DiagramAdapter(),
    plugin_module=__name__,
)
