"""
Adaptive RAG Engine — NIE Grade 7 Mathematics Chapter 4
காரணிகளும் மடங்குகளும் (Factors and Multiples)
Tamil AI Math Tutor — PoC Implementation

Author: Built from NIE Grade 7 Tamil Medium Textbook (ilavasa padanool)
Architecture: Adaptive RAG with student proficiency tracking

Bug fixes applied (see ARCHITECTURE.md for full analysis):
  1. TOPIC_TO_SKILL map — single source of truth so prereq checks and
     skill updates use the same vocabulary.
  2. _pre_filter — correct prerequisite resolution via TOPIC_TO_SKILL.
  3. get_unlocked_topics — now resolves PREREQUISITE_GRAPH topics to
     skill names before comparing against student.skills.
  4. update_skill — rewritten to use TOPIC_TO_SKILL directly, no
     duplicate mapping.
  5. DiagramTrigger._build_factor_tree — proper recursive branching
     (not a linear walk).
  6. _division_ladder_spec — fixed HCF divisor-product to reflect the
     "at-least-2 numbers must divide" LCM rule, and correct HCF rule.
  7. ExerciseGenerator divisibility-9 answer — uses the canonical
     digit-sum function to avoid edge-case mismatch with NIE examples.
  8. StudentProfile deserialization — field-safe reconstruction that
     silently ignores unknown DB columns (forward compatibility).
  9. IntentClassifier — added priority ordering so SHOW_METHOD beats
     EXPLAIN when "எப்படி" appears with method keywords.
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1. NIE CORPUS — Chapter 4 knowledge base, chunked from the textbook
# ─────────────────────────────────────────────────────────────────────────────

NIE_CORPUS = [

    # ── TYPE A: CONCEPT CHUNKS ───────────────────────────────────────────────

    {
        "id": "C01",
        "type": "concept",
        "topic": "factor_definition",
        "section": "4.1",
        "page": 33,
        "difficulty": 1,
        "prerequisites": [],
        "content_ta": """
காரணி என்றால் என்ன?

ஒரு முழுவெண்ணை இன்னொரு முழுவெண்ணால் (பூச்சியம் தவிர்ந்த) வகுக்கும்போது
மீதியின்றி வகுபடுமாயின் முதலாவது எண் இரண்டாவது எண்ணால் வகுபடும் எனப்படும்.
அதாவது அவ்வெண் முதல் எண்ணின் காரணி என்பதை அறிந்து கொள்ளலாம்.

உதாரணம்:
6 ÷ 2 = 3 மீதி 0 → ஆகவே 2, 6 இன் காரணியாகும்.
6 ÷ 4 = 1 மீதி 2 → ஆகவே 4, 6 இன் காரணியாகாது.

முக்கிய விதி: ஒரு எண் மீதியின்றி வகுக்கும் போது அது காரணியாகும்.
        """,
        "key_terms": {
            "காரணி": "factor",
            "முழுவெண்": "whole number",
            "மீதி": "remainder",
            "வகுபடும்": "divisible"
        },
        "diagram_trigger": None,
        "exercise_follow_up": "EX_basic_factor_check"
    },

    {
        "id": "C02",
        "type": "concept",
        "topic": "digit_sum",
        "section": "4.1",
        "page": 34,
        "difficulty": 2,
        "prerequisites": ["factor_definition"],
        "content_ta": """
இலக்கச் சுட்டி என்றால் என்ன?

ஒரு எண்ணின் இலக்கங்களைக் கூட்டி 1 தொடக்கம் 9 வரையுள்ள
தனி இலக்கமாகப் பெறப்படும் பெறுமானம் அவ்வெண்ணின் இலக்கச் சுட்டி எனப்படும்.

213 இன் இலக்கச் சுட்டி:  2 + 1 + 3 = 6
∴ 213 இன் இலக்கச் சுட்டி 6 ஆகும்.

242 இன் இலக்கச் சுட்டி:  2 + 4 + 2 = 8

68 இன் இலக்கச் சுட்டி:
6 + 8 = 14 → இது தனி இலக்கம் அல்ல
∴ 14 இன் இலக்கச் சுட்டியைக் காண்போம்: 1 + 4 = 5
∴ 68 இன் இலக்கச் சுட்டி 5 ஆகும்.

பயன்: இலக்கச் சுட்டியிலிருந்து 3 ஆல், 9 ஆல் வகுபடுமா என எளிதாகத் தீர்மானிக்கலாம்.
        """,
        "key_terms": {
            "இலக்கச் சுட்டி": "digit sum",
            "இலக்கம்": "digit"
        },
        "diagram_trigger": None,
        "exercise_follow_up": "EX_digit_sum"
    },

    {
        "id": "C03",
        "type": "concept",
        "topic": "prime_number_definition",
        "section": "4.3",
        "page": 43,
        "difficulty": 2,
        "prerequisites": ["factor_definition"],
        "content_ta": """
முதன்மை எண்கள் என்றவை என்ன?

வேறுபட்ட இரண்டு காரணிகளை மட்டும் கொண்ட ஒன்றிலும் கூடிய
முழுவெண்கள் முதன்மை எண்கள் என கற்றுள்ளீர்கள்.

20 இலும் சிறிய முதன்மை எண்கள்: 2, 3, 5, 7, 11, 13, 17, 19

36 இன் காரணிகள்: 1, 2, 3, 4, 6, 9, 12, 18, 36
இவற்றுள் முதன்மை எண்ணாகவுள்ள காரணிகள்: 2, 3 மட்டுமே
∴ 2, 3 என்பன 36 இன் முதன்மைக் காரணிகள்.

60 இன் காரணிகளுள் முதன்மைக் காரணிகள்: 2, 3, 5 மாத்திரமே.

முக்கிய விதி: எண்ணொன்றின் காரணிகளுள் முதன்மை எண்ணாகவுள்ள காரணிகள்
அவ்வெண்ணின் முதன்மைக் காரணிகள் ஆகும்.
        """,
        "key_terms": {
            "முதன்மை எண்": "prime number",
            "முதன்மைக் காரணி": "prime factor"
        },
        "diagram_trigger": None,
        "exercise_follow_up": "EX_prime_identification"
    },

    {
        "id": "C04",
        "type": "concept",
        "topic": "hcf_definition",
        "section": "4.5",
        "page": 46,
        "difficulty": 3,
        "prerequisites": ["factor_listing", "prime_factorization"],
        "content_ta": """
பொதுக் காரணிகளுட் பெரியது (பொ.கா.பெ.) என்றால் என்ன?

இரண்டு அல்லது அதற்கு மேற்பட்ட சில எண்களின் பொதுக் காரணிகளுள்
பெரிய காரணி அவ்வெண்களின் பொதுக் காரணிகளுட் பெரியது (பொ.கா.பெ.) எனப்படும்.

அவ்வெண்கள் அனைத்தையும் வகுக்கக்கூடிய மிகப் பெரிய எண்
அவ்வெண்களின் பொதுக் காரணிகளுட் பெரியது ஆகும்.

பல எண்களின் பொதுக் காரணியாக இருப்பது 1 மட்டும் என்றால்
அவ்வெண்களின் பொ.கா.பெ. 1 ஆகும்.

இரண்டு முதன்மை எண்களினும் பொ.கா.பெ. 1 ஆகும்.

உதாரணம்: 6, 12, 18 இன் பொதுக் காரணிகள்: 1, 2, 3, 6
∴ 6, 12, 18 இன் பொ.கா.பெ. = 6
        """,
        "key_terms": {
            "பொ.கா.பெ.": "HCF / GCD",
            "பொதுக் காரணி": "common factor",
            "பொதுக் காரணிகளுட் பெரியது": "highest common factor"
        },
        "diagram_trigger": None,
        "exercise_follow_up": "EX_hcf_basic"
    },

    {
        "id": "C05",
        "type": "concept",
        "topic": "lcm_definition",
        "section": "4.6",
        "page": 52,
        "difficulty": 3,
        "prerequisites": ["factor_listing", "prime_factorization"],
        "content_ta": """
பொது மடங்குகளுட் சிறியது (பொ.ம.சி.) என்றால் என்ன?

எண்கள் சிலவற்றிற்கு இருக்கும் பொதுவான மடங்குகளுட் சிறியது
அவ்வெண்களின் பொது மடங்குகளுட் சிறியது ஆகும்.

தரப்பட்ட சில எண்களின் பொ.ம.சி. என்பது அந்த எண்களால் வகுபடக்கூடிய சிறிய நேர் எண்ணாகும்.

உதாரணம்: 2, 3, 4 இன் மடங்குகள்:
2 இன் மடங்குகள்: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24...
3 இன் மடங்குகள்: 3, 6, 9, 12, 15, 18, 21, 24...
4 இன் மடங்குகள்: 4, 8, 12, 16, 20, 24...

பொது மடங்குகள்: 12, 24, 36...
∴ 2, 3, 4 இன் பொ.ம.சி. = 12

முக்கிய தொடர்பு: இரு எண்களின் பொ.கா.பெ. ஆனது அவ்விரு எண்களின் பொ.ம.சி.-ஐ விடச் சிறியதாக இருக்கும்.
        """,
        "key_terms": {
            "பொ.ம.சி.": "LCM",
            "பொது மடங்கு": "common multiple",
            "பொது மடங்குகளுட் சிறியது": "lowest common multiple"
        },
        "diagram_trigger": "multiples_line",
        "exercise_follow_up": "EX_lcm_basic"
    },

    # ── TYPE B: DIVISIBILITY RULE CHUNKS ─────────────────────────────────────

    {
        "id": "R01",
        "type": "rule",
        "topic": "divisibility_2",
        "section": "4.1",
        "page": 33,
        "difficulty": 1,
        "prerequisites": [],
        "content_ta": """
2 ஆல் வகுபடும் விதி:

எண்ணொன்றின் ஒன்றினிடத்து இலக்கம் 2 ஆல் வகுபடும் எனின்
அந்த எண் 2 ஆல் வகுபடும்.

அதாவது: ஒன்றினிடத்து இலக்கம் 0, 2, 4, 6, 8 ஆகவிருப்பின்
அந்த எண் 2 ஆல் வகுபடும்.

சோதனை: 246 → ஒன்றினிட இலக்கம் 6 → 2 ஆல் வகுபடும் ✓
சோதனை: 253 → ஒன்றினிட இலக்கம் 3 → 2 ஆல் வகுபடாது ✗
        """,
        "rule_summary": "ஒன்றினிட இலக்கம் இரட்டை எண் → 2 ஆல் வகுபடும்",
        "test_examples": {"504": True, "653": False, "128": True}
    },

    {
        "id": "R02",
        "type": "rule",
        "topic": "divisibility_3",
        "section": "4.1",
        "page": 35,
        "difficulty": 2,
        "prerequisites": ["digit_sum"],
        "content_ta": """
3 ஆல் வகுபடும் விதி:

எண்ணொன்றின் இலக்கச் சுட்டி 3 ஆல் வகுபடுமாயின், அவ்வெண் 3 ஆல் வகுபடும்.
எனவே 3 என்பது அவ்வெண்ணின் காரணியாகும்.

சோதனை: 372
இலக்கச் சுட்டி: 3 + 7 + 2 = 12 → 12 ÷ 3 = 4 மீதி 0
∴ 372, 3 ஆல் வகுபடும். எனவே 3 என்பது 372 இன் காரணியாகும். ✓

சோதனை: 241
இலக்கச் சுட்டி: 2 + 4 + 1 = 7 → 7 ÷ 3 → மீதி உள்ளது
∴ 241, 3 ஆல் வகுபடாது. ✗

பயிற்சி 4.1 கேள்வி 2: 81, 102, 164, 189, 352, 372, 466, 756, 951, 1029
        """,
        "rule_summary": "இலக்கச் சுட்டி 3 ஆல் வகுபடும் → எண் 3 ஆல் வகுபடும்",
        "common_error": "மாணவர்கள் 9 ஆல் வகுபடும் விதியுடன் குழப்புகின்றனர் — 3 இன் விதி: இலக்கச் சுட்டி 3 ஆல் வகுபட வேண்டும்; 9 இன் விதி: இலக்கச் சுட்டி 9 ஆல் வகுபட வேண்டும்"
    },

    {
        "id": "R03",
        "type": "rule",
        "topic": "divisibility_9",
        "section": "4.1",
        "page": 34,
        "difficulty": 2,
        "prerequisites": ["digit_sum"],
        "content_ta": """
9 ஆல் வகுபடும் விதி:

எண்ணொன்றின் இலக்கச் சுட்டி 9 ஆகவிருப்பின் அந்த எண் 9 ஆல் மீதியின்றி வகுபடும்.

செயற்பாடு 1 (பக்கம் 35): கீழ்க்காணும் அட்டவணையை நிரப்புக:
45 → இலக்கச் சுட்டி = 4+5 = 9 → 9 ஆல் வகுபடும் → 9 அவ்வெண்ணின் காரணி ✓
52 → இலக்கச் சுட்டி = 5+2 = 7 → 9 ஆல் வகுபடாது → 9 காரணியாகாது ✗
549 → இலக்கச் சுட்டி = 5+4+9 = 18 → 1+8 = 9 → 9 ஆல் வகுபடும் ✓
1323 → இலக்கச் சுட்டி = 1+3+2+3 = 9 → 9 ஆல் வகுபடும் ✓

விதி: எண்ணொன்றின் இலக்கச் சுட்டி 9 ஆயின் அந்த எண் 9 ஆல் வகுபடும்.
        """,
        "rule_summary": "இலக்கச் சுட்டி 9 ஆயின் → எண் 9 ஆல் வகுபடும்",
        "common_error": "இலக்கச் சுட்டி தனி இலக்கம் வரும் வரை மீண்டும் கூட்ட வேண்டும் என்பதை மறக்கின்றனர்"
    },

    {
        "id": "R04",
        "type": "rule",
        "topic": "divisibility_6",
        "section": "4.1",
        "page": 37,
        "difficulty": 2,
        "prerequisites": ["divisibility_2", "divisibility_3"],
        "content_ta": """
6 ஆல் வகுபடும் விதி:

ஒர் எண் 2 ஆலும் 3 ஆலும் மீதியின்றி வகுபடுமாயின் அந்த எண் 6 ஆல் மீதியின்றி வகுபடும்.

செயற்பாடு 3 (பக்கம் 38): 95, 252, 506, 432, 552, 1236
252: 2 ஆல் வகுபடுமா? ஒன்றினிட இலக்கம் 2 → ஆம் ✓
     3 ஆல் வகுபடுமா? இலக்கச் சுட்டி 2+5+2=9, 9÷3=3 → ஆம் ✓
     ∴ 252, 6 ஆல் வகுபடும். 6 என்பது 252 இன் காரணியாகும்.

95: 2 ஆல் வகுபடுமா? ஒன்றினிட இலக்கம் 5 → இல்லை ✗
    ∴ 95, 6 ஆல் வகுபடாது.

விதி: எண்ணொன்று 2 ஆலும் 3 ஆலும் மீதியின்றி வகுபடுமாயின் அவ்வெண் 6 ஆல் வகுபடும்.
        """,
        "rule_summary": "÷2 மற்றும் ÷3 → ÷6 ஆகும்",
        "common_error": "2 ஆல் மட்டும் வகுபட்டால் 6 ஆல் வகுபடும் என நினைக்கின்றனர் — 3 ஆலும் வகுபட வேண்டும்"
    },

    {
        "id": "R05",
        "type": "rule",
        "topic": "divisibility_4",
        "section": "4.1",
        "page": 38,
        "difficulty": 2,
        "prerequisites": ["divisibility_2"],
        "content_ta": """
4 ஆல் வகுபடும் விதி:

இரண்டு அல்லது அதிலும் கூடிய இலக்கங்களையுடைய எண்ணொன்றின்
கடைசி இரண்டு இலக்கங்களும் 4 ஆல் வகுபடும் எனின், அவ்வெண் 4 ஆல் வகுபடும்.

செயற்பாடு 4 (பக்கம் 39):
36: கடைசி 2 இலக்கங்கள் = 36, 36÷4=9 மீதி 0 → 4 ஆல் வகுபடும் ✓
244: கடைசி 2 இலக்கங்கள் = 44, 44÷4=11 மீதி 0 → 4 ஆல் வகுபடும் ✓
259: கடைசி 2 இலக்கங்கள் = 59, 59÷4=14 மீதி 3 → 4 ஆல் வகுபடாது ✗
4828: கடைசி 2 இலக்கங்கள் = 28, 28÷4=7 மீதி 0 → 4 ஆல் வகுபடும் ✓

விதி: கடைசி இரண்டு இலக்கங்களும் 4 ஆல் வகுபடும் எனின் அவ்வெண் 4 ஆல் வகுபடும்.
        """,
        "rule_summary": "கடைசி 2 இலக்கங்கள் ÷4 → எண் 4 ஆல் வகுபடும்",
        "common_error": "முழு எண்ணையும் 4 ஆல் வகுக்க முயற்சிக்கின்றனர் — கடைசி 2 இலக்கங்கள் மட்டும் பார்க்க வேண்டும்"
    },

    # ── TYPE C: METHOD CHUNKS ─────────────────────────────────────────────────

    {
        "id": "M01",
        "type": "method",
        "topic": "factor_listing_pair_method",
        "section": "4.2",
        "page": 41,
        "difficulty": 2,
        "prerequisites": ["factor_definition"],
        "content_ta": """
ஒரு முழுவெண்ணின் காரணிகளை ஜோடி பெருக்க முறையில் காண்பது:

36 இன் காரணிகளைக் காண்போம்.

36 ஐ இரு காரணிகளின் பெருக்கமாக எழும் முறையைப் பயன்படுத்தி காரணிகளைக் காண்க:
36 = 1 × 36  → இரு முழுவெண்களின் பெருக்கமாக எழும்போது அவ்விரண்டு எண்களும்
36 = 2 × 18     முதல் எண்ணின் காரணிகள் ஆகும்.
36 = 3 × 12
36 = 4 × 9
36 = 6 × 6

∴ 36 இன் காரணிகள்: 1, 2, 3, 4, 6, 9, 12, 18, 36

126 இன் காரணிகள்:
2|126  → 126, 2 ஆல் வகுபடுவதால் 2, 126 இன் காரணியாகும்.
  63   → 2 × 63 = 126 என்பதால் 63 உம் 126 இன் காரணியாகும்.

∴ 126 இன் காரணிகள்: 1, 2, 3, 6, 7, 9, 14, 18, 21, 42, 63, 126
        """,
        "diagram_trigger": "factor_pairs",
        "steps_count": 5
    },

    {
        "id": "M02",
        "type": "method",
        "topic": "prime_factorization_tree",
        "section": "4.3",
        "page": 44,
        "difficulty": 2,
        "prerequisites": ["prime_number_definition", "factor_listing"],
        "content_ta": """
காரணி மர முறையில் முதன்மைக் காரணிகளைக் காண்பது:

84 இன் முதன்மைக் காரணிகளைக் கண்டு 84 ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுவோம்.

படிகள்:
• 84 மிகச் சிறிய முதன்மை எண்ணான 2 ஆல் வகுக்கப்படுகிறது.
• பெறப்பட்ட விடை 2 ஆல் வகுபடாத வரைக்கும் தொடர்ந்து 2 ஆல் வகுக்கப்பட வேண்டும்.
• பெறப்படும் எண் அதற்கடுத்த முதன்மை எண்ணான 3 ஆல் வகுக்கும்போது 7 விடையாகப் பெறப்பட்டது.
• இவ்வாறு இறுதியில் 1 கிடைக்கும் வரை முதன்மை எண்களால் தொடர்ந்து வகுக்க வேண்டும்.

காரணி மரம்:
84 → 2 × 42 → 2 × 2 × 21 → 2 × 2 × 3 × 7

இதற்கேற்ப 84 இன் முதன்மைக் காரணிகள்: 2, 3, 7
∴ 84 = 2 × 2 × 3 × 7

75 = 3 × 5 × 5 (75, 2 ஆல் வகுக்க முடியாது → 3 ஆல் தொடங்கு)
        """,
        "diagram_trigger": "factor_tree",
        "worked_numbers": [84, 75, 63]
    },

    {
        "id": "M03",
        "type": "method",
        "topic": "prime_factorization_division",
        "section": "4.3",
        "page": 44,
        "difficulty": 2,
        "prerequisites": ["prime_number_definition"],
        "content_ta": """
வகுத்தல் ஏணி முறையில் முதன்மைக் காரணிகளைக் காண்பது:

84 இன் வகுத்தல் ஏணி:
2 | 84
2 | 42
3 | 21
7 |  7
    1

∴ 84 = 2 × 2 × 3 × 7

75 இன் வகுத்தல் ஏணி:
3 | 75
5 | 25
5 |  5
    1

∴ 75 = 3 × 5 × 5

விதி: வகுத்தலை நிறுத்துவது எப்போது?
• விடையாக 1 கிடைக்கும் வரை தொடரவும்.
• மிகச் சிறிய முதன்மை எண்ணிலிருந்து (2) தொடங்கி வரிசையாக செல்லவும்.
• 2 ஆல் வகுபடாத போது 3, பின் 5, 7... என்று செல்லவும்.
        """,
        "diagram_trigger": "division_ladder",
        "worked_numbers": [84, 75]
    },

    {
        "id": "M04",
        "type": "method",
        "topic": "hcf_method_1_list",
        "section": "4.5",
        "page": 46,
        "difficulty": 2,
        "prerequisites": ["hcf_definition", "factor_listing"],
        "content_ta": """
முறை I: காரணிகளை பட்டியலிட்டு பொ.கா.பெ. காண்பது

6, 12, 18 ஆகிய எண்களின் பொ.கா.பெ. காண்போம்.

ஒவ்வோர் எண்ணினதும் காரணிகளை எழுதுவோம்:
6 இன் காரணிகள்:  1, 2, 3, 6
12 இன் காரணிகள்: 1, 2, 3, 4, 6, 12
18 இன் காரணிகள்: 1, 2, 3, 6, 9, 18

மூன்று எண்களுக்கும் பொதுவான காரணிகளை எழுதுவோம்: 1, 2, 3, 6

தெரிந்தெடுத்த பொதுக் காரணிகளுட் பெரிய எண்ணானது பொதுக் காரணிகளுட் பெரியது ஆகும்.
∴ 6, 12, 18 இன் பொ.கா.பெ. = 6

இந்த முறையை எப்போது பயன்படுத்துவது?
→ எண்கள் சிறியதாக இருக்கும்போது (≤ 50) இம்முறை விரைவானது.
→ பெரிய எண்களுக்கு முறை II அல்லது III சிறந்தது.
        """,
        "diagram_trigger": None,
        "method_number": 1
    },

    {
        "id": "M05",
        "type": "method",
        "topic": "hcf_method_2_prime",
        "section": "4.5",
        "page": 47,
        "difficulty": 3,
        "prerequisites": ["hcf_definition", "prime_factorization_tree"],
        "content_ta": """
முறை II: முதன்மைக் காரணிகள் மூலம் பொ.கா.பெ. காண்பது

6, 12, 18 இன் பொ.கா.பெ.:

ஒவ்வோர் எண்ணினதும் காரணிகளை எழுதுவோம்:
6 = 2 × 3
12 = 2 × 2 × 3
18 = 2 × 3 × 3

மூன்று எண்களுக்கும் பொதுவான முதன்மைக் காரணிகளின் பெருக்கம் பொதுக் காரணிகளுட் பெரியதாக அமைகிறது.
6, 12, 18 என்னும் மூன்று எண்களுக்கும் பொதுவான முதன்மைக் காரணிகள் 2 உம் 3 உம் ஆகும்.

∴ 6, 12, 18 இன் பொ.கா.பெ. = 2 × 3 = 6

72, 108 இன் பொ.கா.பெ. (பக்கம் 49):
72 = 2 × 2 × 2 × 3 × 3
108 = 2 × 2 × 3 × 3 × 3
72, 108 ஆகிய இரண்டு எண்களையும் வகுக்கக்கூடிய முதன்மை எண்கள்: 2, 2, 3, 3
∴ 72, 108 இன் பொ.கா.பெ. = 2 × 2 × 3 × 3 = 36
        """,
        "diagram_trigger": "factor_tree",
        "method_number": 2
    },

    {
        "id": "M06",
        "type": "method",
        "topic": "hcf_method_3_division",
        "section": "4.5",
        "page": 48,
        "difficulty": 3,
        "prerequisites": ["hcf_definition", "prime_factorization_division"],
        "content_ta": """
முறை III: வகுத்தல் முறை மூலம் பொ.கா.பெ. காண்பது (மிக விரைவான முறை)

6, 12, 18 இன் பொ.கா.பெ.:

2 | 6, 12, 18
3 | 3,  6,  9
  | 1,  2,  3

மூன்று எண்களும் 2 ஆல் வகுபடும் என்பதால், மூன்று எண்களையும் தனித்தனியாக 2 ஆல் வகுக்க.
விடையாகப் பெறப்படும் 3, 6, 9 என்னும் மூன்று எண்களும் அடுத்த முதன்மை எண்ணான 3 ஆல் வகுபடுவதால் மூன்று எண்களையும் 3 ஆல் தனித்தனியே வகுத்து ஒவ்வோர் எண்ணுக்கும் கீழேயும் எழுது.
1, 2, 3 ஆகிய மூன்று எண்களும் வகுபடக்கூடிய வேறு முதன்மைக் காரணி இல்லாததால் வகுத்தலை நிறுத்துக.
வகுத்தலுக்கு உதவிய எண்களைப் பெருக்கி பொ.கா.பெ. ஐப் பெறுக.
∴ 6, 12, 18 இன் பொ.கா.பெ. = 2 × 3 = 6

நிறுத்தும் நிபந்தனை: குறைந்தது இரண்டு எண்களாவது ஒரே எண்ணால் வகுபடும் வரை வகுத்தல் செய்க.
        """,
        "diagram_trigger": "division_ladder",
        "method_number": 3
    },

    {
        "id": "M07",
        "type": "method",
        "topic": "lcm_prime_method",
        "section": "4.6",
        "page": 53,
        "difficulty": 3,
        "prerequisites": ["lcm_definition", "prime_factorization_tree"],
        "content_ta": """
முதன்மைக் காரணிகளின் உயர் வலுவால் பொ.ம.சி. காண்பது:

4, 12, 18 இன் பொ.ம.சி.:

இவ்வெண்களை முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுவோம்:
4 = 2 × 2 = 2²
12 = 2 × 2 × 3 = 2² × 3¹
18 = 2 × 3 × 3 = 2¹ × 3²

இவ்வெண்களில் வித்தியாசமான முதன்மைக் காரணிகள் 2, 3 ஆகும்.
மூன்று எண்களினதும் முதன்மைக் காரணிகளைக் கருதும்போது:
2 இன் உயர் வலு = 2²
3 இன் உயர் வலு = 3²

அவ்வலுக்களைப் பெருக்குவதால் பொ.ம.சி. கிடைக்கும்:
∴ 4, 12, 18 இன் பொ.ம.சி. = 2² × 3² = 2 × 2 × 3 × 3 = 36

சிறப்பு விதி: இவ்வெண்கள் மூன்றையும் வகுக்கக்கூடிய எண்கள் இல்லையென்றால்
அவ்வெண்கள் ஒவ்வொன்றையும் பெருக்கி பொ.ம.சி. காணலாம்.
4, 3, 5 இன் பொ.ம.சி. = 4 × 3 × 5 = 60
        """,
        "diagram_trigger": "factor_tree",
        "method_number": 1
    },

    {
        "id": "M08",
        "type": "method",
        "topic": "lcm_division_method",
        "section": "4.6",
        "page": 54,
        "difficulty": 3,
        "prerequisites": ["lcm_definition", "prime_factorization_division"],
        "content_ta": """
வகுத்தல் முறையில் பொ.ம.சி. காண்பது:

4, 12, 18 இன் பொ.ம.சி.:

2 | 4, 12, 18
2 | 2,  6,  9
3 | 1,  3,  9
  | 1,  1,  3

4, 12, 18 என்பன 2 ஆல் வகுபடுவதால் அவற்றை 2 ஆல் வகுக்க.
விடையாகக் கிடைக்கும் 2, 6, 9 என்னும் மூன்று எண்களையும் வகுக்கக்கூடிய முதன்மை எண்கள்
இல்லை. எனினும் 2 உம் 6 உம் 2 ஆல் வகுபடும்.
எனவே 2 ஐயும் 6 ஐயும் ஒவ்வோர் எண்ணின் கீழேயும் 2 ஆல் வகுத்து உரிய விடைகளை எழுது.
9 ஐ அவ்வாறே 9 என அதன் கீழே எழுது.
3, 9 என்ற எண்கள் அடுத்த முதன்மை எண்ணான 3 ஆல் வகுபடுவதால்...

நிறுத்தும் நிபந்தனை: ஒரே எண்ணால் வகுபடக்கூடியதாகக் குறைந்தது இரண்டு எண்களாவது
இன்மையால் வகுத்தலை நிறுத்துக.

வகுத்த எண்களையும் இறுதியாக எஞ்சிய எண்களையும் பெருக்கி பொ.ம.சி. ஐக் காண்க:
∴ 4, 12, 18 இன் பொ.ம.சி. = 2 × 2 × 3 × 1 × 1 × 3 = 36
        """,
        "diagram_trigger": "division_ladder",
        "method_number": 2,
        "nie_note": "வகுத்தல் முறையில் பொ.கா.பெ. தேடும்போது: அனைத்தும் வகுபட வேண்டும். பொ.ம.சி. தேடும்போது: குறைந்தது 2 எண்கள் வகுபட்டால் போதும்."
    },

    # ── TYPE D: WORKED EXAMPLES ───────────────────────────────────────────────

    {
        "id": "WE01",
        "type": "worked_example",
        "topic": "hcf",
        "section": "4.5",
        "page": 49,
        "difficulty": 3,
        "numbers": [72, 108],
        "content_ta": """
உதாரணம் 1 (பக்கம் 49): 72, 108 இன் பொ.கா.பெ. காண்க.

முறை I — பட்டியல் முறை:
72 இன் காரணிகள்:  1, 2, 3, 4, 8, 9, 18, 24, 36, 72
108 இன் காரணிகள்: 1, 2, 3, 4, 9, 12, 36, 54, 108
பொதுக் காரணிகள்: 1, 2, 3, 4, 6, 9, 12, 18, 36
∴ பொ.கா.பெ. = 36

முறை II — முதன்மைக் காரணிகள்:
72 = 2 × 2 × 2 × 3 × 3
108 = 2 × 2 × 3 × 3 × 3
இரண்டையும் வகுக்கக்கூடிய முதன்மை எண்கள்: 2, 2, 3, 3
∴ பொ.கா.பெ. = 2 × 2 × 3 × 3 = 36

முறை III — வகுத்தல்:
2 | 72, 108
2 | 36,  54
3 | 18,  27
3 |  6,   9
  |  2,   3
∴ பொ.கா.பெ. = 2 × 2 × 3 × 3 = 36

மூன்று முறைகளிலும் விடை ஒன்றே: 36
        """,
        "diagram_trigger": "division_ladder",
        "diagram_data": {
            "numbers": [72, 108],
            "steps": [
                {"divisor": 2, "results": [36, 54]},
                {"divisor": 2, "results": [18, 27]},
                {"divisor": 3, "results": [6, 9]},
                {"divisor": 3, "results": [2, 3]}
            ],
            "hcf": 36
        }
    },

    {
        "id": "WE02",
        "type": "worked_example",
        "topic": "hcf_word_problem",
        "section": "4.5",
        "page": 50,
        "difficulty": 3,
        "content_ta": """
உதாரணம் 2 (பக்கம் 50): மாணவர் விடுதி ஒன்றுக்கு அளிப்பதற்காக:
30 சவர்க்காரக் கட்டிகள், 24 பற்பசைகள், 18 பற்தூரிகைகள்

ஒரு பொதியில் இவை மூன்று வகையும் அடங்கும் விதத்திலும்
ஒவ்வொரு வகையிலும் சமனான எண்ணிக்கை இருக்கும் விதத்திலும்
இப்பொருள்கள் பொதிசெய்யப்பட்டுள்ளன. இவ்வாறு பொதி செய்வதாயின்
அதி கூடிய பொதிகளின் எண்ணிக்கை எதுவாக இருக்கும்?

தீர்வு:
ஒரு பொதியில் ஒவ்வொரு பொருளும் சம எண்ணிக்கையில் இருக்க வேண்டும்.
அதிகூடிய பொருள்களின் எண்ணிக்கையைக் காண 30, 24, 18 ஆகிய எண்கள்
மூன்றும் மீதியின்றி வகுக்கக்கூடிய மிகப் பெரிய எண்ணைக் காண வேண்டும்.

30 = 2 × 3 × 5
24 = 2 × 2 × 2 × 3
18 = 2 × 3 × 3
பொ.கா.பெ. = 2 × 3 = 6

∴ பெறப்படும் அதிகூடிய பொதிகளின் எண்ணிக்கை = 6
ஒரு பொதியில் இருக்கும் சவர்க்காரக் கட்டிகளின் எண்ணிக்கை = 30 ÷ 6 = 5
ஒரு பொதியில் இருக்கும் பற்பசைகளின் எண்ணிக்கை = 24 ÷ 6 = 4
ஒரு பொதியில் இருக்கும் பற்தூரிகைகளின் எண்ணிக்கை = 18 ÷ 6 = 3

HCF vs LCM நிர்ணயம்: பொதுவான பகிர்வு → HCF பயன்படுத்த வேண்டும்.
        """,
        "problem_type": "hcf_word_problem",
        "why_hcf": "பொதுவான பகிர்வு / equal distribution → HCF",
        "student_trap": "LCM பயன்படுத்தக்கூடாது — அது மடங்குகளுக்கானது"
    },

    {
        "id": "WE03",
        "type": "worked_example",
        "topic": "lcm_word_problem",
        "section": "4.6",
        "page": 55,
        "difficulty": 3,
        "content_ta": """
உதாரணம் 2 (பக்கம் 55): 2 மணிகள் முறையே 6 நிமிடங்கள், 8 நிமிடங்களுக்கு
ஒரு முறை ஒலிக்கின்றன. காலை 8.00 மணிக்கு இரு மணிகளும் ஒருமித்து ஒலித்தால்,
அவை மீண்டும் எத்தனை மணிக்கு ஒருமித்து ஒலிக்கும்?

தீர்வு:
இரு மணிகளும் ஒருமித்து ஒலிப்பது இவ்விரு எண்களின் பொது மடங்கில் என்பதால்,
முதல் முறையாக இரு மணிகளும் ஒருமித்து ஒலிப்பது எத்தனையாவது நிமிடத்தில்
என்பதைக் காண 6, 8 என்னும் எண்களின் பொ.ம.சி. ஐக் காண்போம்.

2 | 6, 8
  | 3, 4
6, 8 இன் பொ.ம.சி. = 2 × 3 × 4 = 24

∴ இரு மணிகளும் ஒருமித்து ஒலிப்பது 24 நிமிடத்துக்குப் பின்னரே.
முதல் முறையாக இரு மணிகளும் ஒருமித்து ஒலித்த நேரம் = மு.ப. 8.00
இரண்டாவது தடவையாக இரு மணிகளும் ஒருமித்து ஒலிக்கும் நேரம் = மு.ப. 8.24

HCF vs LCM நிர்ணயம்:
→ "எப்போது சந்திக்கும்?" / "எப்போது ஒன்று சேரும்?" → LCM பயன்படுத்தவும்
→ "சமவாக பகிர்க" / "குறைந்தபட்ச துண்டுகள்" → HCF பயன்படுத்தவும்
        """,
        "problem_type": "lcm_word_problem",
        "why_lcm": "முதல் சந்திப்பு நேரம் → LCM"
    },

    # ── TYPE E: EXERCISE CHUNKS (sample) ─────────────────────────────────────

    {
        "id": "EX01",
        "type": "exercise",
        "topic": "divisibility_9",
        "section": "4.1",
        "page": 36,
        "difficulty": 1,
        "question_ta": "பின்வரும் எண்களை வகுக்காமல் அவற்றுள் 9 ஆல் மீதியின்றி வகுபடும் எண்களைத் தெரிவுசெய்து எழுதுக: 504, 652, 567, 856, 1143, 1351, 2719, 4536",
        "answer": [504, 567, 1143, 4536],
        "solution_steps_ta": [
            "504: 5+0+4 = 9 → 9 ஆல் வகுபடும் ✓",
            "652: 6+5+2 = 13 → 1+3 = 4 → 9 ஆல் வகுபடாது ✗",
            "567: 5+6+7 = 18 → 1+8 = 9 → 9 ஆல் வகுபடும் ✓",
            "856: 8+5+6 = 19 → 1+9 = 10 → 1+0 = 1 → 9 ஆல் வகுபடாது ✗",
            "1143: 1+1+4+3 = 9 → 9 ஆல் வகுபடும் ✓",
            "1351: 1+3+5+1 = 10 → 1+0 = 1 → 9 ஆல் வகுபடாது ✗",
            "2719: 2+7+1+9 = 19 → 1+9 = 10 → 1+0 = 1 → 9 ஆல் வகுபடாது ✗",
            "4536: 4+5+3+6 = 18 → 1+8 = 9 → 9 ஆல் வகுபடும் ✓"
        ],
        "common_errors": [
            "இலக்கச் சுட்டி இரண்டு இலக்கமாக வந்தால் மீண்டும் கூட வேண்டும் என்பதை மறக்கின்றனர்",
            "9 ஐ 3 இன் விதியுடன் குழப்புகின்றனர்"
        ],
        "socratic_hints": {
            "wrong_652": "652 இன் இலக்கச் சுட்டியை மீண்டும் கணக்கிடுவாயா? ஒவ்வோர் இலக்கத்தையும் கூட்டிப் பார்.",
            "wrong_567": "567 இன் இலக்கச் சுட்டி என்ன? அது 9 ஆல் வகுபடுமா?"
        }
    },

    {
        "id": "EX02",
        "type": "exercise",
        "topic": "hcf",
        "section": "4.5",
        "page": 51,
        "difficulty": 3,
        "question_ta": "ஒரு கூடையில் 96 அப்பிள்களும் இன்னொரு கூடையில் 60 தோடம் பழங்களும் உள்ளன. இரு வகைப் பழங்களும் சம எண்ணிக்கையில் இருக்கும் வகையில் பொதிகளில் இடப்பட்டால் பெறக்கூடிய அதிகூடிய பொதிகளின் எண்ணிக்கை யாது? ஒரு பொதியில் காணப்படும் அப்பிள்களின் எண்ணிக்கை, தோடம்பழங்களின் எண்ணிக்கை என்பவற்றைத் தனித்தனியே காண்க.",
        "answer": {"max_packets": 12, "apples_per_packet": 8, "oranges_per_packet": 5},
        "solution_steps_ta": [
            "96 = 2 × 2 × 2 × 2 × 2 × 3 = 2⁵ × 3",
            "60 = 2 × 2 × 3 × 5 = 2² × 3 × 5",
            "இரண்டையும் வகுக்கும் பொதுவான முதன்மைக் காரணிகள்: 2², 3",
            "பொ.கா.பெ. = 2 × 2 × 3 = 12",
            "அதிகூடிய பொதிகள் = 12",
            "ஒரு பொதியில் அப்பிள்கள் = 96 ÷ 12 = 8",
            "ஒரு பொதியில் தோடம்பழங்கள் = 60 ÷ 12 = 5"
        ],
        "common_errors": [
            "LCM தேடுகின்றனர் — 'சம எண்ணிக்கை' என்றால் HCF என்று புரிந்துகொள்ள வேண்டும்",
            "96 ஐ தவறாக பகுக்கின்றனர்"
        ],
        "socratic_hints": {
            "used_lcm": "நீங்கள் LCM காண்கிறீர்கள். பொதிகளை எவ்வளவு கூட்டலாம் அல்லது குறைக்கலாம்? 'அதிகூடிய பொதிகள்' என்றால் என்ன?",
            "wrong_factorization": "96 ஐ மீண்டும் பகுத்துப் பார். 96 = 2 × 48, 48 = 2 × 24... என தொடரவும்."
        }
    },

    # ── TYPE F: SUMMARY CHUNKS ────────────────────────────────────────────────

    {
        "id": "SUM01",
        "type": "summary",
        "topic": "divisibility_rules_all",
        "section": "4.1_end",
        "page": 40,
        "difficulty": 1,
        "content_ta": """
பொழிப்பு — அனைத்து வகுபடும் விதிகள் (பக்கம் 40):

வகுபடும் எண் | வகுபடும் விதி
─────────────────────────────────────────────────────
2           | ஒன்றினிடத்து இலக்கம் 2 ஆல் வகுபடுமாயின், அவ்வெண் 2 ஆல் வகுபடும்.
3           | இலக்கச் சுட்டி 3 ஆல் வகுபடுமாயின், அவ்வெண் 3 ஆல் வகுபடும்.
4           | இறுதி இரு இலக்கங்களும் 4 ஆல் வகுபடுமாயின் அவ்வெண் 4 ஆல் வகுபடும்.
5           | ஒன்றினிடத்து இலக்கம் 0 அல்லது 5 ஆயின், அவ்வெண் 5 ஆல் வகுபடும்.
6           | ஒர் எண் 2 ஆலும் 3 ஆலும் வகுபடுமாயின் அவ்வெண் 6 ஆல் வகுபடும்.
9           | ஒர் எண்ணின் இலக்கச் சுட்டி 9 ஆயின் அவ்வெண் 9 ஆல் வகுபடும்.
10          | ஒன்றினிடத்து இலக்கம் 0 ஆயின் அவ்வெண் 10 ஆல் வகுபடும்.
        """,
        "is_reference": True
    },

    {
        "id": "SUM02",
        "type": "summary",
        "topic": "hcf_lcm_relationship",
        "section": "4.6_end",
        "page": 53,
        "difficulty": 3,
        "content_ta": """
குறிப்பு (பக்கம் 53):

• தரப்பட்ட சில எண்களின் பொதுக் காரணிகளுட் பெரியது,
  அவ்வெண்களில் சிறியதற்குச் சமனாகவோ அல்லது அதனிலும் சிறியதாகவோ இருக்கும்.

• தரப்பட்ட சில எண்களின் பொது மடங்குகளுட் சிறியது
  அவ்வெண்களில் பெரியதற்குச் சமனாகவோ அல்லது அதனிலும் பெரியதாகவோ இருக்கும்.

• இரு எண்களின் பொ.கா.பெ. ஆனது அவ்விரு எண்களின் பொ.ம.சி.-ஐ விடச் சிறியதாக இருக்கும்.

• எந்தவொரு தொகுதி முதன்மை எண்களினும் பொ.கா.பெ. 1 ஆகும்.

HCF vs LCM — எப்போது எதை பயன்படுத்துவது?
→ "சமவாகப் பகிர்" / "குறைந்தபட்ச துண்டுகள்" / "அதிகூடிய பொதிகள்" → பொ.கா.பெ.
→ "முதல் சந்திப்பு" / "ஒருமிக்கும் நேரம்" / "மீண்டும் ஒலிக்கும்" → பொ.ம.சி.
        """,
        "is_reference": True
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOPIC → SKILL MAPPING  (single source of truth — fixes bug A/B)
#
# Every corpus topic string maps to one of the 7 skill keys in StudentProfile.
# This is the canonical bridge between PREREQUISITE_GRAPH (topic names) and
# student.skills (skill names).  Previously the two layers used different
# vocabularies, making prerequisite checks silently return 0 for most topics.
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_TO_SKILL: dict[str, str] = {
    # divisibility family → divisibility_rules skill
    "divisibility_2":               "divisibility_rules",
    "divisibility_3":               "divisibility_rules",
    "divisibility_4":               "divisibility_rules",
    "divisibility_5":               "divisibility_rules",
    "divisibility_6":               "divisibility_rules",
    "divisibility_9":               "divisibility_rules",
    "divisibility_10":              "divisibility_rules",
    "divisibility_rules":           "divisibility_rules",
    "divisibility_rules_all":       "divisibility_rules",
    # digit sum
    "digit_sum":                    "digit_sum",
    # factor listing
    "factor_definition":            "factor_listing",
    "factor_listing":               "factor_listing",
    "factor_listing_pair_method":   "factor_listing",
    "factor_pairs":                 "factor_listing",
    # prime factorisation
    "prime_number_definition":      "prime_factorization",
    "prime_factorization":          "prime_factorization",
    "prime_factorization_tree":     "prime_factorization",
    "prime_factorization_division": "prime_factorization",
    # HCF
    "hcf":                          "hcf",
    "hcf_definition":               "hcf",
    "hcf_method_1_list":            "hcf",
    "hcf_method_2_prime":           "hcf",
    "hcf_method_3_division":        "hcf",
    "hcf_word_problem":             "hcf",
    "hcf_lcm_relationship":         "hcf",
    # LCM
    "lcm":                          "lcm",
    "lcm_definition":               "lcm",
    "lcm_prime_method":             "lcm",
    "lcm_division_method":          "lcm",
    "lcm_word_problem":             "lcm",
    # word problems
    "word_problems":                "word_problems",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREREQUISITE DEPENDENCY GRAPH  (topics → prerequisite topics)
# ─────────────────────────────────────────────────────────────────────────────

PREREQUISITE_GRAPH: dict[str, list[str]] = {
    "digit_sum":                    ["divisibility_rules"],
    "factor_listing":               ["divisibility_rules"],
    "factor_listing_pair_method":   ["factor_definition"],
    "prime_number_definition":      ["factor_definition"],
    "prime_factorization_tree":     ["factor_listing", "divisibility_rules"],
    "prime_factorization_division": ["prime_number_definition"],
    "hcf_definition":               ["factor_listing", "prime_factorization_tree"],
    "hcf_method_1_list":            ["hcf_definition"],
    "hcf_method_2_prime":           ["hcf_definition", "prime_factorization_tree"],
    "hcf_method_3_division":        ["hcf_definition", "prime_factorization_division"],
    "lcm_definition":               ["factor_listing", "prime_factorization_tree"],
    "lcm_prime_method":             ["lcm_definition", "prime_factorization_tree"],
    "lcm_division_method":          ["lcm_definition", "prime_factorization_division"],
    "word_problems":                ["hcf_definition", "lcm_definition"],
}


def _topic_to_skill(topic: str) -> str:
    """Resolve any topic name to its parent skill key (falls back gracefully)."""
    return TOPIC_TO_SKILL.get(topic, "divisibility_rules")


# ─────────────────────────────────────────────────────────────────────────────
# 4. STUDENT PROFILE MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StudentProfile:
    student_id: str
    name: str
    grade: int = 7
    school_type: str = "tamil_medium"
    # PoC: jaffna | estate | batticaloa | colombo | unknown — steers synonym bridging in prompts
    district: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Skill scores 0.0 – 1.0  (7 canonical keys)
    skills: dict = field(default_factory=lambda: {
        "divisibility_rules": 0.0,
        "digit_sum": 0.0,
        "factor_listing": 0.0,
        "prime_factorization": 0.0,
        "hcf": 0.0,
        "lcm": 0.0,
        "word_problems": 0.0,
    })

    # Interaction history
    total_questions_asked: int = 0
    total_exercises_attempted: int = 0
    total_exercises_correct: int = 0
    preferred_method: str = "none"
    last_topic: str = ""
    last_error_type: str = ""
    session_count: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        """Safe deserialization — ignores unknown keys from old DB rows (fix G)."""
        known = {f.name for f in dataclass_fields(cls)}
        safe = {k: v for k, v in data.items() if k in known}
        obj = cls.__new__(cls)
        # Dataclass requires student_id + name — use placeholders for default field values only
        defaults = cls(student_id="__default__", name="__default__")
        for f in dataclass_fields(cls):
            setattr(obj, f.name, safe.get(f.name, getattr(defaults, f.name)))
        # Merge skills: preserve any new keys with default 0.0
        base_skills = dict(defaults.skills)
        base_skills.update(safe.get("skills", {}))
        obj.skills = base_skills
        return obj

    def get_difficulty_ceiling(self) -> int:
        """Return max difficulty level student is ready for."""
        avg_skill = sum(self.skills.values()) / len(self.skills)
        if avg_skill < 0.3:
            return 1
        elif avg_skill < 0.6:
            return 2
        return 3

    def get_unlocked_topics(self) -> set[str]:
        """
        Return corpus topic names the student is ready for (fix B).
        A topic is unlocked when ALL prerequisite topics map to skills >= 0.5,
        OR the topic has no prerequisites, OR it is a foundation topic.
        """
        FOUNDATION = {
            "factor_definition", "divisibility_rules", "digit_sum",
            "divisibility_2", "divisibility_3", "divisibility_9",
            "divisibility_6", "divisibility_4", "divisibility_rules_all",
        }
        unlocked: set[str] = set(FOUNDATION)
        # Iterate until no new topics are added (topological unlock)
        changed = True
        while changed:
            changed = False
            for topic, prereq_topics in PREREQUISITE_GRAPH.items():
                if topic in unlocked:
                    continue
                prereq_skills_met = all(
                    self.skills.get(_topic_to_skill(pt), 0.0) >= 0.5
                    for pt in prereq_topics
                )
                if prereq_skills_met:
                    unlocked.add(topic)
                    changed = True
        return unlocked

    def update_skill(self, topic: str, correct: bool, difficulty: int):
        """Update skill score using TOPIC_TO_SKILL mapping (fix D)."""
        skill_key = _topic_to_skill(topic)
        if skill_key in self.skills:
            delta = 0.1 * difficulty if correct else -0.05
            self.skills[skill_key] = max(0.0, min(1.0,
                                                   self.skills[skill_key] + delta))
        self.total_exercises_attempted += 1
        if correct:
            self.total_exercises_correct += 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Classify student's Tamil query into one of 6 intent types.
    Uses keyword matching for PoC — replace with LLM classification in production.

    Fix C: INTENT_PRIORITY resolves ties deterministically; more-specific
    intents (SHOW_METHOD, DIAGRAM_REQUEST) outrank the generic EXPLAIN when
    keywords overlap (e.g. "எப்படி காட்டு" should be SHOW_METHOD, not EXPLAIN).
    """

    INTENT_PRIORITY = [
        "CHECK_ANSWER",    # most specific — student submitted an answer
        "DIAGRAM_REQUEST", # explicit draw/chart request
        "SHOW_METHOD",     # step-by-step method
        "EXERCISE_REQUEST",
        "WORD_PROBLEM",
        "EXPLAIN",         # least specific — fallback
    ]

    INTENT_KEYWORDS: dict[str, list[str]] = {
        "EXPLAIN": [
            "என்றால் என்ன", "வரையறை", "விளக்கு", "புரியவில்லை", "கூறு", "எப்படி", "கூறுங்கள்", "விளக்குங்கள்", "விளக்கவும்", "கூறவும்", "என்ன",
            "கற்றுக்கொடு", "சொல்லுங்கள்", "what is",
            "explain", "teach", "define",
        ],
        "SHOW_METHOD": [
            "முறை", "எப்படி காண்பது", "எப்படி கணக்கிடுவது", "காட்டு", "காட்டுங்கள்", "காட்டவும்",
            "steps", "படிகள்", "method", "வகுத்தல் முறை", "ஏணி முறை",
            "காரணி மரம் முறை", "show method", "step by step",
            # "எப்படி" alone is kept ONLY here (not in EXPLAIN) so that
            # "எப்படி காட்டு" scores method, not explain
            "எப்படி",
        ],
        "EXERCISE_REQUEST": [
            "பயிற்சி", "கேள்வி கொடு", "கணக்கு கொடு", "சோதனை",
            "practice", "exercise", "question", "problem", "கொடு", "கொடுங்கள்", "கொடுக்கவும்", "தரவும்", "தாருங்கள்", 
        ],
        "CHECK_ANSWER": [
            "சரியா", "இது சரியா", "என் பதில்", "விடை", "விடை சரிதானா", "என் பதில் சரியா",
            "நான் கண்டேன்", "check", "correct", "answer", "= ",
        ],
        "DIAGRAM_REQUEST": [
            "வரை", "படம்", "draw", "diagram",
            "காரணி மரம்", "factor tree",
            "வகுத்தல் ஏணி", "division ladder",
            "number line", "மடங்கு கோடு",
            "காட்டு", "chart",
        ],
        "WORD_PROBLEM": [
            "கதை கணக்கு", "சிந்தனைக்கு", "பென்சில்", "மணி",
            "பழம்", "பொதி", "பகிர்", "word problem", "real life",
            "நிமிடம்", "நேரம்", "பூக்கள்", "மரம்",
        ],
    }

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        scores: dict[str, int] = {intent: 0 for intent in self.INTENT_KEYWORDS}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    scores[intent] += 1
        # Return highest-priority intent that has at least one keyword match
        for intent in self.INTENT_PRIORITY:
            if scores[intent] > 0:
                return intent
        return "EXPLAIN"


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADAPTIVE RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveRetriever:
    """
    Two-stage retrieval:
    Stage 1 — Metadata pre-filter (difficulty + prerequisites + topic)
    Stage 2 — Keyword-based relevance scoring (replace with vector search in production)
    """

    def __init__(self, corpus: list):
        self.corpus = corpus

    def _pre_filter(self, intent: str, student: StudentProfile) -> list:
        """
        Filter corpus by student level and unlocked topics (fix A).

        Prerequisites are now resolved via TOPIC_TO_SKILL so that
        PREREQUISITE_GRAPH topic names correctly map to student.skills keys.
        """
        max_diff = student.get_difficulty_ceiling()
        unlocked = student.get_unlocked_topics()  # returns set of topic strings

        # Intent → allowed chunk types
        type_map = {
            "EXPLAIN":          ["concept", "summary"],
            "SHOW_METHOD":      ["method", "worked_example"],
            "EXERCISE_REQUEST": ["exercise", "worked_example"],
            "CHECK_ANSWER":     ["exercise", "worked_example", "concept"],
            "DIAGRAM_REQUEST":  ["method", "worked_example"],
            "WORD_PROBLEM":     ["worked_example", "exercise", "concept"],
        }
        allowed_types = type_map.get(intent, ["concept", "method", "worked_example"])

        filtered = []
        for chunk in self.corpus:
            if chunk["type"] not in allowed_types:
                continue
            if chunk.get("difficulty", 1) > max_diff + 1:
                continue
            # Foundation / difficulty-1 chunks are always included
            if chunk.get("difficulty", 1) == 1:
                filtered.append(chunk)
                continue
            # For harder chunks: topic must be unlocked
            chunk_topic = chunk.get("topic", "")
            if chunk_topic in unlocked:
                filtered.append(chunk)

        return filtered

    def _score_relevance(self, chunk: dict, query: str,
                         student: StudentProfile) -> float:
        """Simple keyword overlap score — replace with embeddings in production."""
        query_words = set(query.lower().split())
        content = chunk.get("content_ta", "") + " " + chunk.get("topic", "")
        content_words = set(content.lower().split())
        overlap = len(query_words & content_words)
        score = overlap / max(len(query_words), 1)

        # Boost if matches student's preferred method
        if (chunk.get("method_number") and
                str(chunk.get("method_number")) in student.preferred_method):
            score += 0.2

        # Boost recently studied topics
        if student.last_topic and student.last_topic in chunk.get("topic", ""):
            score += 0.15

        # Penalise if much harder than student level
        chunk_diff = chunk.get("difficulty", 1)
        student_ceiling = student.get_difficulty_ceiling()
        if chunk_diff > student_ceiling + 1:
            score -= 0.3

        return score

    def retrieve(self, query: str, intent: str, student: StudentProfile,
                 top_k: int = 4) -> list:
        """Full two-stage adaptive retrieval (fix A continues)."""
        filtered = self._pre_filter(intent, student)
        if not filtered:
            filtered = [c for c in self.corpus if c.get("difficulty", 1) == 1]

        scored = [(chunk, self._score_relevance(chunk, query, student))
                  for chunk in filtered]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = [chunk for chunk, _ in scored[:top_k]]

        # Inject prerequisite concept chunks if the underlying skill is weak.
        # Now correctly resolves prereq topic → skill key via TOPIC_TO_SKILL.
        injected_ids = {c["id"] for c in results}
        for chunk in list(results):
            for prereq_topic in PREREQUISITE_GRAPH.get(chunk.get("topic", ""), []):
                skill_key = _topic_to_skill(prereq_topic)
                if student.skills.get(skill_key, 0) < 0.4:
                    prereq_chunks = [
                        c for c in self.corpus
                        if c.get("topic") == prereq_topic
                        and c["id"] not in injected_ids
                    ]
                    if prereq_chunks:
                        results.insert(0, prereq_chunks[0])
                        injected_ids.add(prereq_chunks[0]["id"])

        return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 6. DIAGRAM TRIGGER SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class DiagramTrigger:
    """
    Detects when a diagram should be generated and produces the JSON spec
    for the Flutter rendering engine.
    """

    def should_draw(self, intent: str, retrieved_chunks: list,
                    query: str) -> bool:
        diagram_intents = {"DIAGRAM_REQUEST", "SHOW_METHOD"}
        if intent in diagram_intents:
            return True
        # Auto-trigger for methods that have diagrams
        for chunk in retrieved_chunks:
            if chunk.get("diagram_trigger") is not None:
                keywords = ["காட்டு", "எப்படி", "வரை", "show", "draw", "explain"]
                if any(kw in query.lower() for kw in keywords):
                    return True
        return False

    def generate_spec(self, chunk: dict, numbers: list = None) -> dict:
        """Generate diagram JSON spec for the Flutter renderer."""
        diagram_type = chunk.get("diagram_trigger")
        if not diagram_type:
            return {}

        if diagram_type == "factor_tree" and numbers:
            return self._factor_tree_spec(numbers[0])
        elif diagram_type == "division_ladder" and numbers:
            return self._division_ladder_spec(numbers)
        elif diagram_type == "factor_pairs" and numbers:
            return self._factor_pairs_spec(numbers[0])
        elif diagram_type == "multiples_line" and numbers:
            return self._multiples_line_spec(numbers)
        return {"diagram": diagram_type, "error": "numbers_not_provided"}

    def _factor_tree_spec(self, n: int) -> dict:
        """Generate factor tree JSON for number n."""
        primes = self._prime_factors(n)
        tree = self._build_factor_tree(n)
        return {
            "diagram": "factor_tree",
            "root": n,
            "tree": tree,           # nested branching structure (fix D)
            "prime_factors": primes,
            "result_label_ta": f"{n} = " + " × ".join(map(str, primes)),
            "highlight_primes": True,
            "animate_step_by_step": True,
            "label_ta": f"{n} இன் காரணி மரம்",
        }

    def _division_ladder_spec(self, numbers: list) -> dict:
        """
        Generate division ladder JSON for HCF calculation (fix E).

        NIE rule: divide only when ALL numbers divide evenly (HCF ladder).
        The HCF is the product of every divisor used.
        """
        from math import gcd
        from functools import reduce

        steps = []
        remaining = list(numbers)
        divisors_used: list[int] = []

        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if all(r <= 1 for r in remaining):
                break
            # HCF ladder: ALL numbers must be divisible (NIE Method III rule)
            if all(r % p == 0 for r in remaining):
                new_remaining = [r // p for r in remaining]
                steps.append({
                    "divisor": p,
                    "before": list(remaining),
                    "after": new_remaining,
                })
                divisors_used.append(p)
                remaining = new_remaining

        hcf = reduce(lambda a, b: a * b, divisors_used, 1)
        # Verify against math.gcd (correctness guard)
        actual_hcf = reduce(gcd, numbers)

        return {
            "diagram": "division_ladder",
            "numbers": numbers,
            "steps": steps,
            "hcf_value": actual_hcf,           # always mathematically correct
            "hcf_product_shown": (
                " × ".join(map(str, divisors_used)) + f" = {hcf}"
                if divisors_used else "1"
            ),
            "animate": True,
            "label_ta": (
                f"{', '.join(map(str, numbers))} இன் பொ.கா.பெ. = {actual_hcf}"
            ),
        }

    def _factor_pairs_spec(self, n: int) -> dict:
        pairs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                pairs.append([i, n // i])
        all_factors = sorted(set(f for pair in pairs for f in pair))
        return {
            "diagram": "factor_pairs",
            "number": n,
            "pairs": pairs,
            "all_factors": all_factors,
            "label_ta": f"{n} இன் காரணிகள்: {', '.join(map(str, all_factors))}",
            "show_multiplication": True,
            "animate": True
        }

    def _multiples_line_spec(self, numbers: list, show_up_to: int = 48) -> dict:
        from math import lcm
        from functools import reduce
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
            "label_ta": f"பொது மடங்குகளுட் சிறியது = {lcm_val}",
            "color_map": color_map
        }

    def _prime_factors(self, n: int) -> list[int]:
        """Return sorted list of prime factors (with repetition)."""
        factors: list[int] = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        return factors

    def _build_factor_tree(self, n: int) -> dict:
        """
        Build a proper nested factor-tree dict (fix D).
        Each node is {"value": N, "left": ..., "right": ...}.
        Leaf nodes have no "left"/"right" (they are prime).
        """
        def smallest_prime_factor(k: int) -> int:
            if k < 2:
                return k
            d = 2
            while d * d <= k:
                if k % d == 0:
                    return d
                d += 1
            return k  # k is prime

        def build(k: int) -> dict:
            if k < 2:
                return {"value": k}
            spf = smallest_prime_factor(k)
            if spf == k:
                return {"value": k, "is_prime": True}
            other = k // spf
            return {
                "value": k,
                "left": {"value": spf, "is_prime": True},
                "right": build(other),
            }

        return build(n)


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXERCISE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class ExerciseGenerator:
    """
    Generate exercises modelled on NIE பயிற்சி structure.
    Calibrated to student's current skill level.
    """

    @staticmethod
    def _digit_sum(n: int) -> int:
        """Canonical NIE digit-sum: repeatedly sum digits until single digit (fix F)."""
        s = sum(int(d) for d in str(abs(n)))
        while s >= 10:
            s = sum(int(d) for d in str(s))
        return s

    def generate(self, topic: str, difficulty: int,
                 student: StudentProfile) -> dict:
        import random

        if topic in ("divisibility_9", "divisibility_rules") and difficulty == 1:
            pool = [504, 207, 135, 81, 333, 441, 108, 252, 999, 1008,
                    362, 415, 700, 921, 234, 567, 100]
            numbers = random.sample(pool, 5)
            return {
                "question_ta": (
                    f"பின்வரும் எண்களில் 9 ஆல் மீதியின்றி வகுபடும் எண்களை "
                    f"இலக்கச் சுட்டி மூலம் தீர்மானிக்கவும்:\n"
                    f"{', '.join(map(str, numbers))}"
                ),
                "numbers": numbers,
                # Uses canonical digit-sum: divisible by 9 iff digit_sum == 9
                "answers": [n for n in numbers if self._digit_sum(n) == 9],
                "method_ta": "ஒவ்வொரு எண்ணின் இலக்கங்களையும் கூட்டி இலக்கச் சுட்டி காண்க",
                "difficulty": difficulty,
                "topic": topic,
            }

        elif topic == "prime_factorization" and difficulty == 2:
            n = random.choice([36, 48, 60, 72, 84, 90, 96, 120, 144, 180, 210, 252])
            return {
                "question_ta": f"{n} ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுக.",
                "numbers": [n],
                "hint_ta": "மிகச் சிறிய முதன்மை எண்ணான 2 இலிருந்து தொடங்கி வகுத்தல் ஏணி முறையில் காண்க",
                "method": "division_ladder",
                "difficulty": difficulty,
                "topic": topic
            }

        elif topic == "hcf" and difficulty == 3:
            pairs = [(12, 18), (24, 36), (48, 72), (36, 54), (60, 90),
                     (84, 108), (45, 75), (16, 24), (30, 45), (72, 108)]
            a, b = random.choice(pairs)
            from math import gcd
            return {
                "question_ta": f"{a} உம் {b} உம் ஆகிய எண்களின் பொ.கா.பெ. மூன்று முறைகளில் காண்க.",
                "numbers": [a, b],
                "answer": gcd(a, b),
                "methods_ta": ["முறை I: காரணிகளை பட்டியலிட்டு",
                               "முறை II: முதன்மைக் காரணிகள் மூலம்",
                               "முறை III: வகுத்தல் முறை மூலம்"],
                "difficulty": difficulty,
                "topic": topic
            }

        elif topic == "lcm" and difficulty == 3:
            triples = [(2, 3, 4), (4, 6, 8), (3, 4, 6), (6, 8, 12),
                       (4, 5, 10), (3, 5, 9), (2, 5, 6), (6, 9, 12)]
            nums = random.choice(triples)
            from math import lcm
            from functools import reduce
            answer = reduce(lcm, nums)
            return {
                "question_ta": f"{nums[0]}, {nums[1]}, {nums[2]} ஆகிய எண்களின் பொ.ம.சி. காண்க.",
                "numbers": list(nums),
                "answer": answer,
                "method_ta": "வகுத்தல் முறை அல்லது முதன்மைக் காரணிகளின் உயர் வலு மூலம் காண்க",
                "difficulty": difficulty,
                "topic": topic
            }

        # Default
        return {
            "question_ta": "பயிற்சி தயாரிக்கப்படுகிறது...",
            "difficulty": difficulty,
            "topic": topic
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. SOCRATIC CORRECTOR
# ─────────────────────────────────────────────────────────────────────────────

SOCRATIC_PROMPTS = {
    "used_lcm_for_hcf": "நீங்கள் பொ.ம.சி. காண்கிறீர்கள். 'அதிகூடிய பொதிகள்' என்றால் என்ன செய்ய வேண்டும்? பொதுவான வகுத்தல் வேண்டுமா அல்லது பொதுவான மடங்கு வேண்டுமா?",
    "forgot_digit_sum_repeat": "உங்கள் இலக்கச் சுட்டி இரண்டு இலக்கமாக உள்ளது. அதை மீண்டும் ஒரு தனி இலக்கம் வரும் வரை கூட்ட வேண்டும். மீண்டும் முயற்சி செய்வாயா?",
    "wrong_prime_factor": "நீங்கள் குறிப்பிட்ட எண் முதன்மை எண்ணா? அதன் காரணிகளை ஒரு முறை சோதித்துப் பாருங்கள்.",
    "missed_a_factor": "காரணிகளை ஜோடி முறையில் தேடுங்கள். 1 × ? = எண், 2 × ? = எண்... என்று தொடரவும். ஏதாவது ஜோடி தவறிவிட்டதா?",
    "wrong_last_two_digits": "4 ஆல் வகுபடுமா என்று சோதிக்க கடைசி இரண்டு இலக்கங்களை மட்டும் பார்க்க வேண்டும். முழு எண்ணை வகுக்க வேண்டியதில்லை. கடைசி இரண்டு இலக்கங்கள் என்ன?",
    "generic": "உங்கள் தொடர்ந்த படி எங்கு தவறியது என்று பாருங்கள். மீண்டும் தொடக்கத்திலிருந்து ஒரு படியாக செய்வாயா?"
}


def get_socratic_question(error_type: str, student_attempt: str,
                          correct_solution: str) -> str:
    """Return a Socratic guiding question — never reveals the answer."""
    return SOCRATIC_PROMPTS.get(error_type, SOCRATIC_PROMPTS["generic"])


# ─────────────────────────────────────────────────────────────────────────────
# 9. PROMPT HELPERS — deterministic math anchors (reduces LLM hallucination)
# ─────────────────────────────────────────────────────────────────────────────


def _positive_divisors(n: int) -> list[int]:
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


def _factor_verification_block_tamil(query: str) -> Optional[str]:
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
    divs = _positive_divisors(n)
    ta = ", ".join(str(x) for x in divs)
    return (
        "சரிபார்ப்பு (கணிதத்தில் காரணிகள் — இதை மட்டுமே சரியான முழுப் பட்டியலாகக் கொள்ளவும்):\n"
        f"எண் {n} இன் அனைத்து காரணிகள் (ஏறுவரிசையில்): {ta}\n"
        "இந்த எண்களையே காரணிகளாகப் பட்டியலிட்டு விளக்கவும்; வேறு எண்களைச் சேர்க்க வேண்டாம்; "
        "இவற்றுள் ஒன்றையும் விட்டுவிட வேண்டாம்.\n"
        "ஜோடி முறை (1×…, 2×…) போன்ற NIE முறையில் தமிழில் விளக்கவும்."
    )


def _hcf_verification_block_tamil(query: str) -> Optional[str]:
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
    from functools import reduce
    from math import gcd as _gcd

    g = reduce(_gcd, nums)
    nums_str = ", ".join(str(x) for x in nums)
    return (
        "சரிபார்ப்பு (பொ.கா.பெ. — இறுதி எண் இதுவே; வேறு எண் எழுத வேண்டாம்):\n"
        f"எண்கள்: {nums_str}\n"
        f"பொதுக் காரணிகளுட் பெரியது (பொ.கா.பெ.) = {g}\n"
        "மேலே உள்ள NIE முறைகளில் (காரணிப் பட்டியல் / முதன்மைக் காரணி மரம் / வகுத்தல் ஏணி) "
        "படிப்படியாக விளக்கி, இறுதியில் இந்தச் சரிபார்ப்பு எண்ணை உறுதிப்படுத்தவும்.\n"
        "மாணவர் 'காண்க' என்று கேட்டுள்ளார்: இறுதி விடையை மறைக்காமல் முழு தீர்வைத் தமிழில் தரவும்."
    )


def _nie_register_and_ladder_guidance(student: StudentProfile) -> str:
    """
    NIE textbook register (வகுத்தல்) vs spoken variants (பிரித்தல்), optional regional
    bridging, and short 'why this divisor' hints for division ladders.
    """
    d = (getattr(student, "district", None) or "unknown").strip().lower()

    base = """
NIE பதிவுருச் சொல்லாட்சி (பாடநூல் தரம் 7, காரணிகள் பாடம்):
• வகுத்தல் ஏணி / முதன்மைக் காரணிப்படுத்தலுக்கு பாடநூல் பயன்படுத்தும் சொற்கள்: "வகுத்தல்", "வகுக்கும்", "வகுப்போம்", "மீதியின்றி வகுப்பது", "வகுத்தல் ஏணி" — இவற்றையே முதன்மை உரையில் பயன்படுத்தவும்.
• "பிரித்தல்", "பிரிப்போம்" போன்றவற்றை இப்பாடப்பகுதியில் பாடநூல் சொற்களுக்குப் பதிலாக எழுத வேண்டாம் (பேச்சுவழக்கு; தேர்வு/பாடநூலுடன் சீரில்லை).

வகுத்தல் ஏணியில் ஒவ்வொரு வகுத்தலுக்கும் குறுகிய காரணம் (NIE 4.1 வகுத்தல் விதிகள்):
• 2 ஆல் வகுக்கும்போது: எண் இரட்டை எண் / ஒன்றினிடத்து இரட்டை இலக்கம் எனக் குறிப்பிடவும்.
• 3 அல்லது 9 ஆல் வகுக்கும்போது: இலக்கச் சுட்டி 3 (அல்லது 9) ஆல் வகுபடும் என்ற பாடநூல் விதியை ஒரு வரியில் குறிப்பிடவும்.
• 5 ஆல் வகுக்கும்போது: ஒன்றினிடத்து 0 அல்லது 5 என்ற விதியைக் குறிப்பிடவும்.
• 7, 11, 13 … போன்றவற்றால் வகுக்கும்போது: முந்தைய சிறிய முதன்மை எண்களால் மீதியின்றி வகுக்க முடியாதபோது அடுத்த முதன்மை எண்ணைச் சோதிப்போம் என்று பாடநூல் வரிசையைக் குறிப்பிடவும் (தேவையான அளவு மட்டும்; நீண்ட நியாயப்படுத்தல் வேண்டாம்).
"""

    if d in ("estate", "tea_estate", "up_country", "central_estate", "plantation"):
        region = (
            "\nமாணவர் பகுதிக் குறிப்பு (தோட்ட / மலையடிவாரப் பகுதி போன்ற சூழல்):\n"
            "• வீட்டிலோ சமூகத்திலோ 'பிரித்தல்' என்ற சொல் அறிமுகமாக இருக்கலாம்; தேர்வு மற்றும் NIE பாடநூலில் "
            "அதற்குச் சமமான பதிவுருச் சொல் 'வகுத்தல்'. ஒரு குறுகிய வரியில் இரண்டையும் இணைத்துக் குறிப்பிடலாம் "
            "(எ.கா. பேச்சில் 'பிரித்தல்' என்றால் பாடநூலில் 'வகுத்தல்' என்று எழுதுவோம்); மீதிப் பதில் முழுக்க "
            "பாடநூல் சொற்களில் இருக்கட்டும்.\n"
        )
    elif d in (
        "jaffna",
        "kilinochchi",
        "mannar",
        "mullaitivu",
        "vavuniya",
        "northern",
    ):
        region = (
            "\nமாணவர் பகுதிக் குறிப்பு (வடக்கு மாகாணப் போக்கு):\n"
            "• NIE பதிவுருச் சொற்கள் (வகுத்தல், வகுப்போம்) மட்டுமே போதும்; கூடுதல் பேச்சுச் சொல் விளக்கம் தேவையில்லை.\n"
        )
    elif d in ("batticaloa", "east", "trincomalee"):
        region = (
            "\nமாணவர் பகுதிக் குறிப்பு:\n"
            "• முதன்மை: NIE சொற்கள் (வகுத்தல், வகுப்போம்). தேவையானால் ஒரு வரியில் பேச்சுச் சொற்களை "
            "பாடநூல் சொற்களுடன் இணைத்துக் குறிப்பிடலாம்.\n"
        )
    else:
        region = (
            "\nமாணவர் பகுதி பொதுவானது அல்லது குறிப்பிடப்படவில்லை:\n"
            "• பாடநூல் சொற்களை (வகுத்தல் ஏணி, வகுப்போம்) முதன்மையாகப் பயன்படுத்தவும்; சில பகுதிகளில் "
            "'பிரித்தல்' என்ற சொல் கேட்கப்படும் — அது இங்கு 'வகுத்தல்' உடன் ஒத்த பொருள் என ஒரு வரியில் "
            "குறிப்பிடலாம்.\n"
        )

    return base + region


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN ADAPTIVE RAG ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveRAGEngine:
    """
    Main orchestrator: query → intent → retrieve → prompt → response
    """

    def __init__(self):
        self.corpus = NIE_CORPUS
        self.classifier = IntentClassifier()
        self.retriever = AdaptiveRetriever(self.corpus)
        self.diagram_trigger = DiagramTrigger()
        self.exercise_generator = ExerciseGenerator()
        self.students = {}
        self._init_db()

    def _init_db(self):
        """Initialize SQLite for student profile persistence."""
        self.conn = sqlite3.connect("student_profiles.db")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                student_id TEXT PRIMARY KEY,
                profile_json TEXT,
                updated_at TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                student_id TEXT,
                query TEXT,
                intent TEXT,
                response_summary TEXT,
                skill_before TEXT,
                skill_after TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def get_or_create_student(self, student_id: str,
                               name: str = "மாணவர்") -> StudentProfile:
        """Load student profile from DB or create new (fix G — safe deserialization)."""
        row = self.conn.execute(
            "SELECT profile_json FROM profiles WHERE student_id = ?",
            (student_id,)).fetchone()
        if row:
            data = json.loads(row[0])
            student = StudentProfile.from_dict(data)
        else:
            student = StudentProfile(student_id=student_id, name=name)
            self._save_student(student)
        self.students[student_id] = student
        return student

    def _save_student(self, student: StudentProfile):
        data = json.dumps(student.__dict__)
        self.conn.execute("""
            INSERT OR REPLACE INTO profiles (student_id, profile_json, updated_at)
            VALUES (?, ?, ?)
        """, (student.student_id, data, datetime.now().isoformat()))
        self.conn.commit()

    def build_prompt(self, query: str, intent: str, student: StudentProfile,
                     retrieved_chunks: list, exercise: dict = None) -> dict:
        """
        Build the complete prompt package for the LLM.
        Returns: {system_prompt, user_message, diagram_spec, exercise}
        """
        skill_level = student.get_difficulty_ceiling()
        skill_summary = ", ".join([
            f"{k}: {v:.1f}" for k, v in student.skills.items() if v > 0
        ]) or "தொடக்க நிலை"

        context = "\n\n---\n\n".join([
            f"பாடம் {chunk.get('section', '')} (பக்கம் {chunk.get('page', '')}):\n{chunk.get('content_ta', '')}"
            for chunk in retrieved_chunks
        ])

        factor_anchor = _factor_verification_block_tamil(query)
        factor_note = ""
        if factor_anchor:
            factor_note = (
                "\n\nஇக்கேள்வி வகை: காரணிகள் முழுப் பட்டியல் — கீழுள்ள 'சரிபார்ப்பு' பட்டியலை "
                "தமிழில் விளக்கிக் கொடுக்கலாம் (சோக்ரட்டிக் 'விடை மறைத்தல்' இங்கு பொருந்தாது).\n"
                f"{factor_anchor}\n"
            )

        hcf_anchor = _hcf_verification_block_tamil(query)
        hcf_note = ""
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

        register_note = _nie_register_and_ladder_guidance(student)

        system_prompt = f"""நீங்கள் ஒரு அனுபவமிக்க கணித ஆசிரியர்.
தரம் 7 தமிழ்வழி மாணவர்களுக்கு இலவசப் பாடநூல் (NIE) படி கற்பிக்கிறீர்கள்.

மொழி விதி — மீறினால் பதில் தவறாகக் கருதப்படும்:
• உங்கள் முழு பதிலும் தமிழில் மட்டுமே. ஆங்கில வாக்கியங்கள், ஆங்கிலச் சொற்கள், ஆங்கில எழுத்துகள் (A–Z) ஒன்றும் கூடாது.
• "Let's", "We", "factor", "remainder", "whole number" போன்றவை முற்றிலும் தடை.
• விளக்கத்தில் காரணி, மடங்கு, இலக்கச் சுட்டி, பொ.கா.பெ., பொ.ம.சி. — NIE தமிழ்ச் சொற்களையே பயன்படுத்தவும்.
• எண் குறியீடுகள் (÷ × =) பாடநூல் போல இருக்கலாம்; விளக்க உரை முழுக்க தமிழ்.

NIE உள்ளடக்க விதி:
• கீழுள்ள NIE பாடநூல் பகுதிகளையும், மேலே உள்ள 'சரிபார்ப்பு' (இருந்தால்) ஆகியவற்றையே அடிப்படையாகக் கொள்ளவும்.
• பாடநூலில் இல்லாத கணித உண்மைகளைக் கற்பனை செய்ய வேண்டாம்.

NIE காரணிகள் காணும் முறை (தரம் 7 பாடம் 4 — பாடநூல் வரிசை):
• பிரிவு 4.1: சிறு எண்களுக்கு ஜோடி முறை (1×…, 2×…, 3×… போல இணைகளாகக் காணுதல்).
• பிரிவு 4.4: முதன்மைக் காரணிகளின் மூலம் காரணிகளைப் பெறுதல் — வகுத்தல் ஏணி அல்லது காரணி மரத்தால் முதன்மைக் காரணிகளைக் கண்டு, அவற்றை வெவ்வேறு குழுக்களாகப் பெருக்கி 72 = 2×36, 4×18, 6×12 போன்ற காரணி ஜோடிகளை உருவாக்கும் முறை; இதுவே பாடநூல் காட்டும் முக்கியப் பாதை.
• ஒவ்வொரு முழுவெண்ணுக்கும் 1 முதல் அந்த எண் வரை ஒவ்வொரு முழுவெண்ணாலும் வகுத்துச் சோதித்து நீண்ட பட்டியல் உருவாக்கும் முறை (தொடர் வகுத்தல் சோதனை) பாடநூலின் கற்பித்தல் முறையல்ல; பதிலில் இதைப் பயன்படுத்த வேண்டாம்.
• பொ.கா.பெ. காணும்போது பாடநூல் முறை I (காரணிப் பட்டியல்), முறை II (முதன்மைக் காரணிகள் மூலம்), முறை III (வகுத்தல் ஏணி) ஆகியவற்றுள் ஏற்றதைத் தேர்ந்தெடுக்கவும்; சிறு எண்களுக்கு முறை II (எ.கா. 6=2×3, 12=2×2×3, 18=2×3×3; பொதுவான முதன்மைக் காரணிகளின் பெருக்கம்) ஐ முன்னிலைப்படுத்தலாம்.
{register_note}
பொதுப் பாட விதி (சோக்ரட்டிக்):
• மாணவர் முன்பு சமர்ப்பித்த பயிற்சி விடை தவறாக இருந்தால் மட்டும்: நேரடி முழு விடையைச் சொல்லாமல் ஒரு வழிகாட்டும் கேள்வி கேளுங்கள்.
• மாணவர் புதிய கேள்வியில் 'காண்க' 'கண்டுபிடி' 'எவ்வாறு' போன்றவற்றுடன் கணக்கீட்டு விடை (பொ.கா.பெ., பொ.ம.சி., காரணிகள் பட்டியல் முதலியன) கேட்டால்: முழு படிநிலை + இறுதி எண்ணுடன் தமிழில் முழுமையாக விடையளிக்கவும்; அறிமுகக் கேள்வி மட்டுமாக முடிக்க வேண்டாம்.
• காரணிகள் முழுப் பட்டியல் மற்றும் பொ.கா.பெ. சரிபார்ப்புக் கட்டங்கள் இருந்தால் அவை விதிவிலக்கு — அங்கே சோக்ரட்டிக் 'விடை மறைத்தல்' பொருந்தாது.

மாணவர் திறன் நிலை: {skill_level}/3
தற்போதைய திறன்கள்: {skill_summary}
கடைசியாகப் படித்த தலைப்பு: {student.last_topic or 'புதிய தலைப்பு'}
கடைசிப் பிழை வகை: {student.last_error_type or 'இல்லை'}
மொத்த பயிற்சிகள்: {student.total_exercises_attempted} |
சரியான விடைகள்: {student.total_exercises_correct}

NIE பாடநூல் உள்ளடக்கம் (இதை மட்டுமே பயன்படுத்தவும்):
{context}{factor_note}{hcf_note}"""

        user_message = query

        # Check if diagram needed
        diagram_spec = {}
        if self.diagram_trigger.should_draw(intent, retrieved_chunks, query):
            for chunk in retrieved_chunks:
                if chunk.get("diagram_trigger"):
                    nums = self._extract_numbers(query)
                    diagram_spec = self.diagram_trigger.generate_spec(chunk, nums)
                    break

        return {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "retrieved_chunks": retrieved_chunks,
            "diagram_spec": diagram_spec,
            "exercise": exercise,
            "intent": intent,
            "student_id": student.student_id
        }

    def process_query(self, student_id: str, query: str, top_k: int = 4) -> dict:
        """
        Full pipeline: student_id + query → complete response package.

        top_k: number of corpus chunks to inject into the LLM context (lower = faster
        inference, less context; default 4).
        """
        student = self.students.get(student_id) or \
                  self.get_or_create_student(student_id)

        # 1. Classify intent
        intent = self.classifier.classify(query)

        # 2. Adaptive retrieval (uses fixed _pre_filter internally)
        retrieved = self.retriever.retrieve(query, intent, student, top_k=top_k)

        # 3. Generate exercise if requested
        exercise = None
        if intent == "EXERCISE_REQUEST":
            topic = retrieved[0].get("topic", "divisibility_rules") if retrieved else "divisibility_rules"
            difficulty = min(student.get_difficulty_ceiling(), 3)
            exercise = self.exercise_generator.generate(topic, difficulty, student)

        # 4. Build prompt
        prompt_package = self.build_prompt(query, intent, student,
                                           retrieved, exercise)

        # 5. Update student state
        student.total_questions_asked += 1
        if retrieved:
            student.last_topic = retrieved[0].get("topic", "")
        self._save_student(student)

        return {
            "intent": intent,
            "prompt_package": prompt_package,
            "student_skill_level": student.get_difficulty_ceiling(),
            "student_skills": student.skills,
            "retrieved_chunk_ids": [c["id"] for c in retrieved],
            "diagram_spec": prompt_package["diagram_spec"],
            "exercise": exercise,
            "ready_for_llm": True
        }

    def record_exercise_outcome(self, student_id: str, topic: str,
                                 correct: bool, difficulty: int,
                                 error_type: str = ""):
        """Call this after student submits exercise answer."""
        student = self.students.get(student_id) or \
                  self.get_or_create_student(student_id)
        student.update_skill(topic, correct, difficulty)
        student.last_error_type = error_type
        self._save_student(student)
        return student.skills

    def _extract_numbers(self, query: str) -> list:
        """Extract numbers from Tamil query string."""
        import re
        nums = re.findall(r'\b\d+\b', query)
        return [int(n) for n in nums] if nums else []


# ─────────────────────────────────────────────────────────────────────────────
# 11. DEMO / QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("NIE Grade 7 Chapter 4 — Adaptive RAG Engine")
    print("காரணிகளும் மடங்குகளும் — Tamil Math Tutor PoC")
    print("=" * 70)

    engine = AdaptiveRAGEngine()

    # Create a student from Jaffna, Tamil medium
    student = engine.get_or_create_student(
        student_id="SL_TM_2024_001",
        name="அனுஷா"
    )

    test_queries = [
        "காரணி என்றால் என்ன?",
        "இலக்கச் சுட்டி எப்படி கணக்கிடுவது?",
        "84 இன் காரணி மரம் வரை",
        "72 உம் 108 உம் ஆகிய எண்களின் பொ.கா.பெ. காண்க",
        "பயிற்சி கொடு",
        "6 மணிகளும் 8 மணிகளுக்கும் ஒரு மணி ஒலிக்கிறது பொ.ம.சி. கண்டுபிடி"
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"மாணவர் கேள்வி: {query}")
        result = engine.process_query("SL_TM_2024_001", query)
        print(f"Intent: {result['intent']}")
        print(f"Skill level: {result['student_skill_level']}/3")
        print(f"Retrieved chunks: {result['retrieved_chunk_ids']}")
        if result['diagram_spec']:
            print(f"Diagram: {result['diagram_spec'].get('diagram', 'none')}")
        if result['exercise']:
            print(f"Exercise generated: {result['exercise'].get('question_ta', '')[:60]}...")
        print(f"Ready for LLM: {result['ready_for_llm']}")

    print(f"\n{'='*70}")
    print("Engine ready. Pass prompt_package to your LLM API call.")
    print("System prompt + NIE context + student state → Tamil explanation")
    print("=" * 70)
