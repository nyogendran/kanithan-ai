"""Build Tamil NIE system prompts and call Gemini (stream or full text)."""

from __future__ import annotations

from typing import Iterator

from ..models import QueryContext, RetrievedContext, StudentProfile

# Method number → topic keys in NIE_CORPUS that carry the textbook procedure.
_METHOD_TOPIC_MAP: dict[int, set[str]] = {
    1: {
        "factor_listing_pair_method",
        "hcf_method_1_list",
        "lcm_prime_method",
    },
    2: {
        "prime_factorization_tree",
        "hcf_method_2_prime",
        "lcm_division_method",
    },
    3: {
        "prime_factorization_division",
        "hcf_method_3_division",
    },
}

# Banned-method rules keyed by method number (these don't change per topic).
_METHOD_BAN_RULES: dict[int, str] = {
    1: """தடை:
• 'காரணி மரம்' அல்லது 'வகுத்தல் ஏணி' வடிவத்தைப் பயன்படுத்த வேண்டாம்.
• 'தொடர் வகுத்தல் சோதனை' போன்ற நீண்ட வகுத்தல்-சோதனை பட்டியல் எழுத வேண்டாம்.
• 'முதன்மைக் காரணிகள்' அடிப்படையில் மட்டும் விளக்க வேண்டாம்.""",
    2: """தடை:
• முறை I மாதிரி 'a × b' ஜோடி பெருக்கம் பட்டியல் வடிவத்தை மட்டும் எழுத வேண்டாம்.
• 'வகுத்தல் ஏணி' ('2 | N' போல) வடிவம் எழுத வேண்டாம்.
• '1 முதல் n வரை வகுத்துச் சோதித்தல்' போன்ற நீண்ட வகுத்தல்-சோதனை எழுத வேண்டாம்.""",
    3: """தடை:
• முறை I மாதிரி 'a × b' ஜோடி பெருக்கம் மட்டும் எழுத வேண்டாம்.
• காரணி மரம் முறையை முழுமையாக எழுத வேண்டாம்.
• '1 முதல் n வரை' வகுத்துச் சோதிக்கும் முறையைப் பயன்படுத்த வேண்டாம்.""",
}


class TeachingAgent:
    def __init__(self, gemini_client: object | None = None, model: str = "gemini-2.5-flash"):
        self.gemini_client = gemini_client
        self.model = model

    @staticmethod
    def _extract_dynamic_scaffold(
        expected_method_number: int,
        retrieved: RetrievedContext,
    ) -> str | None:
        """
        Try to build the method scaffold dynamically from retrieved NIE corpus
        chunks rather than using hardcoded text.  Returns the scaffold string
        if a matching method chunk is found, else None.
        """
        desired_topics = _METHOD_TOPIC_MAP.get(expected_method_number, set())
        if not desired_topics:
            return None

        # Also match by method_number metadata.
        for chunk in retrieved.chunks:
            chunk_topic = (chunk.get("topic") or "").strip()
            chunk_method = chunk.get("method_number")
            chunk_type = chunk.get("type", "")

            is_method_chunk = chunk_type in ("method", "worked_example")
            topic_match = chunk_topic in desired_topics
            method_match = chunk_method == expected_method_number

            if is_method_chunk and (topic_match or method_match):
                content = (chunk.get("content_ta") or "").strip()
                if content:
                    ban_rules = _METHOD_BAN_RULES.get(expected_method_number, "")
                    return (
                        f"முறை ஒப்பந்தம் (மட்டுமே): முறை {'I' if expected_method_number == 1 else 'II' if expected_method_number == 2 else 'III'}.\n"
                        f"கீழே உள்ள பாடநூல் முறையை அப்படியே பின்பற்றுங்கள்:\n\n"
                        f"{content}\n\n"
                        f"{ban_rules}"
                    )
        return None

    @staticmethod
    def _fallback_scaffold(expected_method_number: int) -> str:
        """Hardcoded scaffold — used only when no NIE corpus chunk is retrieved."""
        return {
            1: """முறை ஒப்பந்தம் (மட்டுமே): முறை I.
பயன்படுத்த வேண்டிய பாதை:
1) ஒவ்வொரு எண்ணின் காரணிகளை 'a × b' என்ற **ஜோடி பெருக்கம்** வடிவில் காண்க.
2) ஜோடிகளிலிருந்து காரணிகளின் முழுப் பட்டியலை ஏறுவரிசையில் எழுதுங்கள்.
3) பொ.கா.பெ. கேள்வி என்றால்: ஒவ்வொரு எண்ணுக்கும் காரணிப் பட்டியல் → பொதுக் காரணிகள் → பொ.கா.பெ.
""" + _METHOD_BAN_RULES.get(1, ""),
            2: """முறை ஒப்பந்தம் (மட்டுமே): முறை II.
பயன்படுத்த வேண்டிய பாதை:
1) ஒவ்வொரு எண்ணையும் முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுங்கள் (காரணி மரம் பாதை).
2) பொ.கா.பெ. கேள்வி என்றால்: பொதுவான முதன்மைக் காரணிகளின் பெருக்கம் = பொ.கா.பெ.
3) இறுதியில் சரியான விடையை முழுமையாக எழுதுங்கள்.
""" + _METHOD_BAN_RULES.get(2, ""),
            3: """முறை ஒப்பந்தம் (மட்டுமே): முறை III.
பயன்படுத்த வேண்டிய பாதை:
1) 'வகுத்தல் ஏணி' போல இடது பக்கத்தில் வகுக்கும் முதன்மை எண்ணை எழுதுங்கள்; வலது பக்கத்தில் பகுதிகளை கீழே தொடர்ந்து எழுதுங்கள்.
2) நிறுத்தும் நிபந்தனைப்படி தொடரவும்; பிறகு வகுக்கப்பட்ட முதன்மை எண்களைப் பெருக்கி இறுதி விடை காண்க.
3) இறுதியில் சரியான விடையை முழுமையாக எழுதுங்கள்.
""" + _METHOD_BAN_RULES.get(3, ""),
        }.get(expected_method_number, "")

    def build_system_prompt(
        self,
        query_ctx: QueryContext,
        student: StudentProfile,
        retrieved: RetrievedContext,
        factor_note: str,
        hcf_note: str,
        lcm_note: str,
        register_note: str,
        expected_method_number: int | None = None,
        expected_method_label: str | None = None,
    ) -> str:
        skill_level = student.get_difficulty_ceiling()
        skill_summary = ", ".join(
            f"{k}: {v:.1f}" for k, v in student.skills.items() if v > 0
        ) or "தொடக்க நிலை"

        context = "\n\n---\n\n".join(
            f"பாடம் {chunk.get('section', '')} (பக்கம் {chunk.get('page', '')}):\n{chunk.get('content_ta', '')}"
            for chunk in retrieved.chunks
        )

        # ---- Method-only enforcement (textbook compliance contract) ----
        if expected_method_number is None:
            expected_method_number = 2 if (query_ctx.topic or "").lower() in ("hcf", "lcm") else 1

        if not expected_method_label:
            expected_method_label = (
                "முறை I (காரணிப் பட்டியல் / ஜோடி பெருக்கம்)"
                if expected_method_number == 1
                else "முறை II (முதன்மைக் காரணிகள் / காரணி மரம்)"
                if expected_method_number == 2
                else "முறை III (வகுத்தல் ஏணி)"
            )

        # Dynamic scaffold: prefer content from retrieved NIE corpus chunks;
        # fall back to hardcoded scaffold only if no method chunk was retrieved.
        method_scaffold = self._extract_dynamic_scaffold(expected_method_number, retrieved)
        if method_scaffold is None:
            method_scaffold = self._fallback_scaffold(expected_method_number)

        return f"""நீங்கள் ஒரு அனுபவமிக்க கணித ஆசிரியர்.
தரம் 7 தமிழ்வழி மாணவர்களுக்கு இலவசப் பாடநூல் (NIE) படி கற்பிக்கிறீர்கள்.

மொழி விதி — மீறினால் பதில் தவறாகக் கருதப்படும்:
• உங்கள் முழு பதிலும் தமிழில் மட்டுமே. ஆங்கில வாக்கியங்கள், ஆங்கிலச் சொற்கள், ஆங்கில எழுத்துகள் (A–Z) ஒன்றும் கூடாது.
• "Let's", "We", "factor", "remainder", "whole number" போன்றவை முற்றிலும் தடை.
• விளக்கத்தில் காரணி, மடங்கு, இலக்கச் சுட்டி, பொ.கா.பெ., பொ.ம.சி. — NIE தமிழ்ச் சொற்களையே பயன்படுத்தவும்.
• எண் குறியீடுகள் (÷ × =) பாடநூல் போல இருக்கலாம்; விளக்க உரை முழுக்க தமிழ்.

NIE உள்ளடக்க விதி:
• கீழுள்ள NIE பாடநூல் பகுதிகளையும், மேலே உள்ள 'சரிபார்ப்பு' (இருந்தால்) ஆகியவற்றையே அடிப்படையாகக் கொள்ளவும்.
• பாடநூலில் இல்லாத கணித உண்மைகளைக் கற்பனை செய்ய வேண்டாம்.

சுருக்க விதி (மிக முக்கியம்):
• ஒரு எண்ணின் மடங்குகளை 10-க்கு மேல் பட்டியலிட வேண்டாம்.
• காரணிகள்/மடங்குகள் பட்டியலில் "..." குறியீடு பயன்படுத்தி சுருக்கவும்.
• பொ.ம.சி. கேள்விக்கு: மடங்குகளை நீளமாகப் பட்டியலிடுவதற்குப் பதிலாக NIE வகுத்தல் முறை அல்லது முதன்மைக் காரணிகள் உயர் வலு முறையைப் பயன்படுத்தவும்.
• எண்களை எப்போதும் இலக்கங்களாக (6, 12, 18) எழுதவும் — சொல்லாக (ஆறு, பன்னிரண்டு) அல்ல.

நிலையான சொற்கள் (எப்போதும் பயன்படுத்தவும்):
• பொதுக் காரணிகளுட் பெரியது → பொ.கா.பெ. (சுருக்கம் கட்டாயம்)
• பொது மடங்குகளுட் சிறியது → பொ.ம.சி. (சுருக்கம் கட்டாயம்)
• முதன்மை எண், முதன்மைக் காரணி, காரணி மரம், வகுத்தல் ஏணி — NIE சொற்களையே பயன்படுத்தவும்.

முறை ஒப்பந்தம் (மட்டுமே):
இந்தக் கேள்விக்கான NIE எதிர்பார்க்கப்படும் முறை: {expected_method_label}.
மேலே உள்ள method_scaffold-இன் கட்டமைப்பை மட்டும் பின்பற்றுங்கள் (முறை மாற்ற வேண்டாம்).
{method_scaffold}
{register_note}
பொதுப் பாட விதி (சோக்ரட்டிக்):
• மாணவர் முன்பு சமர்ப்பித்த பயிற்சி விடை தவறாக இருந்தால் மட்டும்: நேரடி முழு விடையைச் சொல்லாமல் ஒரு வழிகாட்டும் கேள்வி கேளுங்கள்.
• மாணவர் புதிய கேள்வியில் 'காண்க' 'கண்டுபிடி' 'எவ்வாறு' போன்றவற்றுடன் கணக்கீட்டு விடை (பொ.கா.பெ., பொ.ம.சி., காரணிகள் பட்டியல் முதலியன) கேட்டால்: முழு படிநிலை + இறுதி எண்ணுடன் தமிழில் முழுமையாக விடையளிக்கவும்; அறிமுகக் கேள்வி மட்டுமாக முடிக்க வேண்டாம்.
• காரணிகள் முழுப் பட்டியல் மற்றும் பொ.கா.பெ. சரிபார்ப்புக் கட்டங்கள் இருந்தால் அவை விதிவிலக்கு — அங்கே சோக்ரட்டிக் 'விடை மறைத்தல்' பொருந்தாது.

இறுதித் தொடர்பு (கட்டாயம் — ஒவ்வொரு முழு விடையின் முடிவிலும்):
• இரண்டு வரிகளில்: மாணவர் புரிந்துகொண்டாரா என ஒரு எளிய சுயச் சோதனைக் கேள்வி (எ.கா. "வகுத்தல் ஏணியின் கடைசி வரி எப்படி இருக்க வேண்டும்?" அல்லது "பொ.ம.சி. எண்ணை எப்படிச் சரிபார்ப்பீர்கள்?").
• ஒரு வரியில்: கூடுதல் கேள்விகளுக்கு அழைப்பு — எ.கா. முதன்மை எண் என்றால் என்ன, வகுத்தல் ஏணி எப்படி நிறுத்துவது, ஒரு எண் 15 ஆல் வகுபடுமா எப்படித் தெரியும் — இப்படிப் பின்தொடர்ந்து கேட்கலாம் என்று தெளிவாக எழுதவும்.
• முழு விடைக்குப் பிறகு மட்டும் இந்த இறுதித் தொடர்பைச் சேர்க்கவும்; நடுவில் குறுக்கிட வேண்டாம்.

மாணவர் திறன் நிலை: {skill_level}/3
தற்போதைய திறன்கள்: {skill_summary}
கடைசியாகப் படித்த தலைப்பு: {student.last_topic or 'புதிய தலைப்பு'}
கடைசிப் பிழை வகை: {student.last_error_type or 'இல்லை'}
மொத்த பயிற்சிகள்: {student.total_exercises_attempted} |
சரியான விடைகள்: {student.total_exercises_correct}

NIE பாடநூல் உள்ளடக்கம் (இதை மட்டுமே பயன்படுத்தவும்):
{context}{factor_note}{hcf_note}{lcm_note}"""

    @staticmethod
    def _build_generate_config(
        types: object,
        *,
        system: str,
        temperature: float,
        max_output_tokens: int,
        disable_thinking: bool,
    ) -> object:
        kwargs: dict = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if disable_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kwargs)

    def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        disable_thinking: bool = True,
    ) -> Iterator[str]:
        from google import genai
        from google.genai import types

        client = self.gemini_client or genai.Client(api_key=api_key)
        config = self._build_generate_config(
            types,
            system=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        stream = client.models.generate_content_stream(
            model=self.model,
            contents=user_message,
            config=config,
        )
        for chunk in stream:
            piece = chunk.text or ""
            if piece:
                yield piece

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        from google import genai
        from google.genai import types

        client = self.gemini_client or genai.Client(api_key=api_key)
        config = self._build_generate_config(
            types,
            system=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            disable_thinking=True,
        )
        response = client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=config,
        )
        return (response.text or "").strip()
