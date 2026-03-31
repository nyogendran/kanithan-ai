"""Sri Lankan Tamil regional dialect detection and NIE-standard normalization."""

from __future__ import annotations

from src.data.glossary import DIALECT_NORMALIZER
from src.models.intents import Dialect
from src.models.student import StudentProfile

DIALECT_SIGNATURES: dict[Dialect, list[str]] = {
    Dialect.JAFFNA: ["வகுதல்", "ஆகும்", "என்பது", "காண்போம்", "செய்க"],
    Dialect.BATTICALOA: ["வகுத்தல்னா", "எவ்வளவு", "போடு", "இது என்னன்னு"],
    Dialect.ESTATE: [
        "வகுத்தல்க்கு",
        "பண்ணுவது",
        "சொல்லுங்க",
        "இதுக்கு",
        "எப்படி பண்றது",
    ],
    Dialect.COLOMBO: ["factor", "HCF", "LCM", "find பண்றது", "method காட்டு"],
    Dialect.VANNI: ["வகுத்தல்", "காண்பது", "எப்படி கண்டுபிடிப்பது"],
}

DISTRICT_MAP: dict[str, Dialect] = {
    # Northern — Jaffna vs Vanni split
    "jaffna": Dialect.JAFFNA,
    "mannar": Dialect.JAFFNA,
    "northern": Dialect.JAFFNA,
    "kilinochchi": Dialect.VANNI,
    "mullaitivu": Dialect.VANNI,
    "vavuniya": Dialect.VANNI,
    "vanni": Dialect.VANNI,
    # East
    "batticaloa": Dialect.BATTICALOA,
    "trincomalee": Dialect.BATTICALOA,
    "ampara": Dialect.BATTICALOA,
    "east": Dialect.BATTICALOA,
    # Western / Colombo corridor
    "colombo": Dialect.COLOMBO,
    "gampaha": Dialect.COLOMBO,
    "kalutara": Dialect.COLOMBO,
    "western": Dialect.COLOMBO,
    # Estate / plantation
    "estate": Dialect.ESTATE,
    "tea_estate": Dialect.ESTATE,
    "up_country": Dialect.ESTATE,
    "central_estate": Dialect.ESTATE,
    "plantation": Dialect.ESTATE,
    "nuwara eliya": Dialect.ESTATE,
    "badulla": Dialect.ESTATE,
}


class DialectAgent:
    """Detect SL Tamil dialect and normalize queries toward NIE textbook register."""

    def detect_and_normalize(
        self, raw_query: str, student_district: str
    ) -> tuple[Dialect, str]:
        dkey = (student_district or "").strip().lower()
        dialect: Dialect
        if dkey in DISTRICT_MAP:
            dialect = DISTRICT_MAP[dkey]
        else:
            dialect = self._dialect_from_signatures(raw_query)

        normalized = self._apply_normalizer(raw_query)
        return dialect, normalized

    def _dialect_from_signatures(self, text: str) -> Dialect:
        best: Dialect = Dialect.UNKNOWN
        best_hits = 0
        for dia, keywords in DIALECT_SIGNATURES.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > best_hits:
                best_hits = hits
                best = dia
        return best if best_hits > 0 else Dialect.UNKNOWN

    def _apply_normalizer(self, text: str) -> str:
        out = text
        for src, dst in sorted(
            DIALECT_NORMALIZER.items(), key=lambda kv: -len(kv[0])
        ):
            out = out.replace(src, dst)
        return out

    def get_nie_register_guidance(self, student: StudentProfile) -> str:
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
