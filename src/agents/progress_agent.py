"""Aggregate progress views and Tamil summaries for learners."""

from __future__ import annotations

from ..models.student import StudentProfile

_TOPIC_LABEL_TA = {
    "divisibility_rules": "வகுபடும் விதிகள்",
    "digit_sum": "இலக்கச் சுட்டி",
    "factor_listing": "காரணிகள் பட்டியல்",
    "prime_factorization": "முதன்மைக் காரணிப்படுத்தல்",
    "hcf": "பொ.கா.பெ.",
    "lcm": "பொ.ம.சி.",
    "word_problems": "வார்த்தைக் கணக்குகள்",
}


class ProgressAgent:
    def get_progress_report(self, student: StudentProfile) -> dict:
        skills = dict(student.skills)
        mastered = [k for k, v in skills.items() if v >= 0.75]
        in_progress = [k for k, v in skills.items() if 0.3 <= v < 0.75]
        not_started = [k for k, v in skills.items() if v < 0.3]
        total_skills = len(skills) if skills else 1
        chapter_completion_pct = (
            sum(1 for v in skills.values() if v >= 0.5) / total_skills
        ) * 100.0
        patterns = dict(student.error_patterns)
        common_errors = sorted(patterns.items(), key=lambda x: -x[1])[:3]
        common_errors = [e[0] for e in common_errors]

        return {
            "skills": skills,
            "mastered": mastered,
            "in_progress": in_progress,
            "not_started": not_started,
            "overall_accuracy": student.accuracy(),
            "total_questions": student.total_questions_asked,
            "total_exercises": student.total_exercises_attempted,
            "chapter_completion_pct": chapter_completion_pct,
            "error_patterns": patterns,
            "common_errors": common_errors,
        }

    def get_student_summary_ta(self, student: StudentProfile) -> str:
        n_skills = len(student.skills)
        mastered_n = len([v for v in student.skills.values() if v >= 0.75])
        weak = [k for k, v in student.skills.items() if v < 0.6]
        next_skill = weak[0] if weak else ""
        next_ta = _TOPIC_LABEL_TA.get(next_skill, next_skill or "—")
        return (
            f"நீங்கள் {n_skills} திறன்களில் {mastered_n} ஐ நன்கு கற்றுள்ளீர்கள். "
            f"அடுத்து கவனம்: {next_ta}."
        )

    def compare_with_baseline(self, student: StudentProfile, baseline_scores: dict) -> dict:
        out: dict[str, dict[str, float]] = {}
        for k, cur in student.skills.items():
            base = float(baseline_scores.get(k, 0.0))
            delta = float(cur) - base
            out[k] = {
                "current": float(cur),
                "baseline": base,
                "delta": delta,
                "improved": 1.0 if delta > 0.05 else 0.0,
                "regressed": 1.0 if delta < -0.05 else 0.0,
            }
        return out
