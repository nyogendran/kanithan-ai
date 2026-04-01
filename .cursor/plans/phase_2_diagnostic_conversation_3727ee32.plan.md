---
name: Phase 2 Diagnostic Conversation
overview: Add a Socratic diagnostic conversation loop to the voice tutor. Before teaching an advanced topic, the agent walks the prerequisite graph, probes 1-3 weak skills with targeted questions, evaluates answers (programmatically or via LLM), updates skill scores, and only then proceeds to the original question.
todos: []
isProject: false
---

# Phase 2: Diagnostic Conversation and Gap Detection

## Current State

Phase 1 wired the voice UI to `/api/v1/voice/converse`, which has a basic prereq check: a **single** hardcoded question ("12-இன் காரணிகள் என்ன?") using a flat `_PREREQ_MAP` that only maps to `"factors"`. The student's answer is never evaluated — the next turn just resets to LISTENING.

The infrastructure already in place:

- `**PREREQUISITE_GRAPH`** in [src/data/prerequisite_graph.py](src/data/prerequisite_graph.py) — full dependency DAG (14 topics with prereq edges)
- `**StudentProfile.skills`** — 7 skill scores (0.0-1.0) tracked in SQLite
- `**StudentProfile.get_unlocked_topics()`** and `**weak_topics()`** — already walk the graph
- `**ExerciseAgent`** — procedural question generation by topic (factors, HCF, LCM, divisibility, etc.)
- `**AnswerVerifierAgent**` — LLM-based answer checking with Socratic hints
- `**MathVerifierAgent**` — programmatic factor/HCF/LCM computation

## Conversation Flow

```
Student: "24, 36 இன் மீ.பொ.கா. எப்படி கண்டுபிடிப்பது?"

[Agent walks prereq graph from hcf_definition]
  hcf_definition requires: factor_listing (skill=0.1), prime_factorization (skill=0.0)
  factor_listing requires: divisibility_rules (skill=0.6) -- OK, skip

[Diagnostic queue: factor_listing, prime_factorization]

Agent: "நல்ல கேள்வி! முதலில் உங்கள் அடிப்படையை சோதிக்கிறேன்.
        18-இன் அனைத்து காரணிகளையும் சொல்லுங்கள்."
State: DIAGNOSING

Student: "1, 2, 3, 6, 9, 18"
[Programmatic check: correct!] → factor_listing skill += 0.15

Agent: "சரி! அடுத்த கேள்வி — 30-ஐ முதன்மைக் காரணிகளின் பெருக்கமாக எழுதுங்கள்."

Student: "2 × 3 × 5"
[Programmatic check: correct!] → prime_factorization skill += 0.15

Agent: "மிக நன்று! இப்போது உங்கள் கேள்விக்கு வருவோம்..."
State: RESPONDING → answers the original HCF question
```

If the student answers **incorrectly**:

```
Student: "1, 2, 3, 6
```

