# Feasibility Analysis: AI Math Tutor in Tamil for Sri Lankan Students

**Scope:** Grade 6–9 Mathematics and Science  
**Language:** Tamil (primary); Sinhala (Phase 2)  
**Target communities:** Jaffna Tamil, Batticaloa Tamil, Central Province estate Tamil, Tamil-speaking Muslim students (Colombo, Trincomalee, Mannar)  

---

## 1. The Dialect Landscape — What You Are Actually Dealing With

Sri Lankan Tamil is **not a monolithic dialect**. The variation across communities is significant and pedagogically consequential.

```
Sri Lankan Tamil varieties relevant to this platform:
┌─────────────────────────────────────────────────────────────────┐
│  Community           │ Variety           │ Key characteristics  │
├─────────────────────────────────────────────────────────────────┤
│  Jaffna (Northern)   │ Jaffna Tamil      │ Conservative phonol- │
│                      │                   │ ogy; formal register │
│                      │                   │ close to written std │
├─────────────────────────────────────────────────────────────────┤
│  Batticaloa (East)   │ Batticaloa Tamil  │ Distinct vowel system│
│                      │                   │ "koṭuṅgu Tamil";    │
│                      │                   │ differs from Jaffna  │
│                      │                   │ in prosody & lexicon │
├─────────────────────────────────────────────────────────────────┤
│  Estate (Central)    │ Up-Country Tamil  │ Strong Indian Tamil  │
│                      │                   │ substrate; Sinhala   │
│                      │                   │ borrowings; informal │
│                      │                   │ register dominant    │
├─────────────────────────────────────────────────────────────────┤
│  Tamil-speaking      │ Sri Lanka Moor    │ Heavily Sinhala-     │
│  Muslims             │ Tamil             │ influenced; Arabic   │
│                      │                   │ borrowings; pronunc. │
│                      │                   │ differs from Jaffna  │
├─────────────────────────────────────────────────────────────────┤
│  Colombo urban       │ Colombo Tamil     │ Code-switched with   │
│                      │                   │ English; informal;   │
│                      │                   │ least conservative   │
└─────────────────────────────────────────────────────────────────┘
```

**The NIE textbook uses a standardised written Tamil register** that is neutral across these varieties — it is not Jaffna Tamil, not Batticaloa Tamil, not estate Tamil. It is formal Modern Standard Tamil (MST) of Sri Lanka. This is both a strength and a challenge:

- Strength: the written corpus (textbooks) is consistent.
- Challenge: **student speech** in each community differs significantly from MST, so voice input accuracy will vary by community.

---

## 2. Technical Feasibility by Capability

### 2.1 Tamil Text Generation (LLM responses)

**Feasibility: HIGH**

| Concern | Assessment |
|---------|-----------|
| LLM Tamil quality | llama3, Gemma 3 write grammatical Tamil but default to Indian Tamil register |
| NIE register enforcement | Achievable via RAG (retrieved NIE text anchors vocabulary) + system prompt |
| Dialect variation in output | Not required — written MST is the correct output for all communities |
| Mathematical vocabulary | NIE glossary in TOPIC_TO_SKILL covers G7 Ch4; needs extension for G6–G9 |

**Key risk:** Without retrieved NIE context, llama3 uses Indian Tamil mathematical vocabulary (e.g. மீ.பொ.வ instead of பொ.கா.பெ). The RAG grounding is essential.

### 2.2 Voice Input — Tamil ASR (Speech to Text)

**Feasibility: MEDIUM — varies significantly by community**

```
ASR system comparison for Sri Lankan Tamil:

Google Cloud STT (Tamil)
  Jaffna Tamil:    Good (phonology close to training data)
  Batticaloa:      Moderate (vowel system differs)
  Estate Tamil:    Poor-Moderate (strong accent + Sinhala mix)
  Muslim Tamil:    Moderate (pronunciation diverges)
  Mathematical terms: Poor (technical vocab undertrained)

OpenAI Whisper (medium/large)
  All SL varieties: Moderate overall; better than Google on accented speech
  Mathematical terms: Poor without fine-tuning
  Offline capable:  Yes (critical for Northern Province)

Fine-tuned Whisper on SL Tamil
  Achievable with 50–100 hours of labelled SL Tamil speech
  University of Moratuwa Language Technology Research Lab has datasets
  Cost: ~$5,000–15,000 for data collection + fine-tuning
  Timeline: 3–4 months
```

**Practical recommendation for PoC:**
- Use Google Cloud STT for connected regions (Colombo, Jaffna city)
- Use Whisper locally for offline mode
- Accept ~15–25% WER on mathematical queries as a known limitation
- Add **post-ASR correction**: if recognised text contains no known mathematical terms, prompt student to type

### 2.3 Voice Output — Tamil TTS

**Feasibility: HIGH**

Google Cloud TTS Tamil is production-quality for MST written Tamil. Since the LLM outputs written MST (not dialect), TTS accuracy is high across all student communities. The student hears a neutral educated Tamil voice — this is appropriate for a school context.

ElevenLabs has Tamil voice cloning capability if a more natural, locally-accented voice is needed for community acceptance.

### 2.4 Handwriting Recognition (Writing Pad)

**Feasibility: HIGH for digits/symbols; MEDIUM for Tamil script**

| Input type | Accuracy | Technology |
|-----------|---------|-----------|
| Arabic numerals (0–9) | Very high | ML Kit on-device |
| Mathematical operators (+, ÷, ×, =) | High | ML Kit |
| Tamil digits (௦–௯) | Moderate | ML Kit Tamil model |
| Tamil script words | Moderate-Low | ML Kit has Tamil but accuracy on maths vocabulary is untested |
| Mathematical expressions (36 = 2² × 3²) | Moderate | Custom model needed |

For the PoC: focus handwriting recognition on **numerals and operators only**. Tamil text input for answers should be typed. This is realistic — students in Sri Lankan Tamil schools are accustomed to typing on phones.

### 2.5 Diagram Generation (Factor Trees, Division Ladders)

**Feasibility: HIGH — this is your strongest PoC differentiator**

The `DiagramTrigger` in `adaptive_rag_chapter4.py` generates JSON specs that are mathematically correct (post bug-fixes). Flutter `CustomPainter` can render these deterministically. Animation is achievable in Flutter with `AnimationController`.

No competing platform in Sri Lanka has this. The World Bank, UNICEF, and ADB evaluators will immediately see the differentiation.

### 2.6 Adaptive Learning (Mastery Gating)

**Feasibility: HIGH for PoC; MEDIUM for production accuracy**

The current `StudentProfile` + `PREREQUISITE_GRAPH` + `TOPIC_TO_SKILL` design is pedagogically sound and implementable. The limitation is that **skill estimation from question-answer pairs is noisy** — a student who guesses correctly will be overrated.

For the pilot study, supplement with:
- Pre/post NIE-aligned written assessments (ground truth)
- Teacher observation notes (qualitative)
- Session logs showing which chunks were retrieved and whether the student asked follow-ups

---

## 3. Community-Specific Feasibility Assessment

### 3.1 Jaffna Tamil Students (Northern Province)

**Overall: HIGH feasibility, HIGH impact**

- School infrastructure: government Tamil-medium schools well-established
- Connectivity: unreliable in rural North — **offline-first is mandatory**
- Device: most families have Android phones (low-mid range)
- Teacher cooperation: Northern Province teachers are motivated; Jaffna University partnership available
- Dialect: closest to written MST — lowest ASR error rate

**Critical path:** Offline Ollama + local ChromaDB on phone or shared school server/Raspberry Pi (4 GB RAM is sufficient for Gemma 3:2B + nomic-embed-text).

### 3.2 Batticaloa Tamil Students (Eastern Province)

**Overall: HIGH feasibility, MEDIUM-HIGH impact**

- Dialect is more distinct from MST — some vocabulary differences in everyday speech but mathematical written Tamil is the same (NIE textbook standard)
- ASR will have ~5–10% higher error rate vs Jaffna
- The district has lower school attainment than Northern Province — **higher marginal impact**
- Eastern University partnership for pilot evaluation

**Recommendation:** Include one Batticaloa school in the pilot for dialect-coverage evidence.

### 3.3 Estate Tamil Students (Central Province — Kandy, Nuwara Eliya, Hatton)

**Overall: MEDIUM feasibility, VERY HIGH impact**

This is the **most underserved community** and the one with the greatest potential for measurable learning gains.

| Factor | Assessment |
|--------|-----------|
| Connectivity | Very poor in estate sectors — offline mode is not optional |
| Device access | Lower than North/East; some families share one phone |
| Language register | Significant gap between student spoken Tamil and NIE written Tamil |
| ASR accuracy | Lower — strong accent, Sinhala-mixed speech |
| Teacher support | Estate school teachers are often less equipped in Tamil-medium maths |
| Impact potential | Highest — students most behind NIE benchmark |

**Key technical challenge:** The AI must tolerate **code-switching** (Tamil + Sinhala mixed queries) in voice input. This is a known hard problem. Practical workaround for PoC: text input primarily; voice as enhancement.

**Community trust:** Estate Tamil communities have historical reasons to be cautious of technology-in-education initiatives from Colombo. Partnering with estate sector NGOs (e.g. Save the Children estate sector programme) is essential for adoption.

### 3.4 Tamil-Speaking Muslim Students

**Overall: HIGH feasibility, MEDIUM impact**

- Geographic spread: Colombo, Trincomalee, Mannar, Ampara — varied connectivity
- Most are bilingual (Tamil + Sinhala or Tamil + English) — code-switching in voice input
- Mathematical vocabulary: same NIE Tamil register
- Community trust: requires engagement with mosque schools and Muslim education organisations
- Notable distinction: some Moor Tamil families may prefer the platform also supports English or Sinhala explanations alongside Tamil — plan for Phase 2

---

## 4. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| PDF TSCII encoding — corpus extraction garbled | **Certain** (confirmed) | High | Manual review + TSCII→Unicode conversion; or source UTF-8 PDF from NIE digitally |
| LLM generates Indian Tamil vocabulary | High | High | RAG grounding + system prompt + Tamil-speaker QA review |
| ASR fails on mathematical terms | High | Medium | Post-ASR term correction dict; allow typed fallback |
| Offline inference too slow on mid-range Android | Medium | High | Use Gemma 3:2B (1.9 GB) not 9B; or server-side with cached results |
| Student skill model overestimates progress | Medium | Medium | Weight typed vs guessed answers differently; add confidence signal |
| Community distrust (estate sector) | Medium | High | Co-design with estate community organisations |
| NIE copyright on textbook corpus | **Certain** | Medium | Obtain formal MoE/NIE approval letter; use for non-commercial pilot only |
| Sinhala support requested before ready | Medium | Low | Clearly scope PoC to Tamil only; set expectation explicitly |

---

## 5. What This Requires Beyond Code

### 5.1 Linguist + Pedagogy Reviewer (non-negotiable)

You need one **Sri Lankan Tamil-medium mathematics teacher** with experience across at least two Tamil communities to:
- Review every LLM response in the pilot (20–30 sample per session)
- Validate NIE terminology in generated exercises
- Annotate TSCII-decoded corpus chunks for accuracy

This person can be a retired NIE textbook author, a senior Tamil-medium maths teacher, or a University of Jaffna education faculty member. Budget: $50–80/day for a 2-month engagement.

### 5.2 Data Collection for ASR Fine-tuning (Phase 2)

50 hours of labelled SL Tamil speech in mathematical contexts:
- Students reading NIE exercise problems aloud
- Students dictating answers: "36 இன் காரணிகள் 1, 2, 3, 4, 6, 9, 12, 18, 36"
- Across at least 3 community varieties (Jaffna, Batticaloa, estate)

Without this, ASR will have 20–35% WER on mathematical vocabulary, which will frustrate students enough to abandon voice input.

### 5.3 MoE / NIE Partnership Letter

The corpus extraction from NIE PDFs is for a non-commercial educational pilot. A formal letter from the Ministry of Education or NIE Director-General:
- Protects you from copyright challenge
- Significantly strengthens the World Bank proposal
- Opens access to NIE's digital content team for UTF-8 versions of textbooks

This should be pursued in parallel with technical work.

---

## 6. Verdict by Component

```
Component                  PoC Feasibility   Production Feasibility   Priority
─────────────────────────────────────────────────────────────────────────────
Tamil text generation       ████████ HIGH     ████████ HIGH            Must-have
NIE corpus (manual)         ████████ DONE     ████████ HIGH (CMS)      Must-have
Adaptive retrieval (kw)     ████████ DONE     ████░░░ MEDIUM→vector    Must-have
Adaptive retrieval (vector) ████████ HIGH     ████████ HIGH            Phase 2
Diagram generation (JSON)   ████████ HIGH     ████████ HIGH            Must-have
Flutter rendering (SVG)     ████████ HIGH     ████████ HIGH            Must-have
Student profiling (SQLite)  ████████ DONE     ████████ HIGH (Supabase) Must-have
Mastery gating              ████████ HIGH     ████████ HIGH            Must-have
Voice input (Jaffna Tamil)  ██████░░ GOOD     ████████ HIGH (finetune) Nice-to-have PoC
Voice input (Estate Tamil)  ████░░░░ MEDIUM   ██████░░ MEDIUM          Phase 2+
Handwriting (digits)        ████████ HIGH     ████████ HIGH            Phase 2
Handwriting (Tamil script)  ██░░░░░░ LOW      ████░░░░ MEDIUM          Phase 3
Sinhala support             ░░░░░░░░ N/A      ████████ HIGH            Phase 2
Multi-dialect ASR           ██████░░ MEDIUM   ████████ HIGH (data)     Phase 2
```

---

## 7. Recommended PoC Scope (18 weeks, solo developer)

```
Weeks  1–3:  NIE corpus — TSCII decode + manual review → 100 chunks (Ch4 complete)
Weeks  4–6:  adaptive_rag_chapter4.py — wire to Ollama, evaluate 20 queries with 
              native Tamil speaker, fix vocabulary issues
Weeks  7–9:  Flutter MVP — text query → Tamil response → diagram canvas
Weeks 10–12: Exercise loop — writing pad numerals, step check, Socratic correction
Weeks 13–15: Student pilot — 30 students, 2 schools (Jaffna + Batticaloa)
              Collect: session logs, pre/post assessment, teacher interviews
Weeks 16–18: Analysis + proposal document for World Bank / UNICEF
```

**Total estimated cost (solo developer PoC):**
- Development (self): in-kind
- Cloud inference (Vertex AI / Gemini API for pilot): ~$30–50
- Tamil-medium maths teacher reviewer: ~$2,000–3,000 (2 months, part-time)
- Device procurement for pilot: ~$500 (5 low-end Android phones)
- MoE engagement + travel: ~$500
- **Total: ~$3,000–4,000 cash, rest in-kind**

This is fundable from a single Google.org small grant, UNICEF Innovation seed, or Dialog Axiata CSR engagement.

---

## 8. The Non-Technical Answer

The feasibility question for multi-dialect Sri Lankan Tamil is ultimately this:

**The written language is the same. The spoken language differs. The textbook is in the written language.**

This means:
- **Text generation:** dialect-agnostic. One model, one corpus. ✓
- **Diagram, exercise, mastery:** dialect-agnostic. ✓
- **Voice input:** dialect-sensitive. Requires fine-tuning or tolerance for text input fallback.
- **Community trust:** requires human engagement, not technology.

The technology is ready for Jaffna Tamil text-first today. Voice accuracy for all four Tamil communities at production quality will take one to two years and dedicated data collection. The PoC does not need to solve that — it needs to prove the text-based tutoring loop works, which it can.
