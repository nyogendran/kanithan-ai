# Voice-Based AI Math Tutor — Enterprise Architecture

## 1. Overview

This document defines the architecture for voice-based interaction in the NIE Tamil Math Tutor.
The voice path is **not isolated** — it is a new input/output modality that feeds into the
existing multi-agent orchestrator pipeline.

```
┌────────────────────────────────────────────────────────────────────┐
│                    Student Device (any platform)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │   Mic    │→ │  VAD     │→ │ STT      │→ │  WebSocket /     │    │
│  │ capture  │  │ (local)  │  │ (cloud   │  │  REST client     │    │
│  └──────────┘  └──────────┘  │ or local)│  └────────┬─────────┘    │
│                              └──────────┘           │              │
│  ┌──────────────────────────────────────────────────┘              │
│  │  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  │  TTS playback    │← │ Audio cache  │← │ Speaker      │       │
│  │  └──────────────────┘  └──────────────┘  └──────────────┘       │
│  └─────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────┘
                              │ WebSocket / HTTP
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                       Voice Gateway Service                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐   │
│  │ Session  │→ │ Dialect      │→ │ Math-aware   │→ │ Conver-   │   │
│  │ manager  │  │ normalizer   │  │ post-process │  │ sation    │   │
│  └──────────┘  └──────────────┘  └──────────────┘  │ state     │   │
│                                                    │ machine   │   │
│                                                    └─────┬─────┘   │
└──────────────────────────────────────────────────────────┼─────────┘
                                                           │
                              ▼                            │
┌────────────────────────────────────────────────────────────────────┐
│                  Existing Orchestrator Pipeline                    │
│  InputParser → DialectAgent → IntentAgent → RetrievalAgent →       │
│  MathVerifier → TeachingAgent → DrawingAgent → ExerciseAgent →     │
│  AnswerVerifier → MasteryAgent → SentimentAgent → ProgressAgent    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Response Formatter                             │
│  teaching.explanation_ta + diagram.spec + exercise + TTS audio     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why NOT Isolated (Question 2)

The voice path **must share** these components with text input:

| Shared component         | Why                                                              |
|--------------------------|------------------------------------------------------------------|
| OrchestratorAgent        | Same method enforcement, same diagram generation logic           |
| StudentProfile + DB      | Voice sessions must update the same mastery/progress/sentiment   |
| RetrievalAgent + ChromaDB| Same NIE corpus, same vector embeddings                          |
| TeachingAgent + LLM      | Same Gemini prompt, same method scaffold                         |
| DrawingAgent             | Diagram spec is identical regardless of input modality           |
| ExerciseAgent            | Voice exercises use the same question bank                       |

**What IS voice-specific** (thin layer on top):

| Voice-only component            | Purpose                                           |
|-------------------------------  |---------------------------------------------------|
| VAD (Voice Activity Detection)  | Detect speech boundaries, handle student pauses   |
| STT (Speech-to-Text)            | Convert audio to Tamil text                       |
| Dialect normalizer (audio)      | Acoustic dialect detection + term normalization   |
| Math post-processor             | Number-word→digit, operator→symbol conversion     |
| Conversation state machine      | Multi-turn clarification/fundamentals checking    |
| TTS (Text-to-Speech)            | Convert Tamil explanation to speech               |
| SSML math builder               | Pronounce math expressions correctly              |

**Architecture principle**: Voice is an **adapter layer** that wraps around the existing pipeline,
not a parallel pipeline.

---

## 3. Interactive Conversation State Machine (Question 3)

The voice tutor is NOT a one-shot transcribe→answer system.
It maintains a **conversation state machine** per session:

```
┌──────────────┐
│   LISTENING  │ ← Student speaks
└──────┬───────┘
       │ transcript received
       ▼
┌──────────────┐     low confidence or
│  UNDERSTAND  │────────────────────────┐
│  (analyze)   │                        │
└──────┬───────┘                        ▼
       │                        ┌──────────────┐
       │ complete question      │  CLARIFY     │
       │                        │  (ask back)  │
       ▼                        └──────┬───────┘
┌──────────────┐                       │ student responds
│  CHECK       │←──────────────────────┘
│  FUNDAMEN-   │
│  TALS        │──── student lacks prereqs ──┐
└──────┬───────┘                             │
       │ ready                               ▼
       ▼                             ┌──────────────┐
┌──────────────┐                     │  TEACH       │
│  RESPOND     │                     │  PREREQ      │
│  (full       │←────────────────────└──────────────┘
│   answer)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  EXERCISE    │ ← optional follow-up exercise
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  VERIFY      │ ← check student's spoken answer
└──────────────┘
```

### Conversation state details

#### CLARIFY state
The agent asks back when:
- ASR confidence < 0.6: "கொஞ்சம் மீண்டும் சொல்ல முடியுமா?" (Can you say that again?)
- Ambiguous numbers: "நீங்கள் சொன்னது 114 தானா, அல்லது 140 ஆ?"
- Incomplete question: "எந்த எண்களின் பொ.கா.பெ. காண வேண்டும்?"
- Missing method preference: "எந்த முறையில் காண விரும்புகிறீர்கள்? பட்டியல், காரணி மரம், அல்லது வகுத்தல் ஏணி?"

#### CHECK FUNDAMENTALS state
Before teaching HCF (difficulty 3), the agent checks prerequisites:
- "காரணி என்றால் என்ன என்று உங்களுக்குத் தெரியுமா?" (Do you know what a factor is?)
- If student says "தெரியாது" (I don't know) or gives wrong answer → TEACH PREREQ
- If student says "தெரியும்" (I know) → proceed to RESPOND

#### TEACH PREREQ state
- Agent teaches the prerequisite topic first (e.g., factors before HCF)
- Uses the same TeachingAgent + method scaffold
- After teaching, asks a quick check question
- If passed → return to the original question

#### EXERCISE state
- Agent offers a practice problem vocally
- Student speaks the answer
- Agent verifies using AnswerVerifierAgent

### Implementation approach
The conversation state is managed in the Voice Gateway as a simple FSM:

```python
class ConversationState(Enum):
    LISTENING = "listening"
    UNDERSTANDING = "understanding"
    CLARIFYING = "clarifying"
    CHECKING_FUNDAMENTALS = "checking_fundamentals"
    TEACHING_PREREQ = "teaching_prereq"
    RESPONDING = "responding"
    EXERCISING = "exercising"
    VERIFYING = "verifying"
    IDLE = "idle"
```

Each state transition emits events over WebSocket to the client.

---

## 4. Cross-Platform Test UI (Question 4)

### Strategy: Web-based PWA (not native Flutter for testing)

For testing on Mac, Android, iPad, iPhone — a **web page** is the fastest path:

| Approach     | Mac | Android | iPad | iPhone | Dev effort |
|-------------|-----|---------|------|--------|------------|
| Flutter app | Yes | Yes     | Yes  | Yes    | High (build + deploy per platform) |
| **Web PWA** | Yes | Yes     | Yes  | Yes    | Low (single HTML file, open in browser) |
| React Native| Yes | Yes     | Yes  | Yes    | Medium |

**Decision**: Use a single HTML+JS page with:
- **Web Speech API** (`SpeechRecognition`) for STT — works in Chrome (Mac/Android), Safari (iOS/iPad)
- **Web Speech API** (`SpeechSynthesis`) for TTS — all browsers
- **fetch()** to existing `/api/v1/query` endpoint
- Renders: teaching explanation, diagram spec, exercise

This lets you test voice interaction from ANY device with a browser, without installing anything.

### File location
`src/ui/voice_test.html` — served by FastAPI as a static file.

### Limitations of Web Speech API
- iOS Safari: requires user gesture to start recognition
- Chrome Android: good streaming support
- Firefox: limited SpeechRecognition support
- Tamil language support: varies by OS (Chrome uses Google servers, Safari uses Apple on-device)

### For production
Replace Web Speech API with:
- Google Cloud STT (streaming, math adaptation phrases)
- Google Cloud TTS (WaveNet Tamil voice)
- WebSocket protocol (as in Claude's `voice_server.py`)

---

## 5. Other Aspects for Voice-Based AI Math Tutor (Question 5)

### 5.1 Math pronunciation accuracy
- "36" must be spoken as "முப்பத்தி ஆறு" not "three six"
- "×" must be "பெருக்கல்" not "times"
- "÷" must be "வகுத்தல்" not "divided by"
- "பொ.கா.பெ." must be expanded to "பொதுக் காரணிகளுட் பெரியது"
- Build a Math SSML layer for TTS output

### 5.2 Student pace adaptation
- Grade 7 students think slowly about math
- VAD must handle 3-5 second pauses without cutting off
- If student says numbers, extend the silence timeout
- Speak teaching explanations at adjustable speed (0.8x for struggling students)

### 5.3 Noise resilience
- Classroom environments are noisy
- Use noise suppression (WebRTC noise suppression or RNNoise)
- Confidence thresholds should be higher in noisy environments
- Allow push-to-talk as fallback

### 5.4 Multimodal coordination
- When explaining via voice, simultaneously show the diagram on screen
- When student speaks an answer, show the recognized text for confirmation
- Allow students to switch between voice and text mid-conversation

### 5.5 Latency budget
| Step                    | Target    | Acceptable |
|-------------------------|-----------|------------|
| VAD → final transcript  | < 1.5s   | < 3.0s     |
| Transcript → first word | < 1.0s   | < 2.0s     |
| Full explanation done   | < 5.0s   | < 10.0s    |
| TTS first audio chunk   | < 0.5s   | < 1.5s     |
| Total voice→voice       | < 3.0s   | < 6.0s     |

### 5.6 Offline support
- Cache common teaching explanations as pre-synthesized audio
- On-device Whisper.cpp for STT when offline
- Android TTS engine (ta-IN locale) as TTS fallback
- Queue unanswered questions for when connectivity returns

### 5.7 Accessibility
- Screen reader compatibility
- Visual transcript display alongside voice
- Adjustable speech rate and volume
- High-contrast mode for diagram display

### 5.8 Safety and privacy
- Parental consent required for voice recording
- Audio is NOT stored by default (transcript only)
- Configurable retention policy per school/district
- Encrypted in transit (WSS) and at rest

### 5.9 Analytics specific to voice
- ASR Word Error Rate (WER) per dialect
- Re-prompt rate (how often agent asks for clarification)
- Voice vs text preference per student
- Average utterance length and complexity
- Dropout rate (student stops mid-conversation)

---

## 6. Comparison: Claude's Design vs This Architecture

| Aspect                  | Claude (voice-agent/)          | This architecture              |
|-------------------------|-------------------------------|-------------------------------|
| Pipeline layers         | 6 concrete layers             | Same 6 + conversation FSM     |
| Isolation               | Standalone WebSocket server   | Integrated into FastAPI app    |
| Conversation            | One-shot (speak→answer)       | Multi-turn FSM with clarify/prereqs |
| Test UI                 | Flutter widget (Android only) | Web PWA (all platforms)        |
| Orchestrator coupling   | Imports `agent_orchestrator`  | Uses `src.agents.orchestrator` |
| State management        | `VoiceSession` per WebSocket  | Session DB + conversation FSM  |
| Offline                 | Whisper.cpp + Android TTS     | Same + pre-cached audio        |
| Math SSML               | Detailed builder              | Adopted from Claude's design   |
| Dialect detection       | Acoustic + lexical features   | Same + student profile history |

**Recommendation**: Cherry-pick Claude's STT/TTS/VAD implementations, wrap them in the
conversation FSM, and integrate via the existing FastAPI server (not a separate WebSocket server).

---

## 7. Implementation Phases

### Phase 1: Web test UI (immediate)
- HTML+JS voice test page using Web Speech API
- Calls existing `/api/v1/query` endpoint
- Renders teaching + diagram + exercise
- Works on Mac/Android/iPad/iPhone via browser

### Phase 2: Conversation state machine
- Add `/api/v1/voice/session` endpoint with conversation FSM
- Clarification, fundamentals checking, exercise loop
- Session state persisted in DB

### Phase 3: Production STT/TTS
- Integrate Google Cloud STT v2 with math adaptation phrases
- Integrate Google Cloud TTS with Math SSML builder
- WebSocket streaming for real-time audio
- Cherry-pick from Claude's `voice_stt_tts.py`

### Phase 4: On-device/offline
- Silero VAD (from Claude's `voice_vad.py`)
- Whisper.cpp fallback
- Pre-cached audio phrases
- Offline queue for unanswered questions
