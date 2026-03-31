## Developer Testing (E2E) — NIE Tamil Math Tutor

This document provides repeatable steps to run the FastAPI server and execute end-to-end tests for:
- LLM explanation generation (Tamil-only + method-only enforcement)
- Adaptive retrieval behavior (vector DB + keyword fallback)
- Diagrammatic responses (Stage-2 auto-trigger + method-aligned diagram selection)

### Prerequisites
1. Python deps installed (already reflected in `requirements.txt`).
2. Gemini API key available in your environment or `.env` (used by `LLMClient`).
3. Optional (recommended): BGE-M3 embeddings + ChromaDB ingestion already completed.

### Start the server
From repository root:
```bash
cd /Users/yogi/projects/ai-mathematics
EMBED_OFFLINE_ONLY=1 python3 -m uvicorn src.api.server:app --host 127.0.0.1 --port 8001
```

Then confirm health:
```bash
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
```
Expected:
`{"status":"ok",...}`

### (One-time) Ensure NIE corpus is ingested into ChromaDB
If you see server logs like `Collection is empty — run ingestion first`, run this one-time ingestion command:
```bash
cd /Users/yogi/projects/ai-mathematics
EMBED_OFFLINE_ONLY=1 python3 - <<'PY'
import hashlib
from src.ingestion.vector_store import NIEVectorStore, TamilEmbedder, ChunkMetadata
from src.data.nie_corpus import NIE_CORPUS

embedder = TamilEmbedder()
store = NIEVectorStore()

texts = []
metas = []
for chunk in NIE_CORPUS:
    content = (chunk.get('content_ta') or '').strip()
    if not content:
        continue

    meta = ChunkMetadata(
        chunk_id=chunk['id'],
        grade=7,
        chapter=4,
        subject='mathematics',
        section=chunk.get('section', ''),
        topic=chunk.get('topic', ''),
        chunk_type=chunk.get('type', 'concept'),
        difficulty=chunk.get('difficulty', 1),
        page_start=chunk.get('page', 0),
        page_end=chunk.get('page', 0),
        prerequisites=chunk.get('prerequisites', []),
        diagram_types=[chunk['diagram_trigger']] if chunk.get('diagram_trigger') else [],
        nie_terms=list((chunk.get('key_terms') or {}).keys()),
        has_numbers=any(c.isdigit() for c in content),
        is_answer_scheme=False,
        language='tamil',
        source_file='nie_corpus.py',
        checksum=hashlib.sha256(content.encode()).hexdigest(),
    )
    texts.append((content, meta))
    metas.append(content)

embeddings = embedder.embed_batch(metas)
store.upsert_chunks(texts, embeddings, grade=7, chapter=4, subject='mathematics')
print("Ingestion complete.")
PY
```

Restart the server after ingestion.

### Test API: query endpoint
The main endpoint is:
- `POST /api/v1/query`

Payload fields:
- `student_id` (string)
- `question` (string)  <-- important: it is NOT named `query`
- `district` (string)
- `student_name` (optional)

### Test Matrix

#### T01 — Stage-2 auto-trigger diagram (no explicit காட்டு/வரை)
Purpose:
- Ensure diagrams are auto-shown even when user does not say “காட்டு/வரை”.
- Ensure retrieved method chunks influence scaffold and diagram style.

Request:
```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "student_id":"t01",
    "question":"24, 60, 90 இன் பொ.கா.பெ. காண்க",
    "district":"jaffna"
  }'
```

Assertions (manual):
- `intent` should be `EXPLAIN`
- `retrieved_chunk_ids` should include at least one method chunk ID, typically one of:
  - `M04` (HCF Method I) or `M01` (factor listing pair) depending on retrieval
- `diagram` should NOT be `null`
- `diagram.diagram_type` should be one of:
  - `factor_pairs` (Method I)
  - `factor_tree` (Method II)
  - `division_ladder` (Method III)
- `teaching.explanation_ta` should contain method-scaffold style text, e.g. Method I scaffold lines such as:
  - “ஒவ்வொரு எண்ணின் காரணிகளைப் பட்டியலிடுவோம்” or other method-specific NIE phrasing

#### T02 — Explicit Method III request forces Division Ladder
Purpose:
- Verify method-only enforcement + diagram alignment.

Request:
```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "student_id":"t02",
    "question":"6, 12, 18 இன் பொ.கா.பெ. காண்க (வகுத்தல் முறை)",
    "district":"batticaloa"
  }'
```

Assertions:
- `intent` should be `SHOW_METHOD` (or at least not `UNKNOWN`)
- `retrieved_chunk_ids` should include `M06` (or another Method III chunk)
- `diagram.diagram_type` must be `division_ladder`
- `teaching.explanation_ta` must use the “வகுத்தல் ஏணி” style (look for “வகுத்தல்” / “வகுத்தல் ஏணி” wording and stop conditions like “1”)

#### T03 — Explicit Method II request forces Factor Tree
Purpose:
- Verify method-only enforcement + diagram alignment for Method II.

Request:
```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "student_id":"t03",
    "question":"6, 12, 18 இன் பொ.கா.பெ. காண்க (காரணி மரம் முறை)",
    "district":"colombo"
  }'
```

Assertions:
- `diagram.diagram_type` must be `factor_tree`
- `teaching.explanation_ta` should reflect prime factorization tree style (“முதன்மைக் காரணிகள்”, “காரணி மரம்” wording)

#### T04 — Method I request forces Factor Pairs
Purpose:
- Verify method-only enforcement + diagram alignment for Method I.

Request:
```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "student_id":"t04",
    "question":"6, 12, 18 இன் பொ.கா.பெ. காண்க (பட்டியல் முறை)",
    "district":"estate"
  }'
```

Assertions:
- `diagram.diagram_type` must be `factor_pairs`
- `teaching.explanation_ta` should include factor-pair/listing procedure (“காரணிகள்: …” style + pair listing logic)

#### T05 — Diagrammatic response shape checks (schema sanity)
Purpose:
- Ensure `diagram.spec` contains required fields by diagram type.

How:
1. Re-run T01, T02, T03, T04.
2. For each response, inspect `diagram.diagram_type` and `diagram.spec`.

Manual checks:
- If `diagram_type == "factor_pairs"`:
  - `spec.pairs` exists and is a list of pairs
  - `spec.all_factors` exists
- If `diagram_type == "factor_tree"`:
  - `spec.root` exists
  - `spec.tree` exists
  - `spec.prime_factors` exists
- If `diagram_type == "division_ladder"`:
  - `spec.steps` exists
  - `spec.hcf_value` exists

#### T06 — Tamil-only output guard (quick manual)
Purpose:
- Quickly check “Tamil-only” policy wasn’t violated.

How:
- Look for ASCII English words inside `teaching.explanation_ta`.
- Ensure you don’t see banned terms like `factor`, `remainder`, `whole number`, or English letters `A–Z`.

If you find violations:
- Log the failing `question`, and the response `retrieved_chunk_ids`, then report it for prompt/system constraints tuning.

### What to record when filing a bug
Include:
- The exact `question` payload
- `intent`
- `retrieved_chunk_ids`
- `diagram.diagram_type`
- A 20–50 line excerpt of `teaching.explanation_ta`
- Whether the diagram appeared without explicit “காட்டு/வரை” (yes/no)

### Automated Presentation + Validation (Optional)

The repo includes helper scripts under `tests/scripts/` to:
- Render a “lesson view” in the terminal (Tamil explanation + diagram + diagram spec)
- Generate a simple local HTML preview (open in browser)
- Validate method alignment for Methods I/II/III (diagram type + diagram spec shape + Tamil-only guard)

#### Run the method-alignment matrix end-to-end
Requires:
- FastAPI server running
- Gemini API key configured
- (Recommended) NIE corpus ingested into ChromaDB

Run:
```bash
cd /Users/yogi/projects/ai-mathematics
python3 tests/scripts/run_method_matrix.py \
  --base-url http://127.0.0.1:8001 \
  --out-dir tests/results/matrix_run
```

It will produce:
- `tests/results/matrix_run/T01_auto_diagram_method_any.json` (and others)
- `tests/results/matrix_run/T01_auto_diagram_method_any.html` (and others)
- Console output with `Validation: OK/FAIL` and a terminal lesson view.
- Note: if you hit Gemini rate limits (`429 RESOURCE_EXHAUSTED`), re-run with fewer scenarios, e.g. `--max-scenarios 1` (validate Method I/II/III one-by-one).
- Tip: you can also select exact scenarios via `--only`, e.g. `--only T04_method1_factor_pairs`.

#### Render an already-received JSON response
Terminal view:
```bash
python3 tests/scripts/render_terminal_lesson.py --json /path/to/response.json
```

HTML preview:
```bash
python3 tests/scripts/render_html_lesson.py \
  --json /path/to/response.json \
  --out /path/to/response-preview.html
```

---

### Voice Input Testing

#### Open the Voice Test UI

The server serves a cross-platform voice test UI at `/voice`. After starting the server:

```
http://127.0.0.1:8001/voice
```

This works on:
- **Mac**: Chrome (best), Safari, Firefox
- **Android**: Chrome (full voice support)
- **iPad/iPhone**: Safari (tap mic button to start — iOS requires user gesture)

#### Features

| Feature                | How it works                                |
|------------------------|---------------------------------------------|
| Voice input (STT)      | Web Speech API (`ta-IN` locale)            |
| Text input             | Type in the text box and press Enter        |
| TTS playback           | Click speaker button to hear the response   |
| Speech rate control    | Slider in settings bar (0.5x–1.5x)         |
| District selection     | Dropdown in settings bar                    |
| Diagram display        | JSON spec shown in diagram card             |
| Exercise display       | Exercise card with hint                     |

#### Voice Test Scenarios

##### V01 — Basic Tamil voice query
1. Open the voice UI in Chrome
2. Click the mic button (red pulse animation appears)
3. Say: "காரணிகள் என்றால் என்ன?" (What are factors?)
4. Wait for the response to appear
5. Click the speaker button to hear the response

Expected:
- Transcript appears as a student message
- Teaching explanation appears in Tamil
- Speaker button reads the explanation aloud

##### V02 — Voice with method request
1. Click mic
2. Say: "12 மற்றும் 18 இன் பொ.கா.பெ. காரணி மரம் முறையில் காண்க"
3. Wait for response

Expected:
- Diagram card appears with `factor_tree` type
- Teaching uses Method II scaffold

##### V03 — Cross-device test
1. Find your Mac's local IP: `ifconfig | grep 'inet '`
2. Update the API URL in the settings bar to `http://<your-mac-ip>:8001`
3. Open `http://<your-mac-ip>:8001/voice` on your phone/tablet
4. Test voice input on mobile

Note: On Android Chrome, voice recognition works well.
On iOS Safari, you must tap the mic button (no auto-start), and Tamil recognition depends on iOS language pack.

##### V04 — Multi-turn voice conversation API
The `/api/v1/voice/converse` endpoint supports multi-turn conversations:

```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/voice/converse \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "SL_TM_VOICE_001",
    "transcript": "மீ.பொ.கா. என்றால் என்ன?",
    "confidence": 0.9,
    "district": "jaffna",
    "session_key": "test_session_1"
  }'
```

Test low-confidence clarification:
```bash
curl -s -X POST http://127.0.0.1:8001/api/v1/voice/converse \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "mumble mumble",
    "confidence": 0.3,
    "session_key": "test_session_2"
  }'
```

Expected: Agent asks the student to repeat the question.

#### Troubleshooting voice issues

| Issue                                   | Fix                                                  |
|-----------------------------------------|------------------------------------------------------|
| Mic button does nothing                 | Check browser permissions (Settings → Microphone)    |
| "SpeechRecognition not available"       | Use Chrome (Firefox has limited support)             |
| Tamil not recognized                    | Install Tamil language pack on device/OS              |
| No sound on TTS                         | Check device volume; some browsers block autoplay     |
| CORS error on mobile                    | Server already has `allow_origins=["*"]`              |
| iOS Safari mic blocked                  | Must be HTTPS in production; for local testing use Chrome |

