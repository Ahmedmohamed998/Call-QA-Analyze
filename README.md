# Call QA Analyzer

> AI-powered call quality analysis system for healthcare clinic operations.

An API service that analyzes phone call transcripts from a healthcare clinic's phone system, detecting compliance issues (HIPAA violations, misinformation, rudeness), evaluating agent performance, and making intelligent escalation decisions — all while avoiding false positives and punitive scoring.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Prompting Strategy](#prompting-strategy)
- [Edge Case Handling](#edge-case-handling)
- [LLM Provider Abstraction](#llm-provider-abstraction)
- [API Reference](#api-reference)
- [Sample Transcripts](#sample-transcripts)
- [Tradeoffs](#tradeoffs)
- [Running Tests](#running-tests)

---

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM API key (Azure OpenAI or AWS Bedrock Claude)

### Setup

```bash
# 1. Clone and enter the project
cd call-qa-analyzer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API credentials

# 5. Run the server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive Swagger documentation.

### Quick Test

```bash
curl -X POST http://localhost:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d @sample_transcripts/clean_call.json
```

---

## Architecture

```
app/
├── main.py                        # FastAPI app with lifespan management
├── config.py                      # pydantic-settings configuration
├── models/
│   ├── request.py                 # Input validation (CallTranscript)
│   └── response.py                # Output schema (QualityAnalysis)
├── providers/
│   ├── base.py                    # Abstract LLM provider interface
│   ├── azure_openai_provider.py   # Azure OpenAI implementation
│   └── bedrock_claude_provider.py # AWS Bedrock Claude implementation
├── services/
│   ├── prompt_builder.py          # System/user prompt construction
│   └── analyzer.py                # Core orchestration + post-processing
├── middleware/
│   └── observability.py           # Structured logging + request tracing
└── routes/
    └── analyze.py                 # API endpoint handlers
```

### Design Principles

1. **Separation of Concerns**: Each module has a single responsibility. Models define data shapes, providers handle LLM communication, services contain business logic, and routes handle HTTP concerns.

2. **Provider Abstraction**: The `LLMProvider` abstract base class defines a simple interface (`analyze()`) that both Azure OpenAI and Bedrock Claude implement. Swapping providers requires only changing the `LLM_PROVIDER` value in `.env`.

3. **Pydantic Everywhere**: Input validation, output schema enforcement, LLM structured outputs, and configuration are all managed through Pydantic models. This ensures type safety and schema compliance at every layer.

4. **Deterministic Post-Processing**: Even with structured outputs, LLMs can be inconsistent with business logic. The `CallAnalyzer._post_process()` method applies deterministic rules (e.g., "critical flag → escalation required") to guarantee consistency.

---

## Prompting Strategy

The prompting strategy is the core of this system. It's designed around three key goals:

### 1. Prevent False Positives

The system prompt explicitly instructs the LLM:
- **"ONLY flag issues that are CLEARLY and EXPLICITLY present in the transcript"** — prevents the LLM from inventing issues
- **"Do NOT assume the worst"** — handles ambiguous situations fairly
- **"escalate is RESERVED for genuinely critical issues ONLY"** — prevents over-escalation of minor issues
- **"Every transcript_excerpt you cite MUST be an actual quote"** — prevents fabricated evidence

### 2. Handle Caller vs. Agent Behavior

A critical distinction: only the **agent's** behavior should be evaluated. The prompt explicitly states:
- Evaluate agent conduct, not caller conduct
- If a caller is rude but the agent handles it well, flag it as a "positive_interaction"
- Never penalize the agent for a difficult caller

### 3. Calibrated Scoring

The scoring guidance provides concrete anchors:
- `professionalism_score`: 0.7+ = professional, <0.5 only for clear rudeness
- `accuracy_score`: Default 0.85 if no verifiable claims made, <0.5 only for clear misinformation
- `resolution_score`: Based on whether the caller's actual issue was addressed

### 4. Department-Specific Rules

The prompt builder injects additional evaluation criteria based on the department:

| Department | Key Checks |
|---|---|
| **Scheduling** | Appointment confirmation, identity verification, alternative times |
| **Onboarding** | Lien agreement discussion, insurance verification, welcome info |
| **Helpdesk** | Issue resolution, proper escalation of medical questions |
| **Follow-Ups** | Appointment confirmation, identity verification, callbacks |
| **Records** | Identity verification before records discussion, authorization |

---

## Edge Case Handling

| Scenario | How It's Handled |
|---|---|
| **Very short call** (<30s, <4 exchanges) | Recognized as likely hang-up/wrong number. Assessed as "pass" unless clear issue present. Agent not penalized. |
| **No issues found** | Empty compliance_flags (or positive_interaction only). Overall assessment: "pass". This is the expected outcome for most calls. |
| **Ambiguous transcript** | Ambiguity noted in assessment_reasoning. Not flagged as a compliance issue. "needs_review" only if ambiguity obscures potentially important issues. |
| **Caller is rude** | Caller behavior ignored for scoring. If agent handles it well, flagged as "positive_interaction". |
| **Minor imperfections** | Listed under `improvements` in agent_performance, NOT as compliance flags. These are coaching opportunities, not violations. |
| **Medical questions to agent** | Different handling by department: Helpdesk agents should route to clinical staff, not provide medical advice. |

---

## LLM Provider Abstraction

The system supports two LLM providers, swappable via environment configuration:

### Azure OpenAI

- Uses `AsyncAzureOpenAI` client from the `openai` SDK
- Structured outputs via the SDK's `beta.chat.completions.parse()` method, which natively accepts Pydantic models as `response_format` and handles all schema conversion internally
- Guarantees schema compliance at the API level

### AWS Bedrock Claude

- Uses `boto3` Bedrock Runtime with the Converse API
- Structured outputs via the **tool-use pattern**: the Pydantic schema is defined as a tool, and Claude returns the analysis as a tool call
- Extracts structured JSON from the tool call result

### Swapping Providers

```bash
# In .env, change:
LLM_PROVIDER=azure_openai    # → Azure OpenAI (GPT)
LLM_PROVIDER=bedrock_claude  # → AWS Bedrock (Claude)
```

No code changes required. The factory function in `main.py` instantiates the correct provider.

### Retry Logic

Both providers implement automatic retry with exponential backoff:
- **3 attempts** with delays of 1s → 4s → 16s
- Retries on: API errors, timeouts, rate limits
- Logs each retry attempt with the error for observability

---

## API Reference

### `POST /analyze-call`

Analyze a single call transcript.

**Request Body:**
```json
{
  "call_id": "CALL-2024-00142",
  "agent_name": "Maria Santos",
  "call_date": "2024-11-15",
  "call_duration_seconds": 245,
  "department": "Scheduling",
  "transcript": "Agent: Hello...\nCaller: Hi..."
}
```

**Response (200):**
```json
{
  "overall_assessment": "pass",
  "assessment_reasoning": "The agent handled the scheduling request professionally...",
  "compliance_flags": [
    {
      "type": "positive_interaction",
      "severity": "positive",
      "description": "Agent greeted the caller warmly and verified identity.",
      "transcript_excerpt": "Can I please have your full name and date of birth?"
    }
  ],
  "agent_performance": {
    "professionalism_score": 0.95,
    "accuracy_score": 0.90,
    "resolution_score": 0.95,
    "strengths": ["Warm greeting", "Identity verification", "Clear confirmation"],
    "improvements": ["Could offer additional appointment options"]
  },
  "escalation_required": false,
  "escalation_reason": null
}
```

### `POST /batch-analyze`

Analyze multiple transcripts concurrently.

**Request Body:**
```json
{
  "transcripts": [
    { "call_id": "...", ... },
    { "call_id": "...", ... }
  ]
}
```

### `GET /health`

Health check endpoint. Returns `{"status": "healthy"}`.

---

## Sample Transcripts

Five sample transcripts are included in `sample_transcripts/`:

| File | Scenario | Expected Outcome |
|---|---|---|
| `clean_call.json` | Professional scheduling call with proper verification | **pass** — no issues |
| `problematic_call.json` | HIPAA violation + rudeness + dismissiveness | **escalate** — critical flags |
| `edge_case_short.json` | 12-second wrong number call | **pass** — too short for issues |
| `edge_case_no_issues.json` | Normal call, slightly informal, minor hesitation | **pass** — imperfections noted in improvements only |
| `scheduling_call.json` | Scheduling without confirming final details | **needs_review** — minor protocol violation |

---

## Tradeoffs

### 1. Structured Outputs vs. Prompt-Based JSON

**Choice:** Structured outputs (OpenAI `beta.parse()` with Pydantic models / Claude tool-use pattern)

**Why:** In healthcare, we cannot tolerate unparseable responses. Structured outputs guarantee schema compliance at the API level, eliminating JSON parsing failures. The tradeoff is slightly reduced flexibility in the LLM's response format, but reliability is paramount.

### 2. Single LLM Call vs. Multi-Step Analysis

**Choice:** Single LLM call per transcript

**Why:** A multi-step approach (one call for flag detection, another for scoring, another for assessment) would provide more granular control but doubles/triples latency and cost. For a QA system processing thousands of calls, the single-call approach with strong prompting is the right balance. The post-processing layer catches any business logic inconsistencies.

### 3. Post-Processing Business Rules

**Choice:** Deterministic post-processing after LLM output

**Why:** LLMs are probabilistic — they might flag a critical HIPAA violation but forget to set `escalation_required = true`. Rather than relying entirely on the LLM for business logic consistency, deterministic rules enforce invariants like "critical flag → escalation required." This hybrid approach gets the best of both worlds: LLM's analytical capability + deterministic reliability.

### 4. Temperature Setting

**Choice:** Very low temperature (0.1 for Bedrock Claude; default for Azure OpenAI as GPT-5.3 does not support custom temperature values)

**Why:** For quality analysis, we want consistency and reliability over creativity. The same transcript analyzed twice should produce similar results. Low temperature provides near-deterministic outputs while allowing minor variation in phrasing.

### 5. Synchronous Bedrock Client

**Choice:** boto3 (synchronous) wrapped in async interface

**Why:** boto3 doesn't offer a native async client. For a single-call endpoint this is acceptable — the I/O wait is dominated by the LLM response time, not the HTTP call overhead. For true high-throughput batch processing, migrating to an async HTTP client (like `aioboto3`) would be recommended.

---

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v
pytest tests/test_analyzer.py -v
```

### Evaluation Script

The evaluation script sends all sample transcripts to the running API and validates outcomes:

```bash
# 1. Start the server (in one terminal)
uvicorn app.main:app --reload

# 2. Run evaluation (in another terminal)
python -m tests.evaluate
```

The evaluation checks:
- Every response conforms to the Pydantic output schema
- Clean call → "pass"
- Problematic call → "escalate" with HIPAA + rudeness flags
- Short call → "pass"
- No-issues call → not "escalate"

---

## Observability

Every LLM call is logged with:
- **Prompt content** (system + user) at DEBUG level
- **Response content** at DEBUG level
- **Latency** in milliseconds
- **Token usage** (prompt tokens, completion tokens, total)
- **Model/deployment** used
- **Request ID** for correlation across log entries

Example log output:
```
2024-11-15 14:30:22 | INFO     | app.providers.azure_openai | Azure OpenAI call completed | latency=2341ms | prompt_tokens=1247 | completion_tokens=892 | total_tokens=2139
2024-11-15 14:30:22 | INFO     | app.services.analyzer      | Analysis complete | call_id=CALL-001 | assessment=pass | flags=1 | escalation=False
```
