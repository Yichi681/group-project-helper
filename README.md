# Groupwork Ultra Agent

Groupwork Ultra Agent is a **local, multi-agent AI orchestration system** designed to simulate
a full university group-project workflow — enabling a **single student** to reach
the organizational and analytical capacity of a real multi-person group.

This project focuses on **process automation, deliberation quality, and synthesis**,
rather than blind content generation.

---

## 1. Motivation

Group assignments often suffer from:
- Coordination friction
- Uneven contribution
- Awkward or superficial discussion
- Time wasted on logistics instead of thinking

At the same time, modern AI models excel at:
- Structured reasoning
- Draft comparison
- Iterative refinement
- Role-based critique

**Groupwork Ultra Agent combines these strengths** to simulate a realistic group process,
where different “members” (AI agents) play distinct roles and debate across multiple rounds.

---

## 2. Core Idea

> **One human + multiple AI agents = comparable capacity to a real student group**

Each AI agent:
- Uses a **different model**
- Has a **clear role**
- Produces structured outputs
- Participates in **multi-round deliberation**

A dedicated **Controller model** evaluates the discussion quality and decides
whether more rounds are needed.

---

## 3. System Architecture

### 3.1 Agent Roles

Typical agents include:

- **Leader**
  - Merges proposals
  - Resolves conflicts
  - Produces consolidated outputs

- **Researcher**
  - Grounds ideas in provided documents
  - Focuses on feasibility and data sources

- **Methodologist**
  - Evaluates research design and structure
  - Flags weak logic or missing components

- **Critic**
  - Challenges assumptions
  - Identifies risks and gaps

- **Editor**
  - Improves clarity, coherence, and presentation quality

- **Red Team (optional)**
  - Stress-tests arguments
  - Looks for grading or academic risks

- **Controller (Judge Model)**
  - Evaluates whether the discussion has converged
  - Decides: continue another round or stop

---

## 4. Multi-Round Deliberation Loop

Each deliberation stage (topics, tasks, final draft) follows this loop:

1. Each agent independently proposes or refines outputs (JSON-structured)
2. Leader agent merges and deduplicates
3. Controller model evaluates:
   - Coverage completeness
   - Internal consistency
   - Alignment with assignment requirements
4. If quality is insufficient → next round with focused instructions
5. Loop stops only when Controller approves

This prevents:
- Premature convergence
- Single-model bias
- Shallow first-pass answers

---

## 5. Retrieval-Augmented Generation (RAG)

The system supports **local document ingestion**:

### Supported formats
- PDF
- PPTX
- DOCX / DOC
- TXT
- Markdown

### Typical sources
- Course slides
- Assignment briefs
- Rubrics
- Lecture notes
- Background readings

Documents are:
- Parsed
- Chunked
- Indexed locally (SQLite + FTS)
- Referenced during deliberation

Agents are explicitly instructed **not to invent citations or facts**.

---

## 6. Supported Models

Via **OpenRouter-compatible API**:

Examples:
- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- `x-ai/grok-4.1-fast`
- `google/gemini-3-pro-preview`

Different agents can use **different models** simultaneously.

---

## 7. Workflow Overview

```text
init
 ├─ ingest (documents)
 ├─ propose-topics (multi-agent deliberation)
 ├─ generate-tasks
 ├─ generate-final
 └─ export / serve
````

### Autopilot Mode

Runs the entire pipeline automatically with minimal human input.

---

## 8. Example Usage

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=your_api_key

python main.py init \
  --name "AI Groupwork" \
  --members "Leader,Researcher,Critic,Editor" \
  --leader-model openai/gpt-oss-20b \
  --critic-model openai/gpt-oss-120b \
  --controller-model openai/gpt-oss-120b

python main.py ingest <project_id>
python main.py propose-topics <project_id> --mode autopilot
python main.py generate-tasks <project_id> --mode autopilot
python main.py generate-final <project_id>
```

---

## 9. Ethics & Academic Integrity

This project is **not a ghostwriting tool**.

It is intended for:

* Structuring ideas
* Exploring alternatives
* Improving organization and clarity
* Simulating discussion and critique

Users are responsible for:

* Verifying facts
* Citing sources properly
* Complying with course AI usage policies

---

## 10. Intended Use Cases

* Humanities and social science group projects
* Business and management courses
* Proposal drafting and report structuring
* PPT outline generation
* Learning how high-quality group discussions are structured

---

## 11. Disclaimer

This tool automates **process and coordination**, not academic responsibility.
Final submissions remain the user’s responsibility.
