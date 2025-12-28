# AI Group Team Simulator

A multi-agent AI tool that simulates a **group of teammates** using different large language models (LLMs), allowing **one single user** to achieve group-level planning and ideation capability for academic group assignments.

This project is designed to reduce coordination overhead in group work and to help a single student:
- structure complex assignments,
- identify missing parts and weaknesses,
- design realistic execution plans,
- and generate a refined Markdown outline through multi-round AI discussion.

---

## ✨ Core Idea

> **One human + multiple AI models = a virtual group**

Instead of real group members, this tool assigns different roles to different AI models:

- **Leader** – focuses on global structure and synthesis  
- **Critic** – challenges assumptions and finds weaknesses  
- **Researcher** – adds concrete methods, data sources, and execution details  

They:
1. Read the same course background and assignment requirements  
2. Independently propose plans  
3. Critique and rewrite each other’s drafts over multiple rounds  
4. Produce a final, integrated Markdown document  

---

## 📁 Project Structure

```

.
├── ai_group_team.py
├── Background_Information/
│   ├── syllabus.md
│   ├── lecture_notes.txt
│   └── ...
├── Assignment_Requirement/
│   ├── assignment_prompt.txt
│   ├── rubric.md
│   └── ...
├── README.md
└── README_CN.md

````

### Background_Information/
Put **course-level context** here:
- syllabus
- lecture notes
- key concepts
- methodological guidance

### Assignment_Requirement/
Put **task-specific requirements** here:
- assignment description
- grading rubric
- templates / required sections
- constraints (word count, format, deadline)

All readable text files (`.txt`, `.md`) in these folders will be loaded and provided to the AI models.

---

## 🔧 Requirements

- Python 3.9+
- One dependency:
```bash
pip install requests
````

---

## 🔑 API Configuration

This tool supports **any OpenAI-compatible API**.

### Supported API keys (priority order):

1. `--api-key` CLI argument
2. `LLM_API_KEY`
3. `OPENAI_API_KEY`
4. `OPENROUTER_API_KEY`

Example (recommended):

```bash
export LLM_API_KEY="your_api_key_here"
```

---

## 🚀 Quick Start (OpenRouter Example)

Using OpenRouter with three different models:

* `openai/gpt-5.2` (Leader)
* `x-ai/grok-4.1-fast` (Critic)
* `google/gemini-3-pro-preview` (Researcher)

```bash
python ai_group_team.py \
  --api-base https://openrouter.ai/api/v1 \
  --rounds 3 \
  --out Final_Group_Report.md
```

Output:

* `Final_Group_Report.md` – final refined Markdown plan

---

## ⚙️ Custom Models / Custom API

### OpenAI official API example

```bash
python ai_group_team.py \
  --api-base https://api.openai.com/v1 \
  --leader-model gpt-4.1 \
  --critic-model gpt-4.1-mini \
  --researcher-model gpt-4.1-mini \
  --rounds 2 \
  --out Final_Group_Report.md
```

### Self-hosted or gateway-compatible API

```bash
python ai_group_team.py \
  --api-base https://your-gateway.example.com/v1 \
  --api-key YOUR_KEY \
  --leader-model model-a \
  --critic-model model-b \
  --researcher-model model-c \
  --rounds 4 \
  --out result.md
```

---

## 🧠 Recommended Workflow

1. Put all course background into `Background_Information/`
2. Put assignment details into `Assignment_Requirement/`
3. Run the script to generate a structured Markdown plan
4. Manually:

   * verify facts
   * add references
   * rewrite in your own academic voice
   * convert into report / PPT / proposal

---

## ⚠️ Academic Integrity Notice

This tool:

* **does NOT replace thinking**
* **does NOT guarantee factual correctness**
* **should NOT be used to directly submit AI output as original work**

It is intended for:

* brainstorming
* outlining
* structural planning
* self-review from multiple perspectives

Always follow your institution’s AI usage policies.

---

## 📜 License

This project is provided for educational and research assistance purposes only.
No warranty is provided.
