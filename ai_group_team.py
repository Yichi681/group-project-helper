#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Group Team Simulator
-----------------------

Goal:
  Help a *single* student simulate a group of different AI "teammates"
  (leader / critic / researcher) that read the course background and assignment
  requirements, argue for several rounds, and finally output a refined,
  comprehensive Markdown plan for the group assignment.

Key features:
  - Reads background info from ./Background_Information
  - Reads assignment requirements from ./Assignment_Requirement
  - Uses any OpenAI-compatible Chat Completions API (custom base URL + models)
  - Multi-agent, multi-round refinement:
      * Leader model: structure and synthesis
      * Critic model: finds weaknesses / missing parts
      * Researcher model: adds details, methods, data sources
  - Final result: a single polished Markdown document

Academic integrity:
  This script is meant as a planning and brainstorming assistant.
  You are responsible for:
    - verifying facts and methods,
    - adding your own reasoning and writing,
    - complying with your course / university policies on AI assistance.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import requests


# --------------------------- File helpers ------------------------------------


def load_directory_text(directory: str) -> str:
    """
    Load all text-like files (e.g., .txt, .md) from a directory, concatenate
    them, and label each block with the file name.

    Files that cannot be decoded as UTF-8 are silently skipped.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}. Please create it and add some text/markdown files."
        )

    parts: List[str] = []
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("."):
            continue
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            # Skip binary or non-UTF8 files
            continue
        if not content:
            continue
        parts.append(f"### {filename}\n\n{content}\n")

    if not parts:
        return "(no readable files found in this directory)"

    return "\n".join(parts)


# --------------------------- API call wrapper --------------------------------


class LLMError(RuntimeError):
    pass


def choose_api_key(explicit_key: str | None) -> str:
    """
    Determine which API key to use.

    Priority:
      1. --api-key CLI argument
      2. LLM_API_KEY
      3. OPENAI_API_KEY
      4. OPENROUTER_API_KEY
    """
    if explicit_key:
        return explicit_key

    for env_name in ("LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value

    raise LLMError(
        "No API key provided. Pass --api-key or set LLM_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY."
    )


def strict_or_loose_json_parse(raw: str) -> Dict[str, Any]:
    """
    Try to parse JSON strictly first. If that fails, try to extract the substring
    between the first '{' and the last '}' and parse that.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass
        raise


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.5,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """
    Call an OpenAI-compatible /chat/completions endpoint.

    - api_base: base URL like "https://api.openai.com/v1"
                or "https://openrouter.ai/api/v1"
    - model:    model name for that provider
    - messages: standard OpenAI-style messages list
    - json_mode: if True, send response_format={"type": "json_object"} to
                 request valid JSON output (if provider supports it).

    Returns the "content" string of the first choice.
    """
    url = api_base.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code != 200:
        raise LLMError(
            f"API request failed (HTTP {resp.status_code}). "
            f"Response: {resp.text[:1000]}"
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"Cannot parse completion response JSON: {data}") from e

    # Some providers may return content as a list of parts; join if needed.
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    if not isinstance(content, str):
        content = str(content)

    return content


# --------------------------- Multi-agent setup --------------------------------


@dataclass
class Agent:
    name: str      # internal name
    model: str     # model identifier for the provider
    role: str      # natural language role description


@dataclass
class APIConfig:
    api_base: str
    api_key: str
    temperature: float
    max_tokens: int | None


def make_default_agents(
    leader_model: str,
    critic_model: str,
    researcher_model: str,
) -> List[Agent]:
    """
    Build three default agents with different LLM models and personas.
    """
    return [
        Agent(
            name="Leader",
            model=leader_model,
            role=(
                "You act as the group leader. You are good at global structure, "
                "clarity, and turning messy ideas into a coherent Markdown outline."
            ),
        ),
        Agent(
            name="Critic",
            model=critic_model,
            role=(
                "You act as the critical teammate. You are good at spotting flaws, "
                "missing pieces, and weak reasoning. You always suggest stronger, "
                "sharper ways to structure the assignment."
            ),
        ),
        Agent(
            name="Researcher",
            model=researcher_model,
            role=(
                "You act as the detail-oriented researcher. You focus on concrete "
                "methods, data sources, case examples, and realistic execution steps."
            ),
        ),
    ]


# --------------------------- Agent behaviors ----------------------------------


def agent_initial_proposal(
    cfg: APIConfig,
    agent: Agent,
    background_text: str,
    assignment_text: str,
) -> Dict[str, Any]:
    """
    Ask one agent to produce an initial Markdown plan.

    Expected JSON output:
    {
      "summary": "short explanation of the plan",
      "markdown": "full Markdown outline"
    }
    """
    system_prompt = textwrap.dedent(f"""
    You are a virtual group member: {agent.name}.
    Role description: {agent.role}

    You are part of a 'solo student + multiple AI teammates' system.
    The goal is to help a single student plan and outline a high-quality
    group assignment, in the form of a clear Markdown document.

    OUTPUT FORMAT (IMPORTANT):
    - You MUST output ONLY valid JSON, nothing else.
    - No Markdown code fences, no extra commentary outside JSON.
    - JSON structure:
      {{
        "summary": "a one- or two-sentence summary of your idea",
        "markdown": "your proposed Markdown outline for the assignment"
      }}
    """).strip()

    user_prompt = textwrap.dedent(f"""
    Here is the course background and context:

    --- COURSE BACKGROUND (from ./Background_Information) ---
    {background_text.strip()}

    --- ASSIGNMENT REQUIREMENTS (from ./Assignment_Requirement) ---
    {assignment_text.strip()}
    ---------------------------------------------------------------

    Task for you (initial proposal):
    1. Design a feasible group assignment plan that a small student team could finish
       in 1–3 weeks (typical for a 'water' or medium-difficulty course project).
    2. Your output 'markdown' should be a structured outline, not a full essay:
       - Include a clear title for the project.
       - Use headings and subheadings (e.g. #, ##, ###) to show structure.
       - Under each section, use bullet points or short paragraphs to describe:
         * what needs to be done,
         * what data or materials are needed,
         * what the output of that part should look like.
    3. Try to align with the assignment requirements and grading logic.
    4. Do NOT assume multiple human group members; the actual goal is to help one
       student plus AI scale up to 'group-level' capability.

    Again: you must output valid JSON with keys 'summary' and 'markdown'.
    """)

    raw = call_chat_completion(
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model=agent.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        json_mode=True,
    )

    try:
        data = strict_or_loose_json_parse(raw)
    except Exception as e:
        raise LLMError(
            f"{agent.name} produced non-JSON output in initial proposal:\n{raw}"
        ) from e

    if not isinstance(data, dict) or "markdown" not in data:
        raise LLMError(
            f"{agent.name} initial JSON missing 'markdown' field: {data}"
        )

    return data


def agent_review_and_rewrite(
    cfg: APIConfig,
    agent: Agent,
    background_text: str,
    assignment_text: str,
    current_markdown: str,
    others_comments: List[Tuple[str, str]],
    round_index: int,
) -> Dict[str, Any]:
    """
    Ask one agent to critique the current draft and propose a revised version.

    Expected JSON output:
    {
      "critic":   "commentary on the current draft, its strengths and weaknesses",
      "markdown": "a fully rewritten, improved Markdown draft"
    }
    """
    if others_comments:
        others_block = "\n".join(
            f"- {name} said: {comment}" for name, comment in others_comments
        )
    else:
        others_block = "(No previous comments from other models for this round.)"

    system_prompt = textwrap.dedent(f"""
    You are virtual group member: {agent.name}.
    Role description: {agent.role}

    You are participating in a multi-round AI group discussion to refine
    a group assignment Markdown plan. In each round, you:

    1) Provide a focused critique ('critic') of the current shared draft.
    2) Provide a fully rewritten Markdown version ('markdown') that
       integrates your suggestions and tries to improve clarity, structure,
       completeness, and feasibility.

    OUTPUT FORMAT (IMPORTANT):
    - You MUST output ONLY valid JSON, nothing else.
    - JSON structure:
      {{
        "critic": "your review of the current draft",
        "markdown": "your improved Markdown draft"
      }}
    """).strip()

    user_prompt = textwrap.dedent(f"""
    COURSE BACKGROUND:
    ---
    {background_text.strip()}
    ---

    ASSIGNMENT REQUIREMENTS:
    ---
    {assignment_text.strip()}
    ---

    CURRENT ROUND: {round_index}
    CURRENT GROUP DRAFT (Markdown):
    >>> BEGIN CURRENT DRAFT
    {current_markdown.strip()}
    >>> END CURRENT DRAFT

    COMMENTS FROM OTHER MODELS IN THE PREVIOUS ROUND:
    {others_block}

    Your tasks:
    1. 'critic':
       - Point out where the current draft is unclear, incomplete, unrealistic,
         or misaligned with the assignment requirements.
       - Highlight any missing key sections or steps.
       - Keep it concrete and constructive, not just generic praise.
    2. 'markdown':
       - Produce a full Markdown draft that you personally believe is better.
       - You may reuse good parts of the existing draft but you can also
         restructure or rename sections if that improves clarity.
       - Focus on *what the student should actually do*, not on meta-comments.
    """)

    raw = call_chat_completion(
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model=agent.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        json_mode=True,
    )

    try:
        data = strict_or_loose_json_parse(raw)
    except Exception as e:
        raise LLMError(
            f"{agent.name} produced non-JSON output in review round {round_index}:\n{raw}"
        ) from e

    if (
        not isinstance(data, dict)
        or "markdown" not in data
        or "critic" not in data
    ):
        raise LLMError(
            f"{agent.name} review JSON missing 'critic' or 'markdown' in round {round_index}: {data}"
        )

    return data


def aggregate_markdowns(
    cfg: APIConfig,
    leader_agent: Agent,
    background_text: str,
    assignment_text: str,
    current_markdown: str,
    proposals: List[Tuple[Agent, Dict[str, Any]]],
    round_index: int,
) -> str:
    """
    Use the leader agent to aggregate multiple proposals into a single draft.

    Expected JSON output:
    {
      "markdown": "merged Markdown draft"
    }
    """
    merged_info_lines: List[str] = []
    for agent, data in proposals:
        critic = str(data.get("critic", "")).strip()
        markdown = str(data.get("markdown", "")).strip()
        merged_info_lines.append(
            f"=== {agent.name} ({agent.model}) - Critic ===\n"
            f"{critic}\n\n"
            f"=== {agent.name} ({agent.model}) - Markdown ===\n"
            f"{markdown}\n"
        )
    merged_info = "\n".join(merged_info_lines)

    system_prompt = textwrap.dedent(f"""
    You are the virtual group leader: {leader_agent.name}.
    Model: {leader_agent.model}

    Your responsibility is to read several alternative drafts and comments
    and synthesize them into a *single*, improved Markdown plan.

    OUTPUT FORMAT (IMPORTANT):
    - You MUST output ONLY valid JSON, nothing else.
    - JSON structure:
      {{
        "markdown": "your merged Markdown draft"
      }}

    General guidelines:
    - Keep the final draft well-structured and readable.
    - Minimize repetition, but do not lose important details.
    - Ensure the plan is realistic for one student (with AI help) within the
      given course/assignment constraints.
    """).strip()

    user_prompt = textwrap.dedent(f"""
    COURSE BACKGROUND:
    ---
    {background_text.strip()}
    ---

    ASSIGNMENT REQUIREMENTS:
    ---
    {assignment_text.strip()}
    ---

    CURRENT ROUND: {round_index}

    CURRENT SHARED DRAFT (from previous round):
    >>> BEGIN CURRENT DRAFT
    {current_markdown.strip()}
    >>> END CURRENT DRAFT

    BELOW ARE THE PROPOSALS AND CRITIQUES FROM ALL TEAMMATE MODELS IN THIS ROUND:

    {merged_info}

    Please produce a single merged Markdown draft according to the rules above.
    """)

    raw = call_chat_completion(
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model=leader_agent.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        json_mode=True,
    )

    try:
        data = strict_or_loose_json_parse(raw)
    except Exception as e:
        raise LLMError(
            f"Leader aggregation in round {round_index} returned non-JSON:\n{raw}"
        ) from e

    markdown = data.get("markdown")
    if not isinstance(markdown, str):
        raise LLMError(
            f"Leader aggregation JSON missing 'markdown' in round {round_index}: {data}"
        )

    return markdown


# --------------------------- Orchestration ------------------------------------


def multi_agent_group_process(
    api_cfg: APIConfig,
    agents: List[Agent],
    background_text: str,
    assignment_text: str,
    rounds: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    High-level orchestration:

    1. Each agent produces an initial proposal.
    2. Leader merges them into round-0 draft.
    3. For each subsequent round r:
       - Each agent critiques + rewrites.
       - Leader merges into new draft.
    4. Return final Markdown draft + debug info.

    Returns:
      (final_markdown, debug_info)
    """
    if rounds < 1:
        rounds = 1

    debug: Dict[str, Any] = {
        "agents": [a.__dict__ for a in agents],
        "initial_proposals": {},
        "rounds": [],
    }

    leader = agents[0]  # by convention: first agent is the leader

    # 1) Initial proposals
    print("== Initial proposals from all agents ==")
    initial_proposals: List[Tuple[Agent, Dict[str, Any]]] = []
    for agent in agents:
        print(f"  -> Asking {agent.name} ({agent.model}) for an initial outline...")
        data = agent_initial_proposal(
            cfg=api_cfg,
            agent=agent,
            background_text=background_text,
            assignment_text=assignment_text,
        )
        initial_proposals.append((agent, data))
        debug["initial_proposals"][agent.name] = data

    # 2) Initial aggregation
    print("  -> Leader aggregating initial proposals into round-0 draft...")
    current_markdown = aggregate_markdowns(
        cfg=api_cfg,
        leader_agent=leader,
        background_text=background_text,
        assignment_text=assignment_text,
        current_markdown="(no previous draft yet – this is the initial merge)",
        proposals=[
            # for the leader, treat initial proposals as if they had empty critics
            (agent, {"critic": proposal.get("summary", ""), "markdown": proposal["markdown"]})
            for agent, proposal in initial_proposals
        ],
        round_index=0,
    )

    # 3) Iterative improvement rounds
    for r in range(1, rounds + 1):
        print(f"\n== Iteration round {r} ==")
        round_record: Dict[str, Any] = {"index": r, "agents": {}}

        # figure out "previous comments" (critics) to show to each agent
        previous_comments: Dict[str, str] = {}
        if debug["rounds"]:
            last_round = debug["rounds"][-1]
            for name, info in last_round.get("agents", {}).items():
                previous_comments[name] = str(info.get("critic", ""))

        proposals: List[Tuple[Agent, Dict[str, Any]]] = []

        for agent in agents:
            other_comments = [
                (name, comment)
                for name, comment in previous_comments.items()
                if name != agent.name and comment
            ]
            print(f"  -> Round {r}: {agent.name} ({agent.model}) critiques and rewrites...")
            data = agent_review_and_rewrite(
                cfg=api_cfg,
                agent=agent,
                background_text=background_text,
                assignment_text=assignment_text,
                current_markdown=current_markdown,
                others_comments=other_comments,
                round_index=r,
            )
            proposals.append((agent, data))
            round_record["agents"][agent.name] = data

        print(f"  -> Round {r}: leader merges all proposals...")
        current_markdown = aggregate_markdowns(
            cfg=api_cfg,
            leader_agent=leader,
            background_text=background_text,
            assignment_text=assignment_text,
            current_markdown=current_markdown,
            proposals=proposals,
            round_index=r,
        )
        round_record["merged_markdown"] = current_markdown
        debug["rounds"].append(round_record)

    final_markdown = current_markdown
    return final_markdown, debug


# --------------------------- CLI ----------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a multi-model AI group to design a group assignment "
            "Markdown plan, using OpenAI-compatible APIs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="Base URL for an OpenAI-compatible API (e.g. https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional, otherwise taken from env: LLM_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--leader-model",
        type=str,
        default="openai/gpt-5.2",
        help="Model used by the leader agent.",
    )
    parser.add_argument(
        "--critic-model",
        type=str,
        default="x-ai/grok-4.1-fast",
        help="Model used by the critic agent.",
    )
    parser.add_argument(
        "--researcher-model",
        type=str,
        default="google/gemini-3-pro-preview",
        help="Model used by the researcher agent.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of discussion rounds between agents.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for all models.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Max tokens per completion (approximate upper bound, depends on provider).",
    )
    parser.add_argument(
        "--background-dir",
        type=str,
        default="./Background_Information",
        help="Directory with course background / context files.",
    )
    parser.add_argument(
        "--assignment-dir",
        type=str,
        default="./Assignment_Requirement",
        help="Directory with assignment requirement files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Final_Group_Report.md",
        help="Path to write the final Markdown output.",
    )
    parser.add_argument(
        "--dump-debug-json",
        type=str,
        default=None,
        help="Optional: path to save a JSON file with all intermediate steps.",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        api_key = choose_api_key(args.api_key)
    except LLMError as e:
        raise SystemExit(str(e))

    print("=== Loading background and assignment text ===")
    try:
        background_text = load_directory_text(args.background_dir)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    try:
        assignment_text = load_directory_text(args.assignment_dir)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print(f"- Background directory: {args.background_dir}")
    print(f"- Assignment directory: {args.assignment_dir}")

    agents = make_default_agents(
        leader_model=args.leader_model,
        critic_model=args.critic_model,
        researcher_model=args.researcher_model,
    )

    print("\n=== Agent configuration ===")
    for a in agents:
        print(f"* {a.name}: model={a.model}")
    print(f"* API base: {args.api_base}")
    print(f"* Rounds:   {args.rounds}")
    print("=============================================\n")

    api_cfg = APIConfig(
        api_base=args.api_base,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    try:
        final_md, debug_info = multi_agent_group_process(
            api_cfg=api_cfg,
            agents=agents,
            background_text=background_text,
            assignment_text=assignment_text,
            rounds=args.rounds,
        )
    except LLMError as e:
        raise SystemExit(f"Error while talking to the LLMs: {e}")

    # Write final Markdown
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(final_md)

    print(f"\n✅ Finished. Final Markdown written to: {args.out}")

    # Optional debug dump
    if args.dump_debug_json:
        with open(args.dump_debug_json, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)
        print(f"🔍 Debug info written to: {args.dump_debug_json}")


if __name__ == "__main__":
    main()
