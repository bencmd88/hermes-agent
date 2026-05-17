#!/usr/bin/env python3
"""
Skill Evolution Script

Evolves a hermes-agent skill through iterative evaluation and improvement
using an LLM as both evaluator and editor.

Usage:
    # Evolve using synthetic (LLM-generated) evaluation cases
    python -m evolution.skills.evolve_skill \\
        --skill github-code-review \\
        --iterations 10 \\
        --eval-source synthetic

    # Evolve using real session history from the Hermes SessionDB
    python -m evolution.skills.evolve_skill \\
        --skill github-code-review \\
        --iterations 10 \\
        --eval-source sessiondb

Environment variables:
    HERMES_AGENT_REPO   Path to the hermes-agent repo (defaults to the repo
                        containing this file). Skills are searched for in
                        <HERMES_AGENT_REPO>/skills/ first, then
                        ~/.hermes/skills/.
    HERMES_HOME         Path to the Hermes home directory (default: ~/.hermes).
                        The SessionDB lives at $HERMES_HOME/state.db.

    One of the following API key env vars must be set to drive the LLM calls:
        OPENAI_API_KEY          → uses OpenAI (gpt-4o by default)
        ANTHROPIC_API_KEY       → uses Anthropic (claude-opus-4.5 by default)
        OPENROUTER_API_KEY      → uses OpenRouter (openai/gpt-4o by default)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo / skill discovery
# ---------------------------------------------------------------------------

# Allow override via env var (used by the self-evolution sister-repo).
_REPO_ROOT = Path(
    os.getenv("HERMES_AGENT_REPO", Path(__file__).resolve().parent.parent.parent)
)
_HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_USER_SKILLS_DIR = _HERMES_HOME / "skills"
_REPO_SKILLS_DIR = _REPO_ROOT / "skills"

# These dirs hold the *optional* skills that ship with the repo but are not
# installed into ~/.hermes/ by default.
_OPTIONAL_SKILLS_DIR = _REPO_ROOT / "optional-skills"


def _find_skill_dir(skill_name: str) -> Optional[Path]:
    """Return the directory containing SKILL.md for *skill_name*.

    Search order:
      1. $HERMES_AGENT_REPO/skills/   (repo built-in skills, all categories)
      2. $HERMES_AGENT_REPO/optional-skills/
      3. ~/.hermes/skills/             (user-installed / hub skills)

    Skills may be nested inside category subdirectories
    (e.g., skills/github/github-code-review/).
    """
    search_dirs = [_REPO_SKILLS_DIR, _OPTIONAL_SKILLS_DIR, _USER_SKILLS_DIR]
    for base in search_dirs:
        if not base.exists():
            continue
        for skill_md in base.rglob("SKILL.md"):
            skill_dir = skill_md.parent
            # Match by directory name or by the 'name' frontmatter field.
            if skill_dir.name == skill_name:
                return skill_dir
            try:
                fm = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))[0]
                if fm.get("name") == skill_name:
                    return skill_dir
            except Exception:
                pass
    return None


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from SKILL.md content.

    Returns (frontmatter_dict, body_text).
    """
    frontmatter: Dict[str, Any] = {}
    body = content

    if content.startswith("---"):
        end_match = re.search(r"\n---\s*\n", content[3:])
        if end_match:
            yaml_content = content[3 : end_match.start() + 3]
            body = content[end_match.end() + 3 :]
            try:
                parsed = yaml.safe_load(yaml_content)
                if isinstance(parsed, dict):
                    frontmatter = parsed
            except yaml.YAMLError:
                for line in yaml_content.strip().split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        frontmatter[key.strip()] = val.strip()

    return frontmatter, body


def _load_skill(skill_dir: Path) -> Tuple[Dict[str, Any], str, str]:
    """Load a skill from *skill_dir*.

    Returns (frontmatter, body, full_content).
    """
    skill_md = skill_dir / "SKILL.md"
    full_content = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(full_content)
    return frontmatter, body, full_content


def _save_skill(skill_dir: Path, content: str) -> None:
    """Write *content* back to SKILL.md in *skill_dir*."""
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _build_openai_client():
    """Return an OpenAI-compatible client and the model name to use.

    Resolution order:
      1. OPENAI_API_KEY            → api.openai.com, model gpt-4o
      2. ANTHROPIC_API_KEY         → api.anthropic.com via openai compat
      3. OPENROUTER_API_KEY        → openrouter.ai, model openai/gpt-4o
      4. OPENAI_BASE_URL + OPENAI_API_KEY (custom provider)
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "openai package not found. Install it with: pip install openai"
        ) from exc

    model_override = os.getenv("EVOLUTION_MODEL")

    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        model = model_override or "gpt-4o"
        return client, model

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic as _ant
            # Wrap in a thin shim so the rest of the code uses the same interface.
            return _AnthropicShim(api_key=os.getenv("ANTHROPIC_API_KEY")), (
                model_override or "claude-opus-4-5-20251101"
            )
        except ImportError:
            pass

    if os.getenv("OPENROUTER_API_KEY"):
        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        model = model_override or "openai/gpt-4o"
        return client, model

    raise SystemExit(
        "No LLM API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "OPENROUTER_API_KEY"
    )


class _AnthropicShim:
    """Minimal shim so Anthropic calls use the same interface as OpenAI calls."""

    def __init__(self, api_key: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _chat(self, model: str, messages: List[Dict], max_tokens: int = 4096) -> str:
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append(m)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text


def _call_llm(
    client,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 4096,
) -> str:
    """Make a chat completion call and return the assistant's text."""
    if isinstance(client, _AnthropicShim):
        return client._chat(model=model, messages=messages, max_tokens=max_tokens)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Evaluation case generation
# ---------------------------------------------------------------------------

_SYNTHETIC_GENERATION_PROMPT = """\
You are an expert at creating evaluation test cases for AI agent skill documents.

Below is the SKILL.md content for the "{skill_name}" skill:

<skill>
{skill_content}
</skill>

Generate {n_cases} diverse, realistic test scenarios that a user might bring to an \
agent when invoking this skill. Each test case should:
1. Be a concrete, realistic user request related to the skill's topic
2. Cover different use cases, difficulty levels, and edge cases
3. Be specific enough that the skill's instructions can be meaningfully evaluated

Return ONLY a JSON array of objects with these fields:
- "id": unique identifier (e.g., "case_1")
- "user_request": the user's message/task description
- "expected_behaviors": list of 2-4 behaviors the agent should demonstrate
- "difficulty": "easy", "medium", or "hard"

Example format:
[
  {{
    "id": "case_1",
    "user_request": "Review the code changes I just made before I push them",
    "expected_behaviors": [
      "Uses git diff to inspect changes",
      "Checks for security issues",
      "Reports findings in structured format"
    ],
    "difficulty": "easy"
  }}
]
"""

_SESSIONDB_SEARCH_TERMS_PROMPT = """\
Given the skill "{skill_name}" with this description:

{skill_description}

Generate 5 search terms or short phrases that would appear in real user conversations \
where someone is asking an AI agent to perform this skill's tasks. These terms will be \
used to search a session database (FTS5 full-text search).

Return ONLY a JSON array of strings. Example:
["code review", "review my PR", "check my changes before pushing"]
"""


def _generate_synthetic_cases(
    client, model: str, skill_name: str, skill_content: str, n_cases: int = 8
) -> List[Dict[str, Any]]:
    """Generate synthetic evaluation cases from the skill content."""
    logger.info("Generating %d synthetic evaluation cases...", n_cases)

    prompt = _SYNTHETIC_GENERATION_PROMPT.format(
        skill_name=skill_name,
        skill_content=skill_content[:6000],  # cap to avoid context overflow
        n_cases=n_cases,
    )

    messages = [
        {"role": "system", "content": "You are an expert AI evaluation specialist."},
        {"role": "user", "content": prompt},
    ]

    raw = _call_llm(client, model, messages, max_tokens=2048)

    # Extract JSON from the response (may have markdown code fences)
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        logger.warning("LLM did not return valid JSON for eval cases; using fallback")
        return _fallback_synthetic_cases(skill_name)

    try:
        cases = json.loads(json_match.group(0))
        if not isinstance(cases, list) or not cases:
            raise ValueError("Empty case list")
        logger.info("Generated %d synthetic evaluation cases", len(cases))
        return cases
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse eval cases: %s; using fallback", exc)
        return _fallback_synthetic_cases(skill_name)


def _fallback_synthetic_cases(skill_name: str) -> List[Dict[str, Any]]:
    """Return minimal fallback cases when LLM generation fails."""
    return [
        {
            "id": "case_1",
            "user_request": f"Please help me use the {skill_name} skill",
            "expected_behaviors": ["Follows skill instructions", "Provides useful output"],
            "difficulty": "easy",
        },
        {
            "id": "case_2",
            "user_request": f"I need to perform a complex {skill_name} task",
            "expected_behaviors": [
                "Handles complexity",
                "Uses appropriate tools",
                "Reports results clearly",
            ],
            "difficulty": "hard",
        },
    ]


def _load_sessiondb_cases(
    client, model: str, skill_name: str, frontmatter: Dict[str, Any], limit: int = 20
) -> List[Dict[str, Any]]:
    """Load evaluation cases from the Hermes SessionDB.

    Searches the SessionDB for sessions related to the skill topic,
    extracts relevant user-assistant exchanges, and formats them as
    evaluation cases.
    """
    # Ensure repo root is on sys.path for hermes_state import
    repo_root = str(_REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from hermes_state import SessionDB, DEFAULT_DB_PATH
    except ImportError as exc:
        raise SystemExit(
            f"Could not import hermes_state from {_REPO_ROOT}. "
            "Ensure HERMES_AGENT_REPO points to the hermes-agent repo root."
        ) from exc

    db_path = _HERMES_HOME / "state.db"
    if not db_path.exists():
        logger.warning(
            "No SessionDB found at %s. Have you used hermes-agent CLI before?", db_path
        )
        return []

    db = SessionDB(db_path=db_path)
    try:
        return _extract_cases_from_db(client, model, db, skill_name, frontmatter, limit)
    finally:
        db.close()


def _extract_cases_from_db(
    client,
    model: str,
    db,
    skill_name: str,
    frontmatter: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    """Extract evaluation cases from a SessionDB instance."""
    description = frontmatter.get("description", skill_name)
    tags = []
    hermes_meta = frontmatter.get("metadata", {}).get("hermes", {})
    if isinstance(hermes_meta, dict):
        tags = hermes_meta.get("tags", [])

    # Build search terms from the skill topic
    search_terms = _get_search_terms(client, model, skill_name, description, tags)
    logger.info("Searching SessionDB with terms: %s", search_terms)

    seen_session_ids: set = set()
    raw_cases: List[Dict[str, Any]] = []

    for term in search_terms:
        if len(raw_cases) >= limit:
            break
        try:
            hits = db.search_messages(query=term, role_filter=["user"], limit=10)
        except Exception as exc:
            logger.debug("Search failed for term %r: %s", term, exc)
            continue

        for hit in hits:
            sid = hit.get("session_id")
            if sid in seen_session_ids:
                continue
            seen_session_ids.add(sid)

            # Load the full session and extract the first meaningful exchange
            messages = db.get_messages(sid)
            case = _messages_to_case(sid, messages, term)
            if case:
                raw_cases.append(case)
            if len(raw_cases) >= limit:
                break

    logger.info("Loaded %d cases from SessionDB", len(raw_cases))
    return raw_cases


def _get_search_terms(
    client, model: str, skill_name: str, description: str, tags: List[str]
) -> List[str]:
    """Ask the LLM for good FTS5 search terms for this skill."""
    prompt = _SESSIONDB_SEARCH_TERMS_PROMPT.format(
        skill_name=skill_name, skill_description=description
    )
    messages = [
        {"role": "system", "content": "You are a search query specialist."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = _call_llm(client, model, messages, max_tokens=512)
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if json_match:
            terms = json.loads(json_match.group(0))
            if isinstance(terms, list) and terms:
                return [str(t) for t in terms]
    except Exception as exc:
        logger.debug("Failed to get search terms from LLM: %s", exc)

    # Fallback: use skill name, tags, and first words of description
    fallback = [skill_name] + tags[:3]
    if description:
        fallback += description.split()[:3]
    return list(dict.fromkeys(fallback))  # deduplicate while preserving order


def _messages_to_case(
    session_id: str, messages: List[Dict[str, Any]], search_term: str
) -> Optional[Dict[str, Any]]:
    """Convert a session's messages into an evaluation case dict."""
    # Find the first user message with content
    user_msg = next(
        (m for m in messages if m.get("role") == "user" and m.get("content")),
        None,
    )
    if not user_msg:
        return None

    user_text = (user_msg.get("content") or "").strip()
    if not user_text or len(user_text) < 10:
        return None

    # Find subsequent assistant response
    assistant_msg = next(
        (
            m
            for m in messages
            if m.get("role") == "assistant" and m.get("content")
        ),
        None,
    )
    assistant_text = (assistant_msg.get("content") or "")[:500] if assistant_msg else ""

    return {
        "id": f"sessiondb_{session_id[:8]}",
        "user_request": user_text[:500],
        "assistant_response_preview": assistant_text,
        "search_term": search_term,
        "source": "sessiondb",
        "difficulty": "unknown",
    }


# ---------------------------------------------------------------------------
# Skill evaluation
# ---------------------------------------------------------------------------

_EVALUATION_PROMPT = """\
You are an expert AI skill evaluator. Evaluate how well the following skill \
document would guide an AI agent to handle a given user request.

<skill_name>{skill_name}</skill_name>

<skill_content>
{skill_content}
</skill_content>

<evaluation_cases>
{cases_json}
</evaluation_cases>

For each case, assess:
1. Does the skill provide clear enough instructions for this scenario?
2. Are there missing steps, commands, or context?
3. Are there ambiguities that could confuse an AI agent?
4. Does the skill cover edge cases like this?

Return a JSON object with this structure:
{{
  "overall_score": <float 0.0-1.0>,
  "case_scores": [
    {{
      "id": "<case_id>",
      "score": <float 0.0-1.0>,
      "strengths": ["<what the skill does well for this case>"],
      "weaknesses": ["<gaps or ambiguities for this case>"]
    }}
  ],
  "top_weaknesses": ["<most important gaps to fix, ordered by impact>"],
  "suggested_improvements": ["<concrete, actionable improvement suggestions>"]
}}
"""


def _evaluate_skill(
    client,
    model: str,
    skill_name: str,
    skill_content: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate the skill against the given cases using an LLM."""
    logger.info("Evaluating skill against %d cases...", len(cases))

    # Limit cases and content size to avoid context overflow
    cases_subset = cases[:10]
    cases_json = json.dumps(cases_subset, indent=2, ensure_ascii=False)

    prompt = _EVALUATION_PROMPT.format(
        skill_name=skill_name,
        skill_content=skill_content[:5000],
        cases_json=cases_json[:3000],
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert AI skill evaluator. Return valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]

    raw = _call_llm(client, model, messages, max_tokens=2048)

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        logger.warning("Evaluator returned no JSON; treating as zero score")
        return {"overall_score": 0.0, "top_weaknesses": [], "suggested_improvements": []}

    try:
        result = json.loads(json_match.group(0))
        score = float(result.get("overall_score", 0.0))
        logger.info("Skill evaluation score: %.3f", score)
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse evaluation result: %s", exc)
        return {"overall_score": 0.0, "top_weaknesses": [], "suggested_improvements": []}


# ---------------------------------------------------------------------------
# Skill improvement
# ---------------------------------------------------------------------------

_IMPROVEMENT_PROMPT = """\
You are an expert technical writer specializing in AI agent skill documentation.

Improve the following skill document based on the evaluation feedback provided.

<skill_name>{skill_name}</skill_name>

<current_skill_content>
{skill_content}
</current_skill_content>

<evaluation_feedback>
Overall score: {score:.2f}/1.0

Top weaknesses identified:
{weaknesses}

Suggested improvements:
{suggestions}
</evaluation_feedback>

Rewrite the COMPLETE SKILL.md file to address these issues. Requirements:
1. Keep the YAML frontmatter (---) at the top, preserving name/description/version/author
2. Bump the patch version (e.g., 1.0.0 → 1.0.1) to track this evolution
3. Address ALL identified weaknesses with concrete additions or clarifications
4. Maintain the existing structure and style of the skill document
5. Do NOT remove any existing correct instructions — only add and improve
6. Keep total length reasonable (prefer depth over breadth)

Return ONLY the complete SKILL.md content, starting with --- and ending with \
the last line of the document. No extra commentary.
"""


def _improve_skill(
    client,
    model: str,
    skill_name: str,
    skill_content: str,
    evaluation: Dict[str, Any],
    iteration: int,
) -> str:
    """Generate an improved version of the skill using LLM feedback."""
    logger.info("Generating improved skill (iteration %d)...", iteration)

    weaknesses = evaluation.get("top_weaknesses", [])
    suggestions = evaluation.get("suggested_improvements", [])
    score = float(evaluation.get("overall_score", 0.0))

    weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) or "(none identified)"
    suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "(none)"

    prompt = _IMPROVEMENT_PROMPT.format(
        skill_name=skill_name,
        skill_content=skill_content[:6000],
        score=score,
        weaknesses=weaknesses_text,
        suggestions=suggestions_text,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical writer for AI agent skills. "
                "Return only the complete SKILL.md content."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    improved = _call_llm(client, model, messages, max_tokens=4096)
    improved = improved.strip()

    # Sanity check: the result must start with --- (YAML frontmatter)
    if not improved.startswith("---"):
        logger.warning(
            "LLM returned skill without frontmatter; keeping original for this iteration"
        )
        return skill_content

    # Verify the frontmatter still contains the skill name
    fm, _ = _parse_frontmatter(improved)
    if fm.get("name") != skill_name and "name" in fm:
        logger.warning(
            "LLM changed skill name from %r to %r; rejecting this iteration",
            skill_name,
            fm.get("name"),
        )
        return skill_content

    return improved


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

def _bump_version(frontmatter: Dict[str, Any]) -> str:
    """Bump the patch segment of the skill version (e.g., 1.0.0 → 1.0.1)."""
    version = str(frontmatter.get("version", "1.0.0"))
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
    except (ValueError, IndexError):
        parts.append("1")
    return ".".join(parts)


def evolve_skill(
    skill: str,
    iterations: int = 5,
    eval_source: str = "synthetic",
    n_eval_cases: int = 8,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evolve a skill through iterative evaluation and improvement.

    Args:
        skill:        Name of the skill to evolve (matches the skill directory
                      name or the 'name' field in SKILL.md frontmatter).
        iterations:   Number of evaluation → improvement cycles to run.
        eval_source:  Where to get evaluation cases: "synthetic" (LLM-generated)
                      or "sessiondb" (real Hermes session history).
        n_eval_cases: Number of evaluation cases to generate per iteration
                      (only used for synthetic source).
        dry_run:      If True, run the full loop but do not write the improved
                      skill back to disk.
        verbose:      Enable DEBUG logging.

    Returns:
        A summary dict with keys: skill_name, iterations_run, scores,
        final_score, skill_dir.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --- Find skill ---
    skill_dir = _find_skill_dir(skill)
    if skill_dir is None:
        raise SystemExit(
            f"Skill {skill!r} not found. Searched in:\n"
            f"  {_REPO_SKILLS_DIR}\n"
            f"  {_OPTIONAL_SKILLS_DIR}\n"
            f"  {_USER_SKILLS_DIR}\n"
            "Set HERMES_AGENT_REPO to point to your hermes-agent repo."
        )

    logger.info("Found skill at: %s", skill_dir)

    # --- Load skill ---
    frontmatter, body, skill_content = _load_skill(skill_dir)
    skill_name = frontmatter.get("name", skill_dir.name)
    logger.info("Skill: %s (version %s)", skill_name, frontmatter.get("version", "?"))

    # --- Build LLM client ---
    client, model = _build_openai_client()
    logger.info("Using model: %s", model)

    # --- Load evaluation cases ---
    if eval_source == "sessiondb":
        eval_cases = _load_sessiondb_cases(client, model, skill_name, frontmatter)
        if not eval_cases:
            logger.warning(
                "No SessionDB cases found for %r; falling back to synthetic", skill_name
            )
            eval_cases = _generate_synthetic_cases(
                client, model, skill_name, skill_content, n_eval_cases
            )
    elif eval_source == "synthetic":
        eval_cases = _generate_synthetic_cases(
            client, model, skill_name, skill_content, n_eval_cases
        )
    else:
        raise SystemExit(
            f"Unknown eval-source: {eval_source!r}. Use 'synthetic' or 'sessiondb'."
        )

    if not eval_cases:
        raise SystemExit("No evaluation cases available. Cannot evolve skill.")

    # --- Evolution loop ---
    scores: List[float] = []
    current_content = skill_content

    for i in range(1, iterations + 1):
        logger.info("=" * 60)
        logger.info("Iteration %d / %d", i, iterations)
        logger.info("=" * 60)

        # Evaluate current skill
        evaluation = _evaluate_skill(
            client, model, skill_name, current_content, eval_cases
        )
        score = float(evaluation.get("overall_score", 0.0))
        scores.append(score)
        logger.info("Score: %.3f", score)

        if score >= 0.95:
            logger.info("Skill reached near-perfect score (%.3f); stopping early.", score)
            break

        # Improve the skill
        improved_content = _improve_skill(
            client, model, skill_name, current_content, evaluation, i
        )

        if improved_content == current_content:
            logger.info("No improvement made in iteration %d; stopping early.", i)
            break

        current_content = improved_content

        # Save after each iteration (unless dry-run)
        if not dry_run:
            _save_skill(skill_dir, current_content)
            logger.info("Saved improved skill to %s", skill_dir / "SKILL.md")
        else:
            logger.info("[dry-run] Would have saved improved skill to %s", skill_dir / "SKILL.md")

        # Short pause to avoid rate limiting
        if i < iterations:
            time.sleep(1)

    final_score = scores[-1] if scores else 0.0
    summary = {
        "skill_name": skill_name,
        "skill_dir": str(skill_dir),
        "eval_source": eval_source,
        "iterations_run": len(scores),
        "scores": scores,
        "final_score": final_score,
        "improved": current_content != skill_content,
        "dry_run": dry_run,
    }

    logger.info("=" * 60)
    logger.info("Evolution complete!")
    logger.info("  Skill:         %s", skill_name)
    logger.info("  Iterations:    %d", len(scores))
    logger.info("  Scores:        %s", " → ".join(f"{s:.3f}" for s in scores))
    logger.info("  Final score:   %.3f", final_score)
    if not dry_run and summary["improved"]:
        logger.info("  Saved to:      %s", skill_dir / "SKILL.md")
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evolution.skills.evolve_skill",
        description="Evolve a hermes-agent skill through iterative LLM evaluation and improvement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skill",
        required=True,
        help="Skill name to evolve (e.g., github-code-review)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        metavar="N",
        help="Number of evaluate→improve cycles (default: 5)",
    )
    parser.add_argument(
        "--eval-source",
        choices=["synthetic", "sessiondb"],
        default="synthetic",
        help="Source of evaluation cases: 'synthetic' (LLM-generated) or "
             "'sessiondb' (real Hermes session history). Default: synthetic",
    )
    parser.add_argument(
        "--n-eval-cases",
        type=int,
        default=8,
        metavar="N",
        help="Number of synthetic eval cases to generate per iteration (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the evolution loop without writing changes to disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load .env files so API keys are available
    try:
        from dotenv import load_dotenv
        hermes_env = _HERMES_HOME / ".env"
        if hermes_env.exists():
            load_dotenv(hermes_env)
        repo_env = _REPO_ROOT / ".env"
        if repo_env.exists():
            load_dotenv(repo_env)
    except ImportError:
        pass

    summary = evolve_skill(
        skill=args.skill,
        iterations=args.iterations,
        eval_source=args.eval_source,
        n_eval_cases=args.n_eval_cases,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
