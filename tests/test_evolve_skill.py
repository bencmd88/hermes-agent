"""Tests for evolution/skills/evolve_skill.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evolution.skills.evolve_skill import (
    _AnthropicShim,
    _build_openai_client,
    _bump_version,
    _evaluate_skill,
    _extract_cases_from_db,
    _fallback_synthetic_cases,
    _find_skill_dir,
    _generate_synthetic_cases,
    _get_search_terms,
    _improve_skill,
    _load_skill,
    _messages_to_case,
    _parse_frontmatter,
    _REPO_SKILLS_DIR,
    _save_skill,
    evolve_skill,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_SKILL_MD = """\
---
name: test-skill
description: A test skill for unit tests.
version: 1.0.0
author: Tests
---

# Test Skill

## Overview

This skill does things.

## Usage

Run this command:

```bash
echo "hello"
```
"""

SAMPLE_FRONTMATTER = {
    "name": "test-skill",
    "description": "A test skill for unit tests.",
    "version": "1.0.0",
    "author": "Tests",
}


def _fake_client(response_text: str = "") -> MagicMock:
    """Return a mock OpenAI-compatible client that always returns *response_text*."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ---------------------------------------------------------------------------
# _parse_frontmatter
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_parses_valid_yaml(self):
        fm, body = _parse_frontmatter(SAMPLE_SKILL_MD)
        assert fm["name"] == "test-skill"
        assert fm["version"] == "1.0.0"
        assert "Test Skill" in body

    def test_empty_content(self):
        fm, body = _parse_frontmatter("")
        assert fm == {}
        assert body == ""

    def test_no_frontmatter(self):
        content = "# Just a doc\nNo frontmatter here."
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert "Just a doc" in body

    def test_only_frontmatter(self):
        content = "---\nname: solo\n---\n"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "solo"


# ---------------------------------------------------------------------------
# _find_skill_dir
# ---------------------------------------------------------------------------

class TestFindSkillDir:
    def test_finds_builtin_skill(self):
        """github-code-review ships in the repo's skills/github/ directory."""
        skill_dir = _find_skill_dir("github-code-review")
        assert skill_dir is not None
        assert (skill_dir / "SKILL.md").exists()

    def test_returns_none_for_unknown_skill(self):
        assert _find_skill_dir("this-skill-does-not-exist-xyz") is None

    def test_finds_skill_in_user_dir(self, tmp_path, monkeypatch):
        user_skills = tmp_path / "skills"
        skill_dir = user_skills / "my-custom-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-custom-skill\ndescription: x\n---\n# Body\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "evolution.skills.evolve_skill._USER_SKILLS_DIR", user_skills
        )
        found = _find_skill_dir("my-custom-skill")
        assert found == skill_dir


# ---------------------------------------------------------------------------
# _load_skill and _save_skill
# ---------------------------------------------------------------------------

class TestLoadSaveSkill:
    def test_load_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        fm, body, full = _load_skill(skill_dir)
        assert fm["name"] == "test-skill"
        assert "Test Skill" in body
        assert full == SAMPLE_SKILL_MD

    def test_save_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("original", encoding="utf-8")
        _save_skill(skill_dir, "new content")
        assert (skill_dir / "SKILL.md").read_text() == "new content"


# ---------------------------------------------------------------------------
# _bump_version
# ---------------------------------------------------------------------------

class TestBumpVersion:
    def test_bumps_patch(self):
        assert _bump_version({"version": "1.0.0"}) == "1.0.1"
        assert _bump_version({"version": "2.3.7"}) == "2.3.8"

    def test_missing_version(self):
        result = _bump_version({})
        # Should default to 1.0.0 → 1.0.1
        assert result == "1.0.1"

    def test_single_segment_version(self):
        result = _bump_version({"version": "5"})
        assert result == "6"


# ---------------------------------------------------------------------------
# _fallback_synthetic_cases
# ---------------------------------------------------------------------------

class TestFallbackSyntheticCases:
    def test_returns_two_cases(self):
        cases = _fallback_synthetic_cases("my-skill")
        assert len(cases) == 2
        assert all("id" in c for c in cases)
        assert all("user_request" in c for c in cases)


# ---------------------------------------------------------------------------
# _generate_synthetic_cases
# ---------------------------------------------------------------------------

class TestGenerateSyntheticCases:
    def test_parses_valid_json_response(self):
        cases_json = json.dumps([
            {
                "id": "case_1",
                "user_request": "Do a code review",
                "expected_behaviors": ["reads diff", "reports issues"],
                "difficulty": "easy",
            }
        ])
        client = _fake_client(cases_json)
        cases = _generate_synthetic_cases(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, 1)
        assert len(cases) == 1
        assert cases[0]["id"] == "case_1"

    def test_falls_back_on_invalid_json(self):
        client = _fake_client("Not valid JSON at all")
        cases = _generate_synthetic_cases(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, 2)
        assert len(cases) == 2  # fallback returns 2

    def test_falls_back_on_empty_list(self):
        client = _fake_client("[]")
        cases = _generate_synthetic_cases(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, 2)
        assert len(cases) == 2  # fallback


# ---------------------------------------------------------------------------
# _messages_to_case
# ---------------------------------------------------------------------------

class TestMessagesToCase:
    def test_extracts_user_request(self):
        msgs = [
            {"role": "user", "content": "Please review my code"},
            {"role": "assistant", "content": "Sure, let me check"},
        ]
        case = _messages_to_case("abc123", msgs, "code review")
        assert case is not None
        assert "review my code" in case["user_request"]
        assert case["id"].startswith("sessiondb_")

    def test_returns_none_for_no_user_message(self):
        msgs = [{"role": "assistant", "content": "Hi!"}]
        assert _messages_to_case("abc", msgs, "term") is None

    def test_returns_none_for_empty_user_content(self):
        msgs = [{"role": "user", "content": "   "}]
        assert _messages_to_case("abc", msgs, "term") is None

    def test_returns_none_for_too_short_content(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _messages_to_case("abc", msgs, "term") is None


# ---------------------------------------------------------------------------
# _get_search_terms
# ---------------------------------------------------------------------------

class TestGetSearchTerms:
    def test_parses_valid_json_terms(self):
        client = _fake_client('["code review", "review PR", "check changes"]')
        terms = _get_search_terms(client, "gpt-4o", "test-skill", "A test skill", [])
        assert "code review" in terms

    def test_fallback_on_bad_response(self):
        client = _fake_client("not json")
        terms = _get_search_terms(
            client, "gpt-4o", "code-review", "Review code changes", ["GitHub", "PR"]
        )
        assert "code-review" in terms or "GitHub" in terms


# ---------------------------------------------------------------------------
# _evaluate_skill
# ---------------------------------------------------------------------------

class TestEvaluateSkill:
    def test_parses_evaluation_result(self):
        eval_json = json.dumps({
            "overall_score": 0.75,
            "case_scores": [],
            "top_weaknesses": ["Missing edge case X"],
            "suggested_improvements": ["Add section Y"],
        })
        client = _fake_client(eval_json)
        result = _evaluate_skill(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, [])
        assert result["overall_score"] == pytest.approx(0.75)
        assert "Missing edge case X" in result["top_weaknesses"]

    def test_falls_back_on_no_json(self):
        client = _fake_client("no json here")
        result = _evaluate_skill(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, [])
        assert result["overall_score"] == 0.0


# ---------------------------------------------------------------------------
# _improve_skill
# ---------------------------------------------------------------------------

class TestImproveSkill:
    def test_returns_improved_content(self):
        improved_md = "---\nname: test-skill\ndescription: Better.\nversion: 1.0.1\n---\n# Improved"
        client = _fake_client(improved_md)
        evaluation = {
            "overall_score": 0.5,
            "top_weaknesses": ["Missing edge case"],
            "suggested_improvements": ["Add examples"],
        }
        result = _improve_skill(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, evaluation, 1)
        assert result == improved_md

    def test_rejects_content_without_frontmatter(self):
        client = _fake_client("No frontmatter here at all")
        evaluation = {"overall_score": 0.5, "top_weaknesses": [], "suggested_improvements": []}
        result = _improve_skill(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, evaluation, 1)
        # Should fall back to original
        assert result == SAMPLE_SKILL_MD

    def test_rejects_content_with_changed_skill_name(self):
        bad_md = "---\nname: different-skill\ndescription: x\n---\n# Bad\n"
        client = _fake_client(bad_md)
        evaluation = {"overall_score": 0.5, "top_weaknesses": [], "suggested_improvements": []}
        result = _improve_skill(client, "gpt-4o", "test-skill", SAMPLE_SKILL_MD, evaluation, 1)
        assert result == SAMPLE_SKILL_MD


# ---------------------------------------------------------------------------
# _extract_cases_from_db
# ---------------------------------------------------------------------------

class TestExtractCasesFromDb:
    def test_extracts_cases_from_sessions(self, tmp_path):
        """Integration test against a real in-memory SessionDB."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "test.db")
        db.create_session("s1", source="cli", model="test")
        db.append_message("s1", "user", content="Please review my pull request")
        db.append_message("s1", "assistant", content="I'll review the PR now.")
        db.end_session("s1", "user_exit")

        # Mock search_terms to return the term that matches session content
        terms_json = '["pull request", "review"]'
        client = _fake_client(terms_json)

        cases = _extract_cases_from_db(
            client, "gpt-4o", db, "github-code-review",
            {"description": "Code review skill"}, limit=5
        )
        db.close()
        # Should find at least one case (the session we inserted)
        assert len(cases) >= 1
        assert any("pull request" in c["user_request"].lower() for c in cases)


# ---------------------------------------------------------------------------
# evolve_skill (end-to-end with mocks)
# ---------------------------------------------------------------------------

class TestEvolveSkillEndToEnd:
    def test_dry_run_does_not_modify_skill(self, tmp_path):
        """evolve_skill with dry_run=True should not write to disk."""
        # Set up a fake skill
        skill_dir = tmp_path / "skills" / "my-test-skill"
        skill_dir.mkdir(parents=True)
        original = SAMPLE_SKILL_MD.replace("test-skill", "my-test-skill")
        (skill_dir / "SKILL.md").write_text(original, encoding="utf-8")

        eval_json = json.dumps({
            "overall_score": 0.6,
            "case_scores": [],
            "top_weaknesses": ["Too brief"],
            "suggested_improvements": ["Add more examples"],
        })
        cases_json = json.dumps([
            {
                "id": "c1",
                "user_request": "Do the thing",
                "expected_behaviors": ["does thing"],
                "difficulty": "easy",
            }
        ])
        improved_md = original.replace("1.0.0", "1.0.1").replace(
            "A test skill for unit tests.", "An improved test skill."
        )

        with (
            patch("evolution.skills.evolve_skill._find_skill_dir", return_value=skill_dir),
            patch("evolution.skills.evolve_skill._build_openai_client") as mock_client,
        ):
            client = MagicMock()
            responses = iter([cases_json, eval_json, improved_md])

            def fake_create(*a, **kw):
                resp = next(responses)
                choice = MagicMock()
                choice.message.content = resp
                return MagicMock(choices=[choice])

            client.chat.completions.create.side_effect = fake_create
            mock_client.return_value = (client, "gpt-4o")

            summary = evolve_skill(
                skill="my-test-skill",
                iterations=1,
                eval_source="synthetic",
                n_eval_cases=1,
                dry_run=True,
            )

        # Dry-run: file must remain unchanged
        assert (skill_dir / "SKILL.md").read_text() == original
        assert summary["dry_run"] is True
        assert summary["iterations_run"] == 1

    def test_early_stop_on_high_score(self, tmp_path):
        """evolve_skill should stop early when score >= 0.95."""
        skill_dir = tmp_path / "skills" / "high-score-skill"
        skill_dir.mkdir(parents=True)
        content = SAMPLE_SKILL_MD.replace("test-skill", "high-score-skill")
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        eval_json = json.dumps({"overall_score": 0.97, "case_scores": [], "top_weaknesses": [], "suggested_improvements": []})
        cases_json = json.dumps([{"id": "c1", "user_request": "Do a thing", "expected_behaviors": ["ok"], "difficulty": "easy"}])

        with (
            patch("evolution.skills.evolve_skill._find_skill_dir", return_value=skill_dir),
            patch("evolution.skills.evolve_skill._build_openai_client") as mock_client,
        ):
            client = MagicMock()
            responses = iter([cases_json, eval_json])

            def fake_create(*a, **kw):
                resp = next(responses)
                choice = MagicMock()
                choice.message.content = resp
                return MagicMock(choices=[choice])

            client.chat.completions.create.side_effect = fake_create
            mock_client.return_value = (client, "gpt-4o")

            summary = evolve_skill(
                skill="high-score-skill",
                iterations=10,
                eval_source="synthetic",
                dry_run=True,
            )

        # Should have stopped after 1 iteration (score >= 0.95)
        assert summary["iterations_run"] == 1
        assert summary["final_score"] >= 0.95

    def test_raises_on_unknown_skill(self):
        with patch("evolution.skills.evolve_skill._find_skill_dir", return_value=None):
            with pytest.raises(SystemExit):
                evolve_skill(skill="no-such-skill", iterations=1, dry_run=True)

    def test_raises_on_unknown_eval_source(self, tmp_path):
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
        with patch("evolution.skills.evolve_skill._find_skill_dir", return_value=skill_dir):
            with patch("evolution.skills.evolve_skill._build_openai_client") as mc:
                mc.return_value = (_fake_client(), "gpt-4o")
                with pytest.raises(SystemExit):
                    evolve_skill(skill="test-skill", iterations=1, eval_source="invalid")

    def test_sessiondb_source_falls_back_to_synthetic_when_empty(self, tmp_path):
        """When sessiondb returns no cases, should fall back to synthetic."""
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        content = SAMPLE_SKILL_MD.replace("test-skill", "my-skill")
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        eval_json = json.dumps({"overall_score": 0.99, "case_scores": [], "top_weaknesses": [], "suggested_improvements": []})
        cases_json = json.dumps([{"id": "c1", "user_request": "Do something", "expected_behaviors": [], "difficulty": "easy"}])

        with (
            patch("evolution.skills.evolve_skill._find_skill_dir", return_value=skill_dir),
            patch("evolution.skills.evolve_skill._build_openai_client") as mc,
            patch("evolution.skills.evolve_skill._load_sessiondb_cases", return_value=[]),
            patch("evolution.skills.evolve_skill._generate_synthetic_cases") as mock_synth,
        ):
            mock_synth.return_value = [{"id": "c1", "user_request": "Do something", "expected_behaviors": [], "difficulty": "easy"}]
            client = MagicMock()
            responses = iter([eval_json])

            def fake_create(*a, **kw):
                resp = next(responses)
                choice = MagicMock()
                choice.message.content = resp
                return MagicMock(choices=[choice])

            client.chat.completions.create.side_effect = fake_create
            mc.return_value = (client, "gpt-4o")

            summary = evolve_skill(
                skill="my-skill",
                iterations=1,
                eval_source="sessiondb",
                dry_run=True,
            )

        assert mock_synth.called
        assert summary["eval_source"] == "sessiondb"


# ---------------------------------------------------------------------------
# _build_openai_client
# ---------------------------------------------------------------------------

class TestBuildOpenAIClient:
    def test_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            # reload to pick up the mock
            import importlib
            import evolution.skills.evolve_skill as m
            importlib.reload(m)
            client, model = m._build_openai_client()
            assert "gpt" in model

    def test_openrouter_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            import importlib
            import evolution.skills.evolve_skill as m
            importlib.reload(m)
            client, model = m._build_openai_client()
            assert "gpt" in model or "/" in model

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            import importlib
            import evolution.skills.evolve_skill as m
            importlib.reload(m)
            with pytest.raises(SystemExit):
                m._build_openai_client()

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("EVOLUTION_MODEL", "gpt-custom")
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            import importlib
            import evolution.skills.evolve_skill as m
            importlib.reload(m)
            _, model = m._build_openai_client()
            assert model == "gpt-custom"
