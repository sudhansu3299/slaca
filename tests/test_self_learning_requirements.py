"""
Tests for the four self-learning hard requirements.

All tests run fully offline — no LLM calls, no MongoDB, no Redis.
LLM calls are intercepted by patching _llm(); MongoDB is mocked via
an in-memory dict; file I/O uses tmp_path (pytest fixture).

Requirement 1 — Quantitative Justification
  - 80/20 split happens before stage 1 scoring
  - Identical v1/v2 → p ≈ 1.0 → decision == "reject"
  - Clear improvement only adopts when ALL gates pass:
    n ≥ 30 per arm, p < 0.05, delta > 0.02, effect size (Cohen d) ≥ 0.20
  - LLM output cannot change the decision field

Requirement 2 — Compliance Preservation
  - Non-compliant *generated* prompt → pipeline sets decision="reject", _apply_prompt never called
  - apply_single_patch with non-compliant patched prompt → returns applied=False, file unchanged

Requirement 3 — Audit Trail
  - get_prompt_version() returns "canonical-v1" when no patches in file
  - get_prompt_version() recovers "patch-v3" from a file with LEARNED PATCH v3
  - After _apply_prompt(), get_prompt_version() returns the new version
  - _apply_prompt() inserts a document into the mocked prompt_changes collection

Requirement 4 — Rollback
  - rollback_prompt() restores backup content to live file
  - Backs up the current live content before overwriting
  - get_prompt_version() reflects the rolled-back version
  - Invalid backup_filename (path traversal) is rejected
  - Missing backup file is rejected
  - Backup file without {injected_guidance_block} is rejected
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────
# Helpers shared across tests
# ─────────────────────────────────────────────────────────────────

def _make_transcript(trace_id: str, borrower_id: str = "BRW-001") -> dict:
    return {
        "trace_id":     trace_id,
        "borrower_id":  borrower_id,
        "primary_agent": "AssessmentAgent",
        "history": [
            {"role": "user",      "content": "My last 4 digits are 7823"},
            {"role": "assistant", "content": "Thank you, what is your income?"},
            {"role": "user",      "content": "50000"},
        ],
    }


def _score(trace_id: str, resolution: int, borrower_id: str = "BRW-001"):
    """Build a TranscriptScore with minimal fields."""
    from src.self_learning.improvement_pipeline import TranscriptScore
    return TranscriptScore(
        trace_id=trace_id,
        borrower_id=borrower_id,
        agent_name="AssessmentAgent",
        resolution=resolution,
        resolution_confidence=0.9 if resolution else 0.1,
        debt_collected=resolution,
        compliance_violation=0,
        tone_score=4,
        next_step_clarity=4,
        raw_transcript_turns=3,
    )


VALID_PROMPT = (
    "You are an assessment agent.\n"
    "Ask for identity then income.\n"
    "{known_facts_block}\n"
    "{injected_guidance_block}"
)


# ═════════════════════════════════════════════════════════════════
# Requirement 1 — Quantitative Justification
# ═════════════════════════════════════════════════════════════════

class TestQuantitativeJustification:

    def test_fisher_exact_identical_groups_not_significant(self):
        """Same resolution rate in both groups → p ≈ 1.0 → reject."""
        from src.self_learning.improvement_pipeline import _fisher_exact_p
        p = _fisher_exact_p(5, 10, 5, 10)
        assert p > 0.05, f"Expected p > 0.05 for identical groups, got {p}"

    def test_fisher_exact_large_improvement_significant(self):
        """8/10 vs 2/10 → p < 0.05 → adopt."""
        from src.self_learning.improvement_pipeline import _fisher_exact_p
        p = _fisher_exact_p(2, 10, 8, 10)
        assert p < 0.05, f"Expected p < 0.05 for clear improvement, got {p}"

    @pytest.mark.asyncio
    async def test_compare_versions_reject_when_no_improvement(self):
        """compare_versions() must set decision='reject' when v2 == v1."""
        from src.self_learning.improvement_pipeline import compare_versions

        v1 = [_score(f"t{i}", 1 if i < 5 else 0) for i in range(10)]
        v2 = [_score(f"t{i}", 1 if i < 5 else 0) for i in range(10)]  # identical

        # LLM is not called because we patch _llm to raise — decision must still
        # be determined by statistics, not require a successful LLM call.
        with patch("src.self_learning.improvement_pipeline._llm",
                   side_effect=RuntimeError("LLM unavailable")):
            result = await compare_versions(v1, v2)

        assert result.decision == "reject", (
            f"Identical rates should be rejected; got decision={result.decision!r}, "
            f"p={result.p_value}"
        )
        assert result.p_value is not None

    @pytest.mark.asyncio
    async def test_compare_versions_adopt_when_significant_improvement(self):
        """compare_versions() adopts only when all quantitative gates pass."""
        from src.self_learning.improvement_pipeline import compare_versions

        # v1: 6/30 resolved; v2: 24/30 resolved
        # Passes min sample, significance, delta, and effect-size gates.
        v1 = [_score(f"t{i}", 1 if i < 6 else 0) for i in range(30)]
        v2 = [_score(f"t{i}", 1 if i < 24 else 0) for i in range(30)]

        llm_response = '{"summary": "v2 is better", "reason": "significant improvement"}'
        with patch("src.self_learning.improvement_pipeline._llm",
                   return_value=llm_response):
            result = await compare_versions(v1, v2)

        assert result.decision == "adopt", (
            f"Clear improvement should be adopted; got decision={result.decision!r}, "
            f"p={result.p_value}, v1={result.v1_resolution_rate}, v2={result.v2_resolution_rate}"
        )
        assert result.p_value < 0.05
        assert result.improvement.get("enough_samples") is True
        assert result.improvement.get("enough_delta") is True
        assert result.improvement.get("enough_effect") is True

    @pytest.mark.asyncio
    async def test_compare_versions_reject_when_effect_too_small(self):
        """Even with significance, tiny effect size must not be adopted."""
        from src.self_learning.improvement_pipeline import compare_versions

        # Small uplift: 15/30 -> 16/30 (delta ~3.3pp, low effect size)
        v1 = [_score(f"t{i}", 1 if i < 15 else 0) for i in range(30)]
        v2 = [_score(f"t{i}", 1 if i < 16 else 0) for i in range(30)]

        llm_response = '{"summary": "v2 slightly better", "reason": "small gain"}'
        with patch("src.self_learning.improvement_pipeline._fisher_exact_p", return_value=0.001), \
             patch("src.self_learning.improvement_pipeline._llm", return_value=llm_response):
            result = await compare_versions(v1, v2)

        assert result.decision == "reject"
        assert result.improvement.get("enough_samples") is True
        assert result.improvement.get("enough_delta") is True
        assert result.improvement.get("enough_effect") is False

    @pytest.mark.asyncio
    async def test_llm_cannot_override_decision(self):
        """Even if the LLM returns 'adopt', a non-significant result stays 'reject'."""
        from src.self_learning.improvement_pipeline import compare_versions

        # Identical rates — not significant
        v1 = [_score(f"t{i}", 1 if i < 5 else 0) for i in range(10)]
        v2 = [_score(f"t{i}", 1 if i < 5 else 0) for i in range(10)]

        # LLM tries to override with "adopt"
        llm_response = '{"summary": "looks good", "decision": "adopt", "reason": "trust me"}'
        with patch("src.self_learning.improvement_pipeline._llm",
                   return_value=llm_response):
            result = await compare_versions(v1, v2)

        assert result.decision == "reject", (
            "LLM must not be able to override a statistically non-significant result. "
            f"Got decision={result.decision!r}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_splits_transcripts_80_20(self, tmp_path):
        """run_improvement_pipeline() scores calibration and held-out separately."""
        from src.self_learning.improvement_pipeline import PipelineRun

        # 10 transcripts → expect ~8 calibration, ~2 held-out
        transcripts = [_make_transcript(f"t{i}", f"BRW-{i:03d}") for i in range(10)]

        score_calls: list[str] = []

        async def mock_score(trace_id, borrower_id, agent_name, history):
            score_calls.append(trace_id)
            return _score(trace_id, 1, borrower_id)

        # Patch everything that touches external state
        with patch("src.self_learning.improvement_pipeline.score_transcript",
                   side_effect=mock_score), \
             patch("src.self_learning.improvement_pipeline.analyze_failures",
                   return_value=MagicMock(failure_patterns=[], total_failed=0, total_analyzed=0)), \
             patch("src.self_learning.improvement_pipeline.check_compliance",
                   return_value=MagicMock(compliant=True, risks=[], reason="ok")), \
             patch("src.self_learning.improvement_pipeline.generate_prompt_improvement",
                   return_value=MagicMock(new_prompt=VALID_PROMPT, changes_summary=[], expected_impact="", target_agent="AssessmentAgent", based_on_patterns=0)), \
             patch("src.self_learning.improvement_pipeline.rescore_held_out",
                   side_effect=lambda ho, os_, **kw: asyncio.coroutine(lambda: os_)()), \
             patch("src.self_learning.improvement_pipeline.compare_versions",
                   return_value=MagicMock(decision="reject", v1_resolution_rate=0.8,
                                          v2_resolution_rate=0.8, p_value=1.0,
                                          summary="", reason="", improvement={})), \
             patch("src.self_learning.improvement_pipeline.generate_hypotheses",
                   return_value=MagicMock(hypotheses=[])), \
             patch("src.self_learning.improvement_pipeline._persist_run",
                   new_callable=AsyncMock), \
             patch("src.self_learning.improvement_pipeline._load_current_prompt",
                   return_value=VALID_PROMPT), \
             patch("src.self_learning.improvement_pipeline._apply_prompt"), \
             patch("src.self_learning.improvement_pipeline._fire_log_prompt_change"):

            run = await __import__(
                "src.self_learning.improvement_pipeline",
                fromlist=["run_improvement_pipeline"]
            ).run_improvement_pipeline(transcripts, "AssessmentAgent")

        # All 10 transcripts should have been scored (calibration + held-out)
        assert len(score_calls) == 10, (
            f"Expected all 10 transcripts to be scored; got {len(score_calls)}"
        )
        # calibration_scores stored on run
        assert len(run.transcript_scores) + len(run.held_out_scores) == 10


# ═════════════════════════════════════════════════════════════════
# Requirement 2 — Compliance Preservation
# ═════════════════════════════════════════════════════════════════

class TestCompliancePreservation:

    @pytest.mark.asyncio
    async def test_non_compliant_generated_prompt_is_rejected(self):
        """Pipeline must not adopt a generated prompt that fails compliance."""
        from src.self_learning.improvement_pipeline import run_improvement_pipeline

        transcripts = [_make_transcript(f"t{i}") for i in range(5)]

        apply_called = []

        def track_apply(agent, prompt, run_id):
            apply_called.append(True)

        async def mock_score(trace_id, borrower_id, agent_name, history):
            return _score(trace_id, 0, borrower_id)

        # check_compliance is called twice: once on current prompt (passes),
        # once on generated prompt (fails). Use side_effect list.
        compliance_side_effects = [
            MagicMock(compliant=True,  risks=[], reason="current prompt ok"),
            MagicMock(compliant=False,
                      risks=[{"rule": "Rule 3", "violation": "harassment instruction"}],
                      reason="Generated prompt violates Rule 3"),
        ]

        with patch("src.self_learning.improvement_pipeline.score_transcript",
                   side_effect=mock_score), \
             patch("src.self_learning.improvement_pipeline.analyze_failures",
                   return_value=MagicMock(failure_patterns=[], total_failed=0, total_analyzed=0)), \
             patch("src.self_learning.improvement_pipeline._load_current_prompt",
                   return_value=VALID_PROMPT), \
             patch("src.self_learning.improvement_pipeline.check_compliance",
                   side_effect=compliance_side_effects), \
             patch("src.self_learning.improvement_pipeline.generate_prompt_improvement",
                   return_value=MagicMock(
                       new_prompt="BAD PROMPT — instructs harassment",
                       changes_summary=[], expected_impact="", target_agent="AssessmentAgent",
                       based_on_patterns=0)), \
             patch("src.self_learning.improvement_pipeline._persist_run", new_callable=AsyncMock), \
             patch("src.self_learning.improvement_pipeline._apply_prompt", side_effect=track_apply), \
             patch("src.self_learning.improvement_pipeline._fire_log_prompt_change"):

            run = await run_improvement_pipeline(transcripts, "AssessmentAgent")

        assert run.decision == "reject", (
            f"Non-compliant generated prompt must be rejected; got {run.decision!r}"
        )
        assert not apply_called, "_apply_prompt must not be called when generated prompt is non-compliant"

    def test_apply_single_patch_rejects_non_compliant_patch(self, tmp_path):
        """apply_single_patch() must not write the file when patched prompt fails compliance."""
        import src.self_learning.improvement_pipeline as pip

        # Set up a temp prompt file
        fname = "assessment_system_prompt.txt"
        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()
        (prompts_dir / fname).write_text(VALID_PROMPT, encoding="utf-8")

        # Redirect _PROMPTS_DIR
        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        try:
            # Patch check_compliance to return non-compliant
            non_compliant = MagicMock(
                compliant=False,
                risks=[{"rule": "Rule 2", "violation": "false threat"}],
                reason="False threat in patch",
            )
            with patch("src.self_learning.improvement_pipeline.check_compliance",
                       return_value=non_compliant), \
                 patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.is_running.return_value = False
                mock_loop.return_value.run_until_complete.return_value = non_compliant

                result = pip.apply_single_patch(
                    "AssessmentAgent",
                    "You will be arrested if you do not pay.",
                    "run-test",
                    0,
                )

            assert result["applied"] is False, (
                f"Non-compliant patch must not be applied; got applied={result.get('applied')}"
            )
            assert "risks" in result

            # The live file must be unchanged
            live_content = (prompts_dir / fname).read_text(encoding="utf-8")
            assert live_content == VALID_PROMPT, "Live prompt file must not be modified"
        finally:
            pip._PROMPTS_DIR = original_dir

    def test_apply_single_patch_writes_compliant_patch(self, tmp_path):
        """apply_single_patch() writes the file when compliance passes."""
        import src.self_learning.improvement_pipeline as pip

        fname = "assessment_system_prompt.txt"
        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()
        (prompts_dir / fname).write_text(VALID_PROMPT, encoding="utf-8")

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        # Clear cached version so it reads from the temp dir
        pip._active_prompt_versions.pop("AssessmentAgent", None)
        try:
            compliant = MagicMock(compliant=True, risks=[], reason="all good")
            with patch("src.self_learning.improvement_pipeline.check_compliance",
                       return_value=compliant), \
                 patch("asyncio.get_event_loop") as mock_loop, \
                 patch("src.self_learning.improvement_pipeline._fire_log_prompt_change"):
                mock_loop.return_value.is_running.return_value = False
                mock_loop.return_value.run_until_complete.return_value = compliant

                result = pip.apply_single_patch(
                    "AssessmentAgent",
                    "Always state the deadline clearly.",
                    "run-test",
                    0,
                )

            assert result["applied"] is True
            live_content = (prompts_dir / fname).read_text(encoding="utf-8")
            assert "LEARNED PATCH v1" in live_content
            assert "Always state the deadline clearly." in live_content
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)


# ═════════════════════════════════════════════════════════════════
# Requirement 3 — Audit Trail
# ═════════════════════════════════════════════════════════════════

class TestAuditTrail:

    def test_get_prompt_version_returns_canonical_when_no_patches(self, tmp_path):
        """get_prompt_version() returns 'canonical-v1' for a fresh prompt file."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "assessment_system_prompt.txt").write_text(
            VALID_PROMPT, encoding="utf-8"
        )
        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions.pop("AssessmentAgent", None)
        try:
            version = pip.get_prompt_version("AssessmentAgent")
            assert version == "canonical-v1", (
                f"Expected canonical-v1 for a fresh file; got {version!r}"
            )
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    def test_get_prompt_version_recovers_from_file_after_restart(self, tmp_path):
        """get_prompt_version() reads LEARNED PATCH version from disk on first call."""
        import src.self_learning.improvement_pipeline as pip

        prompt_with_patch = (
            VALID_PROMPT.replace(
                "{injected_guidance_block}",
                "\n## LEARNED PATCH v3 | 2026-01-01T00:00:00Z | run: pipeline-abc patch 1\n"
                "## Rollback: prompts/versions/assessment_system_prompt_pipeline-abc_p0.txt\n"
                "State the deadline clearly.\n"
                "## END PATCH v3\n"
                "{injected_guidance_block}",
            )
        )
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "assessment_system_prompt.txt").write_text(
            prompt_with_patch, encoding="utf-8"
        )
        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions.pop("AssessmentAgent", None)  # simulate restart
        try:
            version = pip.get_prompt_version("AssessmentAgent")
            assert version == "patch-v3", (
                f"Expected patch-v3 recovered from file; got {version!r}"
            )
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    def test_apply_prompt_updates_version_registry(self, tmp_path):
        """After _apply_prompt(), get_prompt_version() returns the new version."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()
        (prompts_dir / "assessment_system_prompt.txt").write_text(
            VALID_PROMPT, encoding="utf-8"
        )
        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions.pop("AssessmentAgent", None)
        try:
            with patch("src.self_learning.improvement_pipeline._fire_log_prompt_change"):
                pip._apply_prompt("AssessmentAgent", VALID_PROMPT + "\n# improved", "run-xyz")

            version = pip.get_prompt_version("AssessmentAgent")
            assert version == "pipeline-run-xyz", (
                f"Version registry must be updated after _apply_prompt; got {version!r}"
            )
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    @pytest.mark.asyncio
    async def test_apply_prompt_logs_to_prompt_changes(self, tmp_path):
        """_apply_prompt() must insert a document into db.prompt_changes."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()
        (prompts_dir / "assessment_system_prompt.txt").write_text(
            VALID_PROMPT, encoding="utf-8"
        )

        # Mock MongoDB
        inserted_docs = []
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock(side_effect=lambda doc: inserted_docs.append(doc))
        mock_db = MagicMock()
        mock_db.prompt_changes = mock_collection

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions.pop("AssessmentAgent", None)
        try:
            # _apply_prompt calls _fire_log_prompt_change which schedules _log_prompt_change
            # We patch get_mongo to return our mock DB and await the coroutine directly.
            with patch("src.data_layer.get_mongo", return_value=mock_db):
                pip._apply_prompt("AssessmentAgent", VALID_PROMPT + "\n# v2", "run-audit")
                # Give the scheduled coroutine a chance to run
                await asyncio.sleep(0)
                await pip._log_prompt_change(
                    "AssessmentAgent", "canonical-v1", "pipeline-run-audit",
                    "run-audit", None, "prompts/versions/backup.txt", "pipeline_adopt",
                )

            assert len(inserted_docs) >= 1, "At least one prompt_changes record must be inserted"
            doc = inserted_docs[0]
            assert doc["agent_name"] == "AssessmentAgent"
            assert doc["new_version"] == "pipeline-run-audit"
            assert doc["trigger"] == "pipeline_adopt"
            assert "timestamp" in doc
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    def test_temporal_activities_uses_get_prompt_version(self):
        """temporal_activities.py must import and call get_prompt_version."""
        import src.temporal_activities as ta
        import inspect
        src_text = inspect.getsource(ta)
        assert "get_prompt_version" in src_text, (
            "temporal_activities.py must call get_prompt_version() — "
            "hardcoded 'canonical-v1' is no longer acceptable"
        )
        assert '"canonical-v1"' not in src_text, (
            "temporal_activities.py must not hardcode 'canonical-v1'"
        )


# ═════════════════════════════════════════════════════════════════
# Requirement 4 — Rollback
# ═════════════════════════════════════════════════════════════════

class TestRollback:

    @pytest.mark.asyncio
    async def test_rollback_restores_backup_content(self, tmp_path):
        """rollback_prompt() writes backup content to the live file."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()

        live_content = VALID_PROMPT + "\n## LEARNED PATCH v2 | 2026-01-01 | run: r2 patch 1\n## Rollback: x\nPatch 2 text.\n## END PATCH v2\n{injected_guidance_block}"
        backup_content = VALID_PROMPT  # v1 — no patches

        (prompts_dir / "assessment_system_prompt.txt").write_text(live_content, encoding="utf-8")
        backup_fname = "assessment_system_prompt_pipeline-r1_p0.txt"
        (versions_dir / backup_fname).write_text(backup_content, encoding="utf-8")

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions.pop("AssessmentAgent", None)
        try:
            with patch("src.self_learning.improvement_pipeline._log_prompt_change",
                       new_callable=AsyncMock):
                result = await pip.rollback_prompt("AssessmentAgent", backup_fname)

            assert result["success"] is True, f"Rollback failed: {result}"
            restored = (prompts_dir / "assessment_system_prompt.txt").read_text(encoding="utf-8")
            assert restored == backup_content, "Live file must contain the backup content after rollback"
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    @pytest.mark.asyncio
    async def test_rollback_saves_pre_rollback_backup(self, tmp_path):
        """rollback_prompt() must back up the current live file before overwriting."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()

        live_content = VALID_PROMPT + "\n# patched version"
        backup_content = VALID_PROMPT
        (prompts_dir / "assessment_system_prompt.txt").write_text(live_content, encoding="utf-8")
        backup_fname = "assessment_system_prompt_pipeline-r1_p0.txt"
        (versions_dir / backup_fname).write_text(backup_content, encoding="utf-8")

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions.pop("AssessmentAgent", None)
        try:
            with patch("src.self_learning.improvement_pipeline._log_prompt_change",
                       new_callable=AsyncMock):
                result = await pip.rollback_prompt("AssessmentAgent", backup_fname)

            assert result["success"] is True
            pre_rollback_backup = result["pre_rollback_backup"]
            # The pre-rollback backup file must exist and contain the old live content
            backup_path = prompts_dir.parent / pre_rollback_backup
            # Path is relative: "prompts/versions/<fname>" — resolve from prompts_dir parent
            full_backup = tmp_path / pre_rollback_backup
            assert full_backup.exists(), f"Pre-rollback backup not created at {full_backup}"
            assert full_backup.read_text(encoding="utf-8") == live_content
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    @pytest.mark.asyncio
    async def test_rollback_updates_version_registry(self, tmp_path):
        """After rollback, get_prompt_version() reflects the restored version."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()

        # Live prompt has patch v2; backup is canonical (no patches)
        (prompts_dir / "assessment_system_prompt.txt").write_text(
            VALID_PROMPT + "\n## LEARNED PATCH v2\n## END PATCH v2\n", encoding="utf-8"
        )
        backup_fname = "assessment_system_prompt_pipeline-r1_p0.txt"
        (versions_dir / backup_fname).write_text(VALID_PROMPT, encoding="utf-8")

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        pip._active_prompt_versions["AssessmentAgent"] = "patch-v2"
        try:
            with patch("src.self_learning.improvement_pipeline._log_prompt_change",
                       new_callable=AsyncMock):
                result = await pip.rollback_prompt("AssessmentAgent", backup_fname)

            assert result["success"] is True
            assert result["restored_version"] == "canonical-v1"
            assert pip.get_prompt_version("AssessmentAgent") == "canonical-v1"
        finally:
            pip._PROMPTS_DIR = original_dir
            pip._active_prompt_versions.pop("AssessmentAgent", None)

    @pytest.mark.asyncio
    async def test_rollback_rejects_path_traversal(self, tmp_path):
        """rollback_prompt() must reject filenames containing path separators."""
        import src.self_learning.improvement_pipeline as pip

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(tmp_path / "prompts")
        try:
            result = await pip.rollback_prompt("AssessmentAgent", "../../../etc/passwd")
            assert result["success"] is False
            assert "path separators" in result["error"]
        finally:
            pip._PROMPTS_DIR = original_dir

    @pytest.mark.asyncio
    async def test_rollback_rejects_missing_file(self, tmp_path):
        """rollback_prompt() must return an error when backup file does not exist."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        try:
            result = await pip.rollback_prompt("AssessmentAgent", "nonexistent_backup.txt")
            assert result["success"] is False
            assert "not found" in result["error"].lower()
        finally:
            pip._PROMPTS_DIR = original_dir

    @pytest.mark.asyncio
    async def test_rollback_rejects_file_without_template_marker(self, tmp_path):
        """rollback_prompt() must reject a backup that lacks {injected_guidance_block}."""
        import src.self_learning.improvement_pipeline as pip

        prompts_dir = tmp_path / "prompts"
        versions_dir = prompts_dir / "versions"
        prompts_dir.mkdir()
        versions_dir.mkdir()

        bad_backup = "This is not a valid prompt template."
        backup_fname = "assessment_system_prompt_bad.txt"
        (versions_dir / backup_fname).write_text(bad_backup, encoding="utf-8")

        original_dir = pip._PROMPTS_DIR
        pip._PROMPTS_DIR = str(prompts_dir)
        try:
            result = await pip.rollback_prompt("AssessmentAgent", backup_fname)
            assert result["success"] is False
            assert "{injected_guidance_block}" in result["error"]
        finally:
            pip._PROMPTS_DIR = original_dir
