import pytest

from app.models.response import (
    AgentPerformance,
    ComplianceFlag,
    ComplianceFlagType,
    OverallAssessment,
    QualityAnalysis,
    Severity,
)
from app.services.analyzer import CallAnalyzer
from app.services.prompt_builder import PromptBuilder


def _make_performance(**overrides) -> AgentPerformance:
    """Helper to create a valid AgentPerformance."""
    defaults = {
        "professionalism_score": 0.85,
        "accuracy_score": 0.85,
        "resolution_score": 0.9,
        "strengths": ["Professional conduct"],
        "improvements": ["Minor areas to improve"],
    }
    defaults.update(overrides)
    return AgentPerformance(**defaults)


def _make_analysis(**overrides) -> QualityAnalysis:
    """Helper to create a valid QualityAnalysis for testing post-processing."""
    defaults = {
        "overall_assessment": OverallAssessment.PASS,
        "assessment_reasoning": "The call was handled well overall.",
        "compliance_flags": [],
        "agent_performance": _make_performance(),
        "escalation_required": False,
        "escalation_reason": None,
    }
    defaults.update(overrides)
    return QualityAnalysis(**defaults)


class TestPostProcessing:
    """Tests for CallAnalyzer._post_process business rules."""

    def setup_method(self):
        """Create an analyzer with no provider (we only test post-processing)."""
        self.analyzer = CallAnalyzer(
            provider=None,  # type: ignore
            prompt_builder=PromptBuilder(),
        )

    def test_clean_analysis_unchanged(self):
        """A clean analysis should pass through unchanged."""
        analysis = _make_analysis()
        result = self.analyzer._post_process(analysis)
        assert result.overall_assessment == OverallAssessment.PASS
        assert result.escalation_required is False
        assert result.escalation_reason is None

    def test_critical_flag_forces_escalation(self):
        """If a critical flag exists but escalation is not set, force it."""
        flag = ComplianceFlag(
            type=ComplianceFlagType.HIPAA_CONCERN,
            severity=Severity.CRITICAL,
            description="Agent disclosed patient information.",
            transcript_excerpt="Agent mentioned Mr. Davis by name.",
        )
        analysis = _make_analysis(
            compliance_flags=[flag],
            escalation_required=False,  # LLM forgot to set this
            overall_assessment=OverallAssessment.NEEDS_REVIEW,
        )
        result = self.analyzer._post_process(analysis)

        assert result.escalation_required is True
        assert result.overall_assessment == OverallAssessment.ESCALATE
        assert result.escalation_reason is not None

    def test_escalation_without_reason_gets_auto_reason(self):
        """If escalation is required but no reason given, auto-generate one."""
        flag = ComplianceFlag(
            type=ComplianceFlagType.RUDENESS,
            severity=Severity.CRITICAL,
            description="Agent was dismissive and rude.",
            transcript_excerpt="I don't have all day for this.",
        )
        analysis = _make_analysis(
            compliance_flags=[flag],
            escalation_required=True,
            escalation_reason=None,  # LLM forgot the reason
            overall_assessment=OverallAssessment.ESCALATE,
        )
        result = self.analyzer._post_process(analysis)

        assert result.escalation_reason is not None
        assert "dismissive" in result.escalation_reason.lower() or len(result.escalation_reason) > 0

    def test_escalate_assessment_forces_escalation_required(self):
        """If overall is 'escalate' but escalation_required is False, fix it."""
        analysis = _make_analysis(
            overall_assessment=OverallAssessment.ESCALATE,
            escalation_required=False,
        )
        result = self.analyzer._post_process(analysis)

        assert result.escalation_required is True
        assert result.escalation_reason is not None

    def test_no_escalation_clears_reason(self):
        """If no escalation needed, reason should be null."""
        analysis = _make_analysis(
            overall_assessment=OverallAssessment.PASS,
            escalation_required=False,
            escalation_reason="This should be cleared",
        )
        result = self.analyzer._post_process(analysis)

        assert result.escalation_reason is None

    def test_minor_flags_dont_force_escalation(self):
        """Minor/moderate flags should NOT trigger escalation."""
        flag = ComplianceFlag(
            type=ComplianceFlagType.PROTOCOL_VIOLATION,
            severity=Severity.MINOR,
            description="Agent did not confirm appointment details.",
            transcript_excerpt="Alright, Thomas, I'll get you set up.",
        )
        analysis = _make_analysis(
            compliance_flags=[flag],
            overall_assessment=OverallAssessment.NEEDS_REVIEW,
        )
        result = self.analyzer._post_process(analysis)

        assert result.escalation_required is False
        assert result.overall_assessment == OverallAssessment.NEEDS_REVIEW

    def test_positive_flags_dont_affect_escalation(self):
        """Positive interaction flags should not trigger escalation."""
        flag = ComplianceFlag(
            type=ComplianceFlagType.POSITIVE_INTERACTION,
            severity=Severity.POSITIVE,
            description="Agent handled the call with great empathy.",
            transcript_excerpt="I completely understand your concern.",
        )
        analysis = _make_analysis(
            compliance_flags=[flag],
            overall_assessment=OverallAssessment.PASS,
        )
        result = self.analyzer._post_process(analysis)

        assert result.escalation_required is False
        assert result.overall_assessment == OverallAssessment.PASS


class TestPromptBuilder:
    """Tests for the PromptBuilder."""

    def setup_method(self):
        self.builder = PromptBuilder()

    def test_scheduling_department_rules(self):
        prompt = self.builder.build_system_prompt("Scheduling")
        assert "appointment" in prompt.lower()
        assert "DEPARTMENT-SPECIFIC RULES" in prompt

    def test_onboarding_department_rules(self):
        prompt = self.builder.build_system_prompt("Onboarding")
        assert "lien agreement" in prompt.lower()

    def test_unknown_department_no_rules(self):
        prompt = self.builder.build_system_prompt("Unknown-Dept")
        assert "DEPARTMENT-SPECIFIC RULES" not in prompt

    def test_case_insensitive_department(self):
        prompt = self.builder.build_system_prompt("SCHEDULING")
        assert "DEPARTMENT-SPECIFIC RULES" in prompt

    def test_user_prompt_includes_metadata(self):
        from app.models.request import CallTranscript

        transcript = CallTranscript(
            call_id="TEST-001",
            agent_name="Test Agent",
            call_date="2024-01-01",
            call_duration_seconds=120,
            department="Scheduling",
            transcript="Agent: Hello\nCaller: Hi",
        )
        prompt = self.builder.build_user_prompt(transcript)
        assert "TEST-001" in prompt
        assert "Test Agent" in prompt
        assert "120 seconds" in prompt
        assert "Agent: Hello" in prompt
