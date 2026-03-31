import pytest
from pydantic import ValidationError

from app.models.request import BatchAnalyzeRequest, CallTranscript
from app.models.response import (
    AgentPerformance,
    ComplianceFlag,
    ComplianceFlagType,
    OverallAssessment,
    QualityAnalysis,
    Severity,
)


class TestCallTranscript:
    """Tests for the CallTranscript input model."""

    def test_valid_transcript(self):
        data = {
            "call_id": "CALL-001",
            "agent_name": "Maria Santos",
            "call_date": "2024-11-15",
            "call_duration_seconds": 245,
            "department": "Scheduling",
            "transcript": "Agent: Hello\nCaller: Hi",
        }
        t = CallTranscript(**data)
        assert t.call_id == "CALL-001"
        assert t.call_duration_seconds == 245

    def test_invalid_date_format(self):
        data = {
            "call_id": "CALL-001",
            "agent_name": "Maria Santos",
            "call_date": "11/15/2024",  # Wrong format
            "call_duration_seconds": 245,
            "department": "Scheduling",
            "transcript": "Agent: Hello",
        }
        with pytest.raises(ValidationError) as exc_info:
            CallTranscript(**data)
        assert "call_date" in str(exc_info.value)

    def test_negative_duration(self):
        data = {
            "call_id": "CALL-001",
            "agent_name": "Maria",
            "call_date": "2024-01-01",
            "call_duration_seconds": -5,
            "department": "Scheduling",
            "transcript": "Agent: Hello",
        }
        with pytest.raises(ValidationError):
            CallTranscript(**data)

    def test_empty_call_id(self):
        data = {
            "call_id": "",
            "agent_name": "Maria",
            "call_date": "2024-01-01",
            "call_duration_seconds": 100,
            "department": "Scheduling",
            "transcript": "Agent: Hello",
        }
        with pytest.raises(ValidationError):
            CallTranscript(**data)

    def test_empty_transcript(self):
        data = {
            "call_id": "CALL-001",
            "agent_name": "Maria",
            "call_date": "2024-01-01",
            "call_duration_seconds": 100,
            "department": "Scheduling",
            "transcript": "",
        }
        with pytest.raises(ValidationError):
            CallTranscript(**data)

    def test_zero_duration_valid(self):
        """Duration of 0 should be valid (e.g., missed call)."""
        data = {
            "call_id": "CALL-001",
            "agent_name": "Maria",
            "call_date": "2024-01-01",
            "call_duration_seconds": 0,
            "department": "Scheduling",
            "transcript": "Agent: Hello",
        }
        t = CallTranscript(**data)
        assert t.call_duration_seconds == 0


class TestBatchAnalyzeRequest:
    """Tests for the BatchAnalyzeRequest model."""

    def test_valid_batch(self):
        transcript = {
            "call_id": "CALL-001",
            "agent_name": "Maria",
            "call_date": "2024-01-01",
            "call_duration_seconds": 100,
            "department": "Scheduling",
            "transcript": "Agent: Hello",
        }
        batch = BatchAnalyzeRequest(transcripts=[transcript])
        assert len(batch.transcripts) == 1

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchAnalyzeRequest(transcripts=[])


class TestQualityAnalysis:
    """Tests for the QualityAnalysis output model."""

    def _make_valid_analysis(self, **overrides) -> dict:
        base = {
            "overall_assessment": "pass",
            "assessment_reasoning": "The call was handled professionally.",
            "compliance_flags": [],
            "agent_performance": {
                "professionalism_score": 0.9,
                "accuracy_score": 0.85,
                "resolution_score": 0.95,
                "strengths": ["Professional greeting"],
                "improvements": ["Could confirm details at end"],
            },
            "escalation_required": False,
            "escalation_reason": None,
        }
        base.update(overrides)
        return base

    def test_valid_analysis(self):
        data = self._make_valid_analysis()
        analysis = QualityAnalysis(**data)
        assert analysis.overall_assessment == OverallAssessment.PASS
        assert analysis.escalation_required is False

    def test_escalate_assessment(self):
        data = self._make_valid_analysis(
            overall_assessment="escalate",
            escalation_required=True,
            escalation_reason="HIPAA violation detected",
        )
        analysis = QualityAnalysis(**data)
        assert analysis.overall_assessment == OverallAssessment.ESCALATE

    def test_invalid_assessment_value(self):
        data = self._make_valid_analysis(overall_assessment="fail")
        with pytest.raises(ValidationError):
            QualityAnalysis(**data)

    def test_score_out_of_range(self):
        data = self._make_valid_analysis()
        data["agent_performance"]["professionalism_score"] = 1.5
        with pytest.raises(ValidationError):
            QualityAnalysis(**data)

    def test_negative_score(self):
        data = self._make_valid_analysis()
        data["agent_performance"]["accuracy_score"] = -0.1
        with pytest.raises(ValidationError):
            QualityAnalysis(**data)

    def test_too_many_strengths(self):
        data = self._make_valid_analysis()
        data["agent_performance"]["strengths"] = [
            "a", "b", "c", "d"  # Max is 3
        ]
        with pytest.raises(ValidationError):
            QualityAnalysis(**data)

    def test_empty_strengths_rejected(self):
        data = self._make_valid_analysis()
        data["agent_performance"]["strengths"] = []
        with pytest.raises(ValidationError):
            QualityAnalysis(**data)


class TestComplianceFlag:
    """Tests for the ComplianceFlag model."""

    def test_valid_flag(self):
        flag = ComplianceFlag(
            type=ComplianceFlagType.HIPAA_CONCERN,
            severity=Severity.CRITICAL,
            description="Agent disclosed another patient's information.",
            transcript_excerpt='Agent: "Yeah, that was Mr. Davis."',
        )
        assert flag.type == ComplianceFlagType.HIPAA_CONCERN
        assert flag.severity == Severity.CRITICAL

    def test_positive_flag(self):
        flag = ComplianceFlag(
            type=ComplianceFlagType.POSITIVE_INTERACTION,
            severity=Severity.POSITIVE,
            description="Agent handled a difficult caller with patience.",
            transcript_excerpt="Agent: I completely understand your frustration.",
        )
        assert flag.severity == Severity.POSITIVE

    def test_invalid_flag_type(self):
        with pytest.raises(ValidationError):
            ComplianceFlag(
                type="invalid_type",
                severity="critical",
                description="Test",
                transcript_excerpt="Test",
            )
