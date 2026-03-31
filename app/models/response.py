from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ComplianceFlagType(str, Enum):
    """Categories of compliance flags that can be detected in a call."""

    HIPAA_CONCERN = "hipaa_concern"
    MISINFORMATION = "misinformation"
    RUDENESS = "rudeness"
    PROTOCOL_VIOLATION = "protocol_violation"
    POSITIVE_INTERACTION = "positive_interaction"


class Severity(str, Enum):
    """Severity levels for compliance flags."""

    CRITICAL = "critical"
    MODERATE = "moderate"
    MINOR = "minor"
    POSITIVE = "positive"


class OverallAssessment(str, Enum):
    """Possible overall assessment outcomes for a call."""

    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    ESCALATE = "escalate"


class ComplianceFlag(BaseModel):
    """
    A single compliance flag detected in the call transcript.

    Attributes:
        type: Category of the issue or positive behavior.
        severity: How severe the issue is.
        description: 1-2 sentence description of the specific issue.
        transcript_excerpt: The relevant portion of the transcript.
    """

    type: ComplianceFlagType = Field(
        ..., description="Category of the compliance flag"
    )
    severity: Severity = Field(..., description="Severity level of the flag")
    description: str = Field(
        ...,
        min_length=1,
        description="1-2 sentence description of the specific issue or positive behavior",
    )
    transcript_excerpt: str = Field(
        ...,
        min_length=1,
        description="The relevant portion of the transcript that triggered this flag",
    )


class AgentPerformance(BaseModel):
    """
    Performance metrics for the agent on this call.

    All scores are floats between 0.0 and 1.0.
    Strengths and improvements each have 1-3 items.
    """

    professionalism_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score based on tone, language, and empathy (0.0 to 1.0)",
    )
    accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score based on correctness of information provided (0.0 to 1.0)",
    )
    resolution_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score based on whether the caller's issue was addressed (0.0 to 1.0)",
    )
    strengths: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="1-3 specific things the agent did well",
    )
    improvements: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="1-3 specific areas for improvement",
    )


class QualityAnalysis(BaseModel):
    """
    Complete quality analysis result for a single call transcript.
    This is the primary output schema returned by the /analyze-call endpoint.
    """

    overall_assessment: OverallAssessment = Field(
        ...,
        description='One of "pass", "needs_review", or "escalate"',
    )
    assessment_reasoning: str = Field(
        ...,
        min_length=1,
        description="2-4 sentences explaining the overall assessment",
    )
    compliance_flags: list[ComplianceFlag] = Field(
        default_factory=list,
        description="List of compliance flags detected in the call",
    )
    agent_performance: AgentPerformance = Field(
        ...,
        description="Performance metrics for the agent",
    )
    escalation_required: bool = Field(
        ...,
        description="True only if a critical issue was detected",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="If escalation is required, explain why. Null otherwise.",
    )


class AnalysisError(BaseModel):
    """Structured error response for API errors."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Human-readable error description")
    call_id: Optional[str] = Field(
        default=None, description="Call ID if available"
    )
