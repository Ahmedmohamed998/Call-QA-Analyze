from pydantic import BaseModel, Field


class CallTranscript(BaseModel):
    """
    Represents a single phone call transcript submitted for quality analysis.

    Attributes:
        call_id: Unique identifier for the call.
        agent_name: Name of the agent who handled the call.
        call_date: Date of the call in YYYY-MM-DD format.
        call_duration_seconds: Duration of the call in seconds.
        department: The department handling the call.
        transcript: Multi-turn conversation text with "Agent:" and "Caller:" lines.
    """

    call_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the call",
        examples=["CALL-2024-00142"],
    )
    agent_name: str = Field(
        ...,
        min_length=1,
        description="Name of the agent who handled the call",
        examples=["Maria Santos"],
    )
    call_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date of the call in YYYY-MM-DD format",
        examples=["2024-11-15"],
    )
    call_duration_seconds: int = Field(
        ...,
        ge=0,
        description="Duration of the call in seconds",
        examples=[245],
    )
    department: str = Field(
        ...,
        min_length=1,
        description=(
            'Department handling the call (e.g., "Scheduling", '
            '"Onboarding", "Helpdesk", "Follow-Ups", "Records")'
        ),
        examples=["Scheduling"],
    )
    transcript: str = Field(
        ...,
        min_length=1,
        description=(
            "Multi-turn conversation between agent and caller, "
            'formatted as "Agent: ..." and "Caller: ..." lines'
        ),
    )


class BatchAnalyzeRequest(BaseModel):
    """
    Request payload for batch analysis of multiple transcripts.

    Attributes:
        transcripts: List of call transcripts to analyze (1-50).
    """

    transcripts: list[CallTranscript] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of call transcripts to analyze",
    )
