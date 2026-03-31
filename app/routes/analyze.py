import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.models.request import BatchAnalyzeRequest, CallTranscript
from app.models.response import AnalysisError, QualityAnalysis
from app.providers.base import LLMProviderError
from app.services.analyzer import CallAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter()

# This will be set during app startup via dependency injection
_analyzer: CallAnalyzer | None = None


def set_analyzer(analyzer: CallAnalyzer) -> None:
    """Inject the analyzer instance (called during app startup)."""
    global _analyzer
    _analyzer = analyzer


def get_analyzer() -> CallAnalyzer:
    """Get the current analyzer instance."""
    if _analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Please try again in a moment.",
        )
    return _analyzer


@router.post(
    "/analyze-call",
    response_model=QualityAnalysis,
    summary="Analyze a single call transcript",
    description=(
        "Receives a phone call transcript and returns a structured "
        "quality analysis including compliance flags, agent performance "
        "scores, and escalation decisions."
    ),
    responses={
        200: {
            "description": "Successful analysis",
            "model": QualityAnalysis,
        },
        422: {
            "description": "Validation error - invalid input payload",
        },
        503: {
            "description": "LLM provider unavailable after retries",
            "model": AnalysisError,
        },
    },
)
async def analyze_call(request: CallTranscript) -> QualityAnalysis:
    """
    Analyze a single call transcript for quality and compliance.

    The endpoint sends the transcript to the configured LLM provider,
    which returns a structured analysis including:
    - Overall assessment (pass / needs_review / escalate)
    - Compliance flags with severity and evidence
    - Agent performance scores
    - Escalation decision with reasoning
    """
    analyzer = get_analyzer()

    try:
        result = await analyzer.analyze_call(request)
        return result

    except LLMProviderError as e:
        logger.error("LLM analysis failed for call %s: %s", request.call_id, str(e))
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider error: {str(e)}. Please try again later.",
        )
    except Exception as e:
        logger.error(
            "Unexpected error analyzing call %s: %s",
            request.call_id,
            str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during analysis: {str(e)}",
        )


@router.post(
    "/batch-analyze",
    response_model=list[QualityAnalysis],
    summary="Analyze multiple call transcripts",
    description=(
        "Accepts a batch of up to 50 call transcripts and returns "
        "analysis results for all of them. Transcripts are processed "
        "concurrently for efficiency."
    ),
    responses={
        200: {
            "description": "Successful batch analysis",
            "model": list[QualityAnalysis],
        },
        422: {
            "description": "Validation error - invalid input payload",
        },
        503: {
            "description": "LLM provider unavailable",
            "model": AnalysisError,
        },
    },
)
async def batch_analyze(request: BatchAnalyzeRequest) -> list[QualityAnalysis]:
    """
    Analyze multiple call transcripts concurrently.

    Processes all transcripts in parallel using asyncio.gather.
    Returns results in the same order as the input transcripts.
    If any individual analysis fails, the entire batch returns an error.
    """
    analyzer = get_analyzer()

    logger.info(
        "Starting batch analysis | count=%d", len(request.transcripts)
    )

    try:
        # Process all transcripts concurrently
        tasks = [
            analyzer.analyze_call(transcript)
            for transcript in request.transcripts
        ]
        results = await asyncio.gather(*tasks)

        logger.info(
            "Batch analysis complete | count=%d | assessments=%s",
            len(results),
            [r.overall_assessment.value for r in results],
        )

        return list(results)

    except LLMProviderError as e:
        logger.error("Batch analysis failed: %s", str(e))
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider error during batch analysis: {str(e)}",
        )
    except Exception as e:
        logger.error("Batch analysis error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during batch analysis: {str(e)}",
        )
