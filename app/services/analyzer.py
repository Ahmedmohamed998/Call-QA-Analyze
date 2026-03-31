import logging

from app.models.request import CallTranscript
from app.models.response import (
    OverallAssessment,
    QualityAnalysis,
    Severity,
)
from app.providers.base import LLMProvider
from app.services.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class CallAnalyzer:
    """
    Orchestrates call quality analysis by coordinating the LLM provider
    and prompt builder, then applying deterministic post-processing rules.
    """

    def __init__(self, provider: LLMProvider, prompt_builder: PromptBuilder):
        self.provider = provider
        self.prompt_builder = prompt_builder

    async def analyze_call(self, transcript: CallTranscript) -> QualityAnalysis:
        """
        Analyze a single call transcript.

        Steps:
          1. Build system prompt (with department-specific rules)
          2. Build user prompt (with transcript data)
          3. Call LLM for structured analysis
          4. Apply post-processing business rules

        Args:
            transcript: The call transcript to analyze.

        Returns:
            Validated QualityAnalysis with business rules enforced.
        """
        logger.info(
            "Analyzing call | call_id=%s | department=%s | duration=%ds",
            transcript.call_id,
            transcript.department,
            transcript.call_duration_seconds,
        )

        # Step 1–2: Build prompts
        system_prompt = self.prompt_builder.build_system_prompt(
            transcript.department
        )
        user_prompt = self.prompt_builder.build_user_prompt(transcript)

        logger.debug(
            "Prompts built | system_prompt_len=%d | user_prompt_len=%d",
            len(system_prompt),
            len(user_prompt),
        )

        # Step 3: Call LLM
        result = await self.provider.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=QualityAnalysis,
        )

        # Step 4: Post-processing
        result = self._post_process(result)

        logger.info(
            "Analysis complete | call_id=%s | assessment=%s | "
            "flags=%d | escalation=%s",
            transcript.call_id,
            result.overall_assessment.value,
            len(result.compliance_flags),
            result.escalation_required,
        )

        return result

    def _post_process(self, result: QualityAnalysis) -> QualityAnalysis:
        """
        Apply deterministic business rules to the LLM output.

        Even with structured outputs, LLMs can be inconsistent with
        business logic. These guards ensure:
          1. Critical flags always trigger escalation
          2. Escalation always has a reason
          3. Overall assessment matches escalation status
          4. Non-escalation calls don't have escalation set

        Args:
            result: Raw QualityAnalysis from the LLM.

        Returns:
            QualityAnalysis with business rules enforced.
        """
        # Rule 1: If any critical flag exists, escalation must be required
        critical_flags = [
            f
            for f in result.compliance_flags
            if f.severity == Severity.CRITICAL
        ]
        has_critical = len(critical_flags) > 0

        if has_critical and not result.escalation_required:
            logger.warning(
                "Post-processing: Forcing escalation due to critical flag(s)"
            )
            result.escalation_required = True

        # Rule 2: If escalation is required, there must be a reason
        if result.escalation_required and not result.escalation_reason:
            reasons = [f.description for f in critical_flags]
            result.escalation_reason = "; ".join(reasons) if reasons else (
                "Critical issue detected requiring human review"
            )
            logger.warning(
                "Post-processing: Auto-generated escalation reason"
            )

        # Rule 3: If escalation is required, overall must be "escalate"
        if (
            result.escalation_required
            and result.overall_assessment != OverallAssessment.ESCALATE
        ):
            logger.warning(
                "Post-processing: Correcting overall_assessment to 'escalate'"
            )
            result.overall_assessment = OverallAssessment.ESCALATE

        # Rule 4: If overall is "escalate", escalation must be required
        if (
            result.overall_assessment == OverallAssessment.ESCALATE
            and not result.escalation_required
        ):
            result.escalation_required = True
            if not result.escalation_reason:
                result.escalation_reason = (
                    "Overall assessment indicates escalation is needed"
                )

        # Rule 5: If no escalation, reason should be null
        if not result.escalation_required and result.escalation_reason:
            logger.debug(
                "Post-processing: Clearing escalation_reason (no escalation)"
            )
            result.escalation_reason = None

        return result
