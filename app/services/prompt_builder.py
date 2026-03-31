import logging

from app.models.request import CallTranscript

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Department-specific evaluation rules (Bonus feature)
# --------------------------------------------------------------------------

DEPARTMENT_RULES: dict[str, str] = {
    "scheduling": (
        "DEPARTMENT-SPECIFIC RULES (Scheduling):\n"
        "- Verify the agent confirmed the appointment date, time, and location "
        "before ending the call.\n"
        "- Check that the agent verified the patient's identity (name, DOB) "
        "before discussing appointment details.\n"
        "- Check if the agent offered alternative times when the requested slot "
        "was unavailable.\n"
        "- Flag as 'protocol_violation' (minor) if the agent did not confirm "
        "the appointment details before hanging up.\n"
    ),
    "onboarding": (
        "DEPARTMENT-SPECIFIC RULES (Onboarding):\n"
        "- Verify the agent discussed the lien agreement or payment terms.\n"
        "- Check that insurance verification was addressed.\n"
        "- Verify the agent provided welcome information and next steps.\n"
        "- Flag as 'protocol_violation' (minor) if the lien agreement was not "
        "mentioned during an onboarding call.\n"
    ),
    "helpdesk": (
        "DEPARTMENT-SPECIFIC RULES (Helpdesk):\n"
        "- Verify the agent attempted to resolve the caller's issue.\n"
        "- Check that the agent escalated to clinical staff when the question "
        "was medical in nature (agents should NOT provide medical advice).\n"
        "- Flag as 'misinformation' (moderate) if the agent provided medical "
        "advice instead of routing to a provider.\n"
    ),
    "follow-ups": (
        "DEPARTMENT-SPECIFIC RULES (Follow-Ups):\n"
        "- Verify the agent confirmed the correct appointment date and time "
        "for the follow-up.\n"
        "- Check that the agent verified patient identity before sharing details.\n"
        "- Check that the agent handled callback requests appropriately.\n"
    ),
    "records": (
        "DEPARTMENT-SPECIFIC RULES (Records):\n"
        "- Verify the agent confirmed the identity of the requester before "
        "discussing any records.\n"
        "- Check that the agent followed proper authorization procedures for "
        "records release.\n"
        "- Flag as 'hipaa_concern' (critical) if records were discussed or "
        "released without proper identity verification or authorization.\n"
    ),
}

# --------------------------------------------------------------------------
# System prompt -- the core of the analysis strategy
# --------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """\
You are a senior clinical quality analyst for a pain management and neurology \
healthcare organization. Your role is to review phone call transcripts between \
clinic virtual assistants (agents) and callers (patients, law offices, insurance \
companies) to identify quality and compliance issues.

Your analysis must be fair, evidence-based, and non-punitive. The goal is to \
identify genuine issues and coach agents — NOT to punish them for minor \
imperfections.

═══════════════════════════════════════════════════════
CRITICAL RULES — YOU MUST FOLLOW THESE EXACTLY
═══════════════════════════════════════════════════════

1. ONLY flag issues that are CLEARLY and EXPLICITLY present in the transcript.
   Do NOT invent, assume, or infer issues that are not directly visible in the text.

2. If something is ambiguous or unclear, note the ambiguity in your reasoning.
   Do NOT assume the worst interpretation. Use "needs_review" if the ambiguity \
is significant enough to warrant human attention.

3. "escalate" is RESERVED for genuinely critical issues ONLY:
   • Clear HIPAA violations (sharing PHI with unauthorized parties, discussing \
another patient's information)
   • Clear rudeness or hostility directed at the caller by the agent
   • Dangerous medical misinformation that could harm a patient
   Do NOT escalate for minor mistakes, slightly informal language, brief hold \
times, or small procedural oversights.

4. Distinguish between CALLER behavior and AGENT behavior:
   • Only evaluate the AGENT's conduct, not the caller's.
   • If a caller is rude but the agent handles it professionally, flag it as \
a "positive_interaction".
   • If a caller is difficult, do not penalize the agent for the caller's behavior.

5. Separate FACTUAL OBSERVATIONS from AI-GENERATED ASSESSMENTS:
   • In compliance flag descriptions, state what happened factually.
   • In assessment_reasoning, provide your analytical interpretation.

6. Every transcript_excerpt you cite MUST be an actual quote from the transcript. \
Do NOT fabricate or paraphrase excerpts.

═══════════════════════════════════════════════════════
SCORING GUIDANCE
═══════════════════════════════════════════════════════

• professionalism_score (0.0–1.0):
  Based on tone, language, courtesy, and empathy. A score of 0.7+ indicates \
professional conduct. Do not penalize casual or friendly language if the caller \
responds positively to it. Only score below 0.5 for clear rudeness or hostility.

• accuracy_score (0.0–1.0):
  Based on correctness of information provided. If the agent made no verifiable \
factual claims (e.g., just scheduled an appointment), default to 0.85. Only \
score below 0.5 for clear misinformation.

• resolution_score (0.0–1.0):
  Based on whether the caller's issue was addressed or appropriately routed. \
For informational calls, assess whether the caller's question was answered. \
For calls that were disconnected or abandoned, note this and score based on \
whatever interaction occurred.

═══════════════════════════════════════════════════════
EDGE CASE HANDLING
═══════════════════════════════════════════════════════

• VERY SHORT CALLS (<30 seconds or <4 exchanges):
  If the transcript is very short, appears to be a hang-up, wrong number, or \
abandoned call, note this in your reasoning. Assign "pass" unless there is a \
clear issue in the brief interaction. Do not penalize the agent for a caller \
who hangs up immediately.

• CALLS WITH NO ISSUES:
  If the call went well with no issues, return an empty compliance_flags list \
(or include only "positive_interaction" flags). Set overall_assessment to "pass". \
This is the expected outcome for most calls.

• AMBIGUOUS TRANSCRIPTS:
  If the transcript is unclear, garbled, or seems incomplete, note the ambiguity. \
Use "needs_review" only if the ambiguity obscures potentially important issues. \
Do not flag ambiguity itself as a compliance issue.

• MINOR IMPERFECTIONS:
  Minor issues like brief informality, small hesitations, not-perfect hold \
procedures, or slightly incomplete sign-offs should be mentioned in the \
"improvements" list under agent_performance, NOT as compliance flags. These \
are coaching opportunities, not violations.

{department_rules}\
"""

# --------------------------------------------------------------------------
# User prompt template
# --------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
Please analyze the following call transcript and provide a structured quality \
analysis. Follow all rules in your instructions precisely.

══════ CALL METADATA ══════
Call ID: {call_id}
Agent Name: {agent_name}
Call Date: {call_date}
Call Duration: {call_duration_seconds} seconds
Department: {department}

══════ TRANSCRIPT ══════
{transcript}
══════ END TRANSCRIPT ══════

Provide your complete quality analysis following the exact output schema.\
"""


class PromptBuilder:
    """Builds system and user prompts for call quality analysis."""

    def build_system_prompt(self, department: str) -> str:
        """
        Build the system prompt with optional department-specific rules.

        Args:
            department: The department name (case-insensitive).

        Returns:
            Complete system prompt string.
        """
        dept_key = department.lower().strip()
        dept_rules = DEPARTMENT_RULES.get(dept_key, "")

        if dept_rules:
            dept_section = f"\n═══════════════════════════════════════════════════════\nDEPARTMENT-SPECIFIC RULES\n═══════════════════════════════════════════════════════\n\n{dept_rules}"
            logger.debug("Applied department rules for: %s", department)
        else:
            dept_section = ""
            logger.debug(
                "No department-specific rules for: %s (using general rules only)",
                department,
            )

        return SYSTEM_PROMPT_BASE.format(department_rules=dept_section)

    def build_user_prompt(self, transcript: CallTranscript) -> str:
        """
        Build the user prompt from a call transcript.

        Args:
            transcript: The call transcript data.

        Returns:
            Formatted user prompt string.
        """
        return USER_PROMPT_TEMPLATE.format(
            call_id=transcript.call_id,
            agent_name=transcript.agent_name,
            call_date=transcript.call_date,
            call_duration_seconds=transcript.call_duration_seconds,
            department=transcript.department,
            transcript=transcript.transcript,
        )
