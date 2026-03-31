import json
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from app.models.response import QualityAnalysis, ComplianceFlagType

BASE_URL = "http://localhost:8000"
SAMPLE_DIR = Path(__file__).parent.parent / "sample_transcripts"

# --------------------------------------------------------------------------
# Expected outcomes for each sample transcript
# --------------------------------------------------------------------------

EXPECTATIONS = {
    "clean_call.json": {
        "expected_assessment": "pass",
        "should_escalate": False,
        "description": "Clean professional scheduling call",
    },
    "problematic_call.json": {
        "expected_assessment": "escalate",
        "should_escalate": True,
        "expected_flag_types": ["hipaa_concern", "rudeness"],
        "description": "HIPAA violation + rudeness",
    },
    "edge_case_short.json": {
        "expected_assessment": "pass",
        "should_escalate": False,
        "description": "Very short wrong-number call (12s)",
    },
    "edge_case_no_issues.json": {
        "expected_assessment_not": "escalate",
        "should_escalate": False,
        "description": "Normal call with minor imperfections",
    },
    "scheduling_call.json": {
        "expected_assessment_not": "escalate",
        "should_escalate": False,
        "description": "Scheduling call — missed confirmation",
    },
}


def print_header():
    print("\n" + "=" * 70)
    print("  CALL QA ANALYZER — EVALUATION SUITE")
    print("=" * 70)


def print_result(name: str, passed: bool, details: str):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n  {status}  {name}")
    print(f"         {details}")


def validate_schema(data: dict) -> tuple[bool, str]:
    """Validate response conforms to QualityAnalysis schema."""
    try:
        QualityAnalysis.model_validate(data)
        return True, "Schema valid"
    except ValidationError as e:
        return False, f"Schema validation failed: {e}"


def main():
    print_header()

    total = 0
    passed = 0
    failed_tests = []

    # Check server is running
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"\n  [ERROR] Server not healthy: {health.status_code}")
            sys.exit(1)
        print(f"\n  [OK] Server is healthy at {BASE_URL}")
    except httpx.ConnectError:
        print(f"\n  [ERROR] Cannot connect to server at {BASE_URL}")
        print("     Start it with: uvicorn app.main:app --reload")
        sys.exit(1)

    print("\n" + "-" * 70)

    for filename, expectations in EXPECTATIONS.items():
        filepath = SAMPLE_DIR / filename
        if not filepath.exists():
            print_result(filename, False, f"Sample file not found: {filepath}")
            failed_tests.append(filename)
            total += 1
            continue

        with open(filepath) as f:
            payload = json.load(f)

        print(f"\n  Testing: {filename}")
        print(f"     {expectations['description']}")

        try:
            response = httpx.post(
                f"{BASE_URL}/analyze-call",
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:
                print_result(
                    filename,
                    False,
                    f"HTTP {response.status_code}: {response.text[:200]}",
                )
                failed_tests.append(filename)
                total += 1
                continue

            data = response.json()

            # Test 1: Schema validation
            total += 1
            schema_ok, schema_msg = validate_schema(data)
            print_result(f"{filename} — Schema", schema_ok, schema_msg)
            if schema_ok:
                passed += 1
            else:
                failed_tests.append(f"{filename} (schema)")

            result = QualityAnalysis.model_validate(data)

            # Test 2: Assessment check
            total += 1
            if "expected_assessment" in expectations:
                assessment_ok = (
                    result.overall_assessment.value
                    == expectations["expected_assessment"]
                )
                print_result(
                    f"{filename} — Assessment",
                    assessment_ok,
                    f"Expected '{expectations['expected_assessment']}', "
                    f"got '{result.overall_assessment.value}'",
                )
            elif "expected_assessment_not" in expectations:
                assessment_ok = (
                    result.overall_assessment.value
                    != expectations["expected_assessment_not"]
                )
                print_result(
                    f"{filename} — Assessment",
                    assessment_ok,
                    f"Expected NOT '{expectations['expected_assessment_not']}', "
                    f"got '{result.overall_assessment.value}'",
                )
            else:
                assessment_ok = True

            if assessment_ok:
                passed += 1
            else:
                failed_tests.append(f"{filename} (assessment)")

            # Test 3: Escalation check
            total += 1
            escalation_ok = (
                result.escalation_required == expectations["should_escalate"]
            )
            print_result(
                f"{filename} — Escalation",
                escalation_ok,
                f"Expected escalation={expectations['should_escalate']}, "
                f"got {result.escalation_required}",
            )
            if escalation_ok:
                passed += 1
            else:
                failed_tests.append(f"{filename} (escalation)")

            # Test 4: Expected flag types (if specified)
            if "expected_flag_types" in expectations:
                total += 1
                actual_types = {f.type.value for f in result.compliance_flags}
                expected_types = set(expectations["expected_flag_types"])
                flags_ok = expected_types.issubset(actual_types)
                print_result(
                    f"{filename} — Flag Types",
                    flags_ok,
                    f"Expected {expected_types} in {actual_types}",
                )
                if flags_ok:
                    passed += 1
                else:
                    failed_tests.append(f"{filename} (flags)")

            # Print analysis summary
            print(f"\n     Assessment: {result.overall_assessment.value}")
            print(f"     Reasoning: {result.assessment_reasoning[:100]}...")
            print(f"     Flags: {len(result.compliance_flags)}")
            print(
                f"     Scores: prof={result.agent_performance.professionalism_score:.2f} "
                f"acc={result.agent_performance.accuracy_score:.2f} "
                f"res={result.agent_performance.resolution_score:.2f}"
            )

        except Exception as e:
            total += 1
            print_result(filename, False, f"Error: {str(e)}")
            failed_tests.append(filename)

    # Summary
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed}/{total} tests passed")
    if failed_tests:
        print(f"  FAILED: {', '.join(failed_tests)}")
    else:
        print("  All tests passed!")
    print("=" * 70 + "\n")

    sys.exit(0 if not failed_tests else 1)


if __name__ == "__main__":
    main()
