"""Chapter 4 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch04_todo_01..06) and Coding Exam (ch04_exam_01..03).
"""

from __future__ import annotations

import math
from typing import Any

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 4.1: WHO BMI Classification (if / elif / else)
# ---------------------------------------------------------------------------
def _verify_todo_01(val: Any) -> CheckResult:
    if val is None:
        return CheckResult(
            "ch04_todo_01",
            False,
            "unattempted",
            "To-Do 4.1 — Ready for Your Solution",
            "Assign bmi_category using an if / elif / else chain on bmi = 27.4.",
        )
    if not isinstance(val, str):
        return CheckResult(
            "ch04_todo_01",
            False,
            "failed",
            "To-Do 4.1 — Expected Category String",
            f"Expected a str ('Underweight', 'Normal', 'Overweight', or 'Obese'), got {type(val).__name__} ({val!r}).",
        )
    if val.strip().lower() == "overweight" and val != "Overweight":
        return CheckResult(
            "ch04_todo_01",
            False,
            "failed",
            "To-Do 4.1 — Check Exact Capitalization",
            f"Almost! Expected exact string 'Overweight', got {val!r}.",
        )
    if val != "Overweight":
        return CheckResult(
            "ch04_todo_01",
            False,
            "failed",
            "To-Do 4.1 — Check WHO BMI Cutoff Boundaries",
            f"For bmi = 27.4 (25.0 <= bmi < 30.0), expected 'Overweight', got {val!r}.",
            details=[
                "WHO adult cutoffs: < 18.5 Underweight | < 25.0 Normal | < 30.0 Overweight | >= 30.0 Obese.",
                "Make sure you used elif (not separate if statements) so earlier matches are not overwritten.",
            ],
        )
    return CheckResult(
        "ch04_todo_01",
        True,
        "passed",
        "To-Do 4.1 Passed — WHO BMI Classifier Verified!",
        "bmi = 27.4 correctly routed to 'Overweight' via mutually exclusive elif branches.",
        tip="Ordered if / elif / else chains guarantee that exactly one clinical category is assigned.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_todo_01",
        chapter="ch04",
        title="To-Do 4.1: WHO BMI Classification",
        book_section="Section 4.3.3 (if / elif / else Decision Trees)",
        points=10,
        hints={
            1: "Check BMI thresholds in ascending order using if / elif / elif / else: < 18.5, < 25.0, < 30.0, and else.",
            2: "Algorithm:\n1. if bmi < 18.5: bmi_category = 'Underweight'\n2. elif bmi < 25.0: bmi_category = 'Normal'\n3. elif bmi < 30.0: bmi_category = 'Overweight'\n4. else: bmi_category = 'Obese'",
            3: "if bmi < 18.5:\n    bmi_category = 'Underweight'\nelif bmi < 25.0:\n    bmi_category = 'Normal'\nelif bmi < ___:\n    bmi_category = 'Overweight'\nelse:\n    bmi_category = 'Obese'",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 4.2: Chained Comparisons — Normal Vital Signs Range
# ---------------------------------------------------------------------------
def _verify_todo_02(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 4:
        return CheckResult(
            "ch04_todo_02",
            False,
            "failed",
            "To-Do 4.2 — Pass All Four Boolean Flags",
            "Pass (temp_normal, sbp_normal, spo2_normal, all_vitals_normal) into tutor.check('ch04_todo_02', ...).",
        )
    temp_normal, sbp_normal, spo2_normal, all_vitals_normal = val
    if any(x is None for x in val):
        return CheckResult(
            "ch04_todo_02",
            False,
            "unattempted",
            "To-Do 4.2 — Ready for Your Solution",
            "Replace all None placeholders with chained comparison expressions and logical and.",
        )

    errors = []
    if type(temp_normal) is not bool or temp_normal is not True:
        errors.append(f"temp_normal: expected bool True (36.1 <= 36.8 <= 37.2), got {temp_normal!r}")
    if type(sbp_normal) is not bool or sbp_normal is not True:
        errors.append(f"sbp_normal: expected bool True (90 <= 114 < 120), got {sbp_normal!r}")
    if type(spo2_normal) is not bool or spo2_normal is not True:
        errors.append(f"spo2_normal: expected bool True (95 <= 98 <= 100), got {spo2_normal!r}")
    if type(all_vitals_normal) is not bool or all_vitals_normal is not True:
        errors.append(f"all_vitals_normal: expected bool True (all three flags True), got {all_vitals_normal!r}")

    if errors:
        return CheckResult(
            "ch04_todo_02",
            False,
            "failed",
            "To-Do 4.2 — Check Chained Vital-Sign Ranges",
            "Ensure each vital-sign check evaluates to a real Python bool using clinical range bounds.",
            details=errors,
        )
    return CheckResult(
        "ch04_todo_02",
        True,
        "passed",
        "To-Do 4.2 Passed — Chained Vital-Sign Checks Verified!",
        "All three clinical ranges passed: temp_normal=True, sbp_normal=True, spo2_normal=True, all_vitals_normal=True.",
        tip="Python's chained comparisons (90 <= sbp_mmhg < 120) mirror textbook math notation and reduce typos.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_todo_02",
        chapter="ch04",
        title="To-Do 4.2: Chained Comparisons for Normal Vitals",
        book_section="Section 4.3.1 (Chained Comparisons)",
        points=10,
        hints={
            1: "Use Python chained inequalities like low <= x <= high for each vital sign, then combine all three with and.",
            2: "Bounds:\n- temp_normal: 36.1 <= temp_c <= 37.2\n- sbp_normal: 90 <= sbp_mmhg < 120\n- spo2_normal: 95 <= spo2_pct <= 100\n- all_vitals_normal: temp_normal and sbp_normal and spo2_normal",
            3: "temp_normal = 36.1 <= temp_c <= 37.2\nsbp_normal = 90 <= sbp_mmhg < ___\nspo2_normal = 95 <= spo2_pct <= ___\nall_vitals_normal = temp_normal and sbp_normal and ___",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 4.3: Boolean Algebra (and / or / not) — Vaccine Trial Screen
# ---------------------------------------------------------------------------
def _verify_todo_03(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 4:
        return CheckResult(
            "ch04_todo_03",
            False,
            "failed",
            "To-Do 4.3 — Pass All Four Eligibility Booleans",
            "Pass (age_ok, medically_safe, can_enroll, needs_review) into tutor.check('ch04_todo_03', ...).",
        )
    age_ok, medically_safe, can_enroll, needs_review = val
    if any(x is None for x in val):
        return CheckResult(
            "ch04_todo_03",
            False,
            "unattempted",
            "To-Do 4.3 — Ready for Your Solution",
            "Fill in age_ok, medically_safe, can_enroll, and needs_review using and / or / not.",
        )

    errors = []
    if type(age_ok) is not bool or age_ok is not True:
        errors.append(f"age_ok: expected bool True (18 <= 24 <= 65), got {age_ok!r}")
    if type(medically_safe) is not bool or medically_safe is not True:
        errors.append(f"medically_safe: expected bool True ((not is_pregnant) and (not has_severe_allergy)), got {medically_safe!r}")
    if type(can_enroll) is not bool or can_enroll is not True:
        errors.append(f"can_enroll: expected bool True (age_ok and medically_safe and consented), got {can_enroll!r}")
    if type(needs_review) is not bool or needs_review is not False:
        errors.append(f"needs_review: expected bool False (is_pregnant or has_severe_allergy), got {needs_review!r}")

    if errors:
        return CheckResult(
            "ch04_todo_03",
            False,
            "failed",
            "To-Do 4.3 — Check Boolean Algebra Logic",
            "Review the truth tables for and, or, and not—especially De Morgan's rule for medically_safe.",
            details=errors,
        )
    return CheckResult(
        "ch04_todo_03",
        True,
        "passed",
        "To-Do 4.3 Passed — Trial Inclusion/Exclusion Logic Verified!",
        "age_ok=True, medically_safe=True, can_enroll=True, needs_review=False.",
        tip="Notice De Morgan's equivalence: (not is_pregnant) and (not has_severe_allergy) == not (is_pregnant or has_severe_allergy).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_todo_03",
        chapter="ch04",
        title="To-Do 4.3: Vaccine Trial Eligibility Truth Table",
        book_section="Section 4.3.2 (Boolean Algebra & Precedence)",
        points=10,
        hints={
            1: "A volunteer is medically_safe only when they are NOT pregnant AND do NOT have a severe allergy.",
            2: "Algorithm:\n1. age_ok = 18 <= age_years <= 65\n2. medically_safe = (not is_pregnant) and (not has_severe_allergy)\n3. can_enroll = age_ok and medically_safe and consented\n4. needs_review = is_pregnant or has_severe_allergy",
            3: "age_ok = 18 <= age_years <= 65\nmedically_safe = (not is_pregnant) and (not ___)\ncan_enroll = age_ok and medically_safe and ___\nneeds_review = is_pregnant or ___",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 4.4: WHO Maternal Anemia Severity Ladder
# ---------------------------------------------------------------------------
def _verify_todo_04(val: Any) -> CheckResult:
    if val is None:
        return CheckResult(
            "ch04_todo_04",
            False,
            "unattempted",
            "To-Do 4.4 — Ready for Your Solution",
            "Classify hb_g_dl = 8.9 into 'Severe', 'Moderate', 'Mild', or 'Normal'.",
        )
    if not isinstance(val, str):
        return CheckResult(
            "ch04_todo_04",
            False,
            "failed",
            "To-Do 4.4 — Expected Severity Label String",
            f"Expected a str severity label, got {type(val).__name__} ({val!r}).",
        )
    if val == "Mild":
        return CheckResult(
            "ch04_todo_04",
            False,
            "failed",
            "To-Do 4.4 — Classic if-vs-elif Overwrite Trap!",
            "Got 'Mild' for hb_g_dl = 8.9! Because 8.9 < 10.0 AND 8.9 < 11.0, using separate 'if' statements overwrote 'Moderate' with 'Mild'.",
            details=["Replace the second and third 'if' statements with 'elif' so Python stops at the first matching threshold."],
        )
    if val != "Moderate":
        return CheckResult(
            "ch04_todo_04",
            False,
            "failed",
            "To-Do 4.4 — Check WHO Maternal Anemia Thresholds",
            f"For hb_g_dl = 8.9, expected 'Moderate' (7.0 <= Hb < 10.0), got {val!r}.",
            details=["WHO pregnancy thresholds: < 7.0 Severe | < 10.0 Moderate | < 11.0 Mild | >= 11.0 Normal."],
        )
    return CheckResult(
        "ch04_todo_04",
        True,
        "passed",
        "To-Do 4.4 Passed — WHO Maternal Anemia Severity Verified!",
        "hb_g_dl = 8.9 correctly classified as 'Moderate' without falling through to 'Mild'.",
        tip="In medical grading ladders, always order thresholds strictly from most severe to least severe (or vice versa) with elif.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_todo_04",
        chapter="ch04",
        title="To-Do 4.4: WHO Maternal Anemia Severity",
        book_section="Section 4.3.3 (if vs elif Clinical Bug)",
        points=10,
        hints={
            1: "Test the strictest cutoff first (< 7.0), then use elif for < 10.0 and < 11.0, and finish with else for 'Normal'.",
            2: "Algorithm:\n1. if hb_g_dl < 7.0: anemia_grade = 'Severe'\n2. elif hb_g_dl < 10.0: anemia_grade = 'Moderate'\n3. elif hb_g_dl < 11.0: anemia_grade = 'Mild'\n4. else: anemia_grade = 'Normal'",
            3: "if hb_g_dl < 7.0:\n    anemia_grade = 'Severe'\nelif hb_g_dl < ___:\n    anemia_grade = 'Moderate'\nelif hb_g_dl < 11.0:\n    anemia_grade = 'Mild'\nelse:\n    anemia_grade = 'Normal'",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 4.5: Short-Circuit Zero-Population Guard
# ---------------------------------------------------------------------------
def _verify_todo_05(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 3:
        return CheckResult(
            "ch04_todo_05",
            False,
            "failed",
            "To-Do 4.5 — Pass All Three Guarded Outputs",
            "Pass (has_valid_pop, is_high_incidence, incidence_per_100k) into tutor.check('ch04_todo_05', ...).",
        )
    has_valid_pop, is_high_incidence, incidence_per_100k = val
    if has_valid_pop is None or is_high_incidence is None:
        return CheckResult(
            "ch04_todo_05",
            False,
            "unattempted",
            "To-Do 4.5 — Ready for Your Solution",
            "Complete has_valid_pop, is_high_incidence (using short-circuit and), and incidence_per_100k.",
        )

    errors = []
    if type(has_valid_pop) is not bool or has_valid_pop is not False:
        errors.append(f"has_valid_pop: expected bool False (since population == 0), got {has_valid_pop!r}")
    if type(is_high_incidence) is not bool or is_high_incidence is not False:
        errors.append(f"is_high_incidence: expected bool False (short-circuited safely!), got {is_high_incidence!r}")
    if incidence_per_100k is not None:
        errors.append(
            f"incidence_per_100k: expected None when population == 0 (never invent 0.0 for missing rate!), got {incidence_per_100k!r}"
        )

    if errors:
        return CheckResult(
            "ch04_todo_05",
            False,
            "failed",
            "To-Do 4.5 — Check Zero-Population Guard Logic",
            "Place population > 0 FIRST before dividing by population, and return None when population <= 0.",
            details=errors,
        )
    return CheckResult(
        "ch04_todo_05",
        True,
        "passed",
        "To-Do 4.5 Passed — Zero-Population Guard Defeated ZeroDivisionError!",
        "has_valid_pop=False, is_high_incidence=False, incidence_per_100k=None (zero crashes!).",
        tip="Short-circuit evaluation stops evaluating 'A and B' the instant A is False—making A a built-in safety shield for B.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_todo_05",
        chapter="ch04",
        title="To-Do 4.5: Short-Circuit Zero-Population Guard",
        book_section="Section 4.3.4 (Short-Circuit Guard Clauses)",
        points=10,
        hints={
            1: "Put population > 0 on the LEFT side of 'and' so Python never executes cases / population when population is 0.",
            2: "Algorithm:\n1. has_valid_pop = population > 0\n2. is_high_incidence = population > 0 and (cases / population) * 100000 >= 50.0\n3. if population > 0: incidence_per_100k = round((cases / population) * 100000, 2)\n   else: incidence_per_100k = None",
            3: "has_valid_pop = population > 0\nis_high_incidence = population > 0 and (cases / population) * 100000 >= 50.0\nif population > 0:\n    incidence_per_100k = round((cases / population) * 100000, 2)\nelse:\n    incidence_per_100k = ___",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 4.6: Missing Sentinel (None / 999) Guard & Clinic Triage
# ---------------------------------------------------------------------------
def _verify_todo_06(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 3:
        return CheckResult(
            "ch04_todo_06",
            False,
            "failed",
            "To-Do 4.6 — Pass All Three Triage Outputs",
            "Pass (hb_missing, anemia_status, triage_priority) into tutor.check('ch04_todo_06', ...).",
        )
    hb_missing, anemia_status, triage_priority = val
    if hb_missing is None or anemia_status is None or triage_priority is None:
        return CheckResult(
            "ch04_todo_06",
            False,
            "unattempted",
            "To-Do 4.6 — Ready for Your Solution",
            "Check for None or 999 first, then assign anemia_status and triage_priority.",
        )

    errors = []
    if type(hb_missing) is not bool or hb_missing is not True:
        errors.append(f"hb_missing: expected bool True (raw_hb == 999 is a missing sentinel!), got {hb_missing!r}")
    if anemia_status != "Missing":
        errors.append(
            f"anemia_status: expected 'Missing' (do NOT treat 999 g/dL as 'Normal'!), got {anemia_status!r}"
        )
    if triage_priority != "Moderate":
        errors.append(
            f"triage_priority: expected 'Moderate' (sbp_mmhg = 148 >= 140 and hb_missing is True), got {triage_priority!r}"
        )

    if errors:
        return CheckResult(
            "ch04_todo_06",
            False,
            "failed",
            "To-Do 4.6 — Check Missing-Sentinel Guard & Triage Priority",
            "Survey missing codes like 999 must be intercepted BEFORE any numeric comparison!",
            details=errors,
        )
    return CheckResult(
        "ch04_todo_06",
        True,
        "passed",
        "To-Do 4.6 Passed — Sentinel 999 Intercepted & Triage Routed Safely!",
        "hb_missing=True, anemia_status='Missing', triage_priority='Moderate'.",
        tip="Real DHS/MICS codebooks routinely encode missing lab readings as 99, 999, or -9. Always guard sentinels first!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_todo_06",
        chapter="ch04",
        title="To-Do 4.6: Missing Sentinel (None / 999) Guard & Triage",
        book_section="Section 4.3.4 & Section 4.4 (Guard Clauses + Triage)",
        points=10,
        hints={
            1: "First test hb_missing = (raw_hb is None) or (raw_hb == 999). Only compare raw_hb < 11.0 when hb_missing is False!",
            2: "Algorithm:\n1. hb_missing = raw_hb is None or raw_hb == 999\n2. if hb_missing -> 'Missing'; elif raw_hb < 11.0 -> 'Anemic'; else -> 'Normal'\n3. triage_priority: 'Urgent' if sbp_mmhg >= 160 or (not hb_missing and raw_hb < 7.0); elif sbp_mmhg >= 140 or hb_missing -> 'Moderate'; else -> 'Routine'",
            3: "hb_missing = (raw_hb is None) or (raw_hb == 999)\nif hb_missing:\n    anemia_status = 'Missing'\nelif raw_hb < 11.0:\n    anemia_status = 'Anemic'\nelse:\n    anemia_status = 'Normal'\nif sbp_mmhg >= 160 or (not hb_missing and raw_hb < 7.0):\n    triage_priority = 'Urgent'\nelif sbp_mmhg >= 140 or hb_missing:\n    triage_priority = 'Moderate'\nelse:\n    triage_priority = 'Routine'",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 4.1 (30 pts): WHO BMI & Maternal Anemia Dual Classifier
# ---------------------------------------------------------------------------
def _verify_exam_01(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch04_exam_01",
            False,
            "failed",
            "Exam Task 4.1 — Expected Dictionary Submission",
            "Pass {'bmi': ..., 'bmi_category': ..., 'anemia_severity': ..., 'high_risk_nutrition': ...}.",
        )
    req = ["bmi", "bmi_category", "anemia_severity", "high_risk_nutrition"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch04_exam_01",
            False,
            "unattempted",
            "Exam Task 4.1 — Incomplete Submission",
            "Compute bmi, bmi_category, anemia_severity, and high_risk_nutrition.",
            points_possible=30,
        )

    expected_bmi = round(46.0 / (1.60 ** 2), 2)  # 17.97
    errors = []
    if not _is_number(val["bmi"]) or not math.isclose(val["bmi"], expected_bmi, abs_tol=1e-6):
        errors.append(f"bmi: expected {expected_bmi} (round(46.0 / (1.60 ** 2), 2)), got {val['bmi']!r}")
    if val["bmi_category"] != "Underweight":
        errors.append(f"bmi_category: expected 'Underweight' (since 17.97 < 18.5), got {val['bmi_category']!r}")
    if val["anemia_severity"] != "Severe":
        errors.append(f"anemia_severity: expected 'Severe' (since hb_g_dl = 6.7 < 7.0), got {val['anemia_severity']!r}")
    if type(val["high_risk_nutrition"]) is not bool or val["high_risk_nutrition"] is not True:
        errors.append(f"high_risk_nutrition: expected bool True (Underweight and Severe/Moderate anemia), got {val['high_risk_nutrition']!r}")

    if errors:
        return CheckResult(
            "ch04_exam_01",
            False,
            "failed",
            "Exam Task 4.1 — Check BMI & Anemia Classification Rules",
            "Verify your BMI formula, mutually exclusive elif branches, and high-risk nutrition flag.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch04_exam_01",
        True,
        "passed",
        "Exam Task 4.1 Passed (30/30 pts) — Dual ANC Risk Classifier Verified!",
        "bmi=17.97 ('Underweight'), anemia_severity='Severe', high_risk_nutrition=True.",
        points_earned=30,
        points_possible=30,
        tip="Combining guarded clinical classifiers with boolean risk flags is the core of automated ANC decision support.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_exam_01",
        chapter="ch04",
        title="Exam 4.1: WHO BMI & Maternal Anemia Dual Classifier",
        book_section="Chapter 4 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Compute bmi = round(weight_kg / (height_m ** 2), 2), classify bmi_category and anemia_severity with elif ladders (guarding None/999 first), then set high_risk_nutrition.",
            2: "Algorithm:\n1. bmi = round(46.0 / (1.60 ** 2), 2) -> 17.97\n2. bmi_category: <18.5 Underweight, <25.0 Normal, <30.0 Overweight, else Obese\n3. anemia_severity: None/999 Missing, <7.0 Severe, <10.0 Moderate, <11.0 Mild, else Normal\n4. high_risk_nutrition = (bmi_category == 'Underweight') and (anemia_severity == 'Severe' or anemia_severity == 'Moderate')",
            3: "bmi = round(weight_kg / (height_m ** 2), 2)\n# classify bmi_category and anemia_severity with if/elif/else\nhigh_risk_nutrition = (bmi_category == 'Underweight') and (anemia_severity in ('Severe', 'Moderate'))",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 4.2 (35 pts): Upazila Clinic Emergency Dengue Triage Tree
# ---------------------------------------------------------------------------
def _verify_exam_02(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch04_exam_02",
            False,
            "failed",
            "Exam Task 4.2 — Expected Dictionary Submission",
            "Pass {'critical_thrombocytopenia': ..., 'pediatric_high_fever': ..., 'triage_priority': ..., 'admit_inpatient': ...}.",
        )
    req = [
        "critical_thrombocytopenia",
        "pediatric_high_fever",
        "triage_priority",
        "admit_inpatient",
    ]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch04_exam_02",
            False,
            "unattempted",
            "Exam Task 4.2 — Incomplete Submission",
            "Compute critical_thrombocytopenia, pediatric_high_fever, triage_priority, and admit_inpatient.",
            points_possible=35,
        )

    errors = []
    if type(val["critical_thrombocytopenia"]) is not bool or val["critical_thrombocytopenia"] is not True:
        errors.append(f"critical_thrombocytopenia: expected bool True (platelets_k = 42 < 50), got {val['critical_thrombocytopenia']!r}")
    if type(val["pediatric_high_fever"]) is not bool or val["pediatric_high_fever"] is not True:
        errors.append(f"pediatric_high_fever: expected bool True (age 9 < 12 and temp 39.4 >= 39.0), got {val['pediatric_high_fever']!r}")
    if val["triage_priority"] != "Urgent":
        errors.append(f"triage_priority: expected 'Urgent' (platelets_k < 50 triggers Urgent!), got {val['triage_priority']!r}")
    if type(val["admit_inpatient"]) is not bool or val["admit_inpatient"] is not True:
        errors.append(f"admit_inpatient: expected bool True (Urgent cases are always admitted), got {val['admit_inpatient']!r}")

    if errors:
        return CheckResult(
            "ch04_exam_02",
            False,
            "failed",
            "Exam Task 4.2 — Check Dengue Triage Decision Rules",
            "Check operator precedence (use parentheses around compound and/or clauses) and Urgent-first branch ordering.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch04_exam_02",
        True,
        "passed",
        "Exam Task 4.2 Passed (35/35 pts) — Upazila Dengue Triage Verified!",
        "Platelets 42k (< 50k) & pediatric high fever -> triage_priority='Urgent', admit_inpatient=True.",
        points_earned=35,
        points_possible=35,
        tip="Always place life-threatening 'Urgent' red-flag rules at the very top of a clinical triage tree!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_exam_02",
        chapter="ch04",
        title="Exam 4.2: Upazila Clinic Emergency Dengue Triage",
        book_section="Chapter 4 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Evaluate critical_thrombocytopenia (platelets_k < 50) and pediatric_high_fever (age_years < 12 and temp_c >= 39.0), then route Urgent -> Moderate -> Routine.",
            2: "Algorithm:\n1. critical_thrombocytopenia = platelets_k < 50\n2. pediatric_high_fever = (age_years < 12) and (temp_c >= 39.0)\n3. if has_bleeding or critical_thrombocytopenia or temp_c >= 40.0 -> 'Urgent'\n   elif (50 <= platelets_k < 100) or (temp_c >= 38.5) -> 'Moderate'\n   else -> 'Routine'\n4. admit_inpatient = (triage_priority == 'Urgent') or (triage_priority == 'Moderate' and age_years < 12)",
            3: "critical_thrombocytopenia = platelets_k < 50\npediatric_high_fever = (age_years < 12) and (temp_c >= 39.0)\nif has_bleeding or critical_thrombocytopenia or temp_c >= 40.0:\n    triage_priority = 'Urgent'\nelif (50 <= platelets_k < 100) or (temp_c >= 38.5):\n    triage_priority = 'Moderate'\nelse:\n    triage_priority = 'Routine'\nadmit_inpatient = (triage_priority == 'Urgent') or (triage_priority == 'Moderate' and age_years < 12)",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 4.3 (35 pts): Safe Outbreak Surveillance Guard Pipeline
# ---------------------------------------------------------------------------
def _verify_exam_03(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch04_exam_03",
            False,
            "failed",
            "Exam Task 4.3 — Expected Dictionary Submission",
            "Pass {'rate_a': ..., 'alert_a': ..., 'rate_b': ..., 'alert_b': ...}.",
        )
    req = ["rate_a", "alert_a", "rate_b", "alert_b"]
    if any(k not in val for k in req) or val.get("rate_a") is None or val.get("alert_a") is None or val.get("alert_b") is None:
        return CheckResult(
            "ch04_exam_03",
            False,
            "unattempted",
            "Exam Task 4.3 — Incomplete Submission",
            "Compute rate_a, alert_a, rate_b (should be None for corrupted Ward B), and alert_b.",
            points_possible=35,
        )

    errors = []
    if not _is_number(val["rate_a"]) or not math.isclose(val["rate_a"], 70.0, abs_tol=1e-6):
        errors.append(f"rate_a: expected 70.0 (round((28 / 40000) * 100000, 2)), got {val['rate_a']!r}")
    if val["alert_a"] != "Red":
        errors.append(f"alert_a: expected 'Red' (since rate_a = 70.0 >= 50.0), got {val['alert_a']!r}")
    if val["rate_b"] is not None:
        errors.append(f"rate_b: expected None for Ward B (pop_b == 0 and cases_b == 999), got {val['rate_b']!r}")
    if val["alert_b"] != "Data Error":
        errors.append(f"alert_b: expected 'Data Error' when ward inputs are invalid/missing, got {val['alert_b']!r}")

    if errors:
        return CheckResult(
            "ch04_exam_03",
            False,
            "failed",
            "Exam Task 4.3 — Check Surveillance Guard Clauses",
            "Ensure BOTH pop > 0 AND valid cases (not None, != 999, >= 0) are checked before computing rates.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch04_exam_03",
        True,
        "passed",
        "Exam Task 4.3 Passed (35/35 pts) — Bulletproof Surveillance Guards Verified!",
        "Ward A: 70.0/100k ('Red'); Ward B safely intercepted -> rate_b=None, alert_b='Data Error'.",
        points_earned=35,
        points_possible=35,
        tip="Defensive guard clauses keep automated district dashboards running even when one clinic submits corrupted data!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch04_exam_03",
        chapter="ch04",
        title="Exam 4.3: Safe Outbreak Surveillance Guard Pipeline",
        book_section="Chapter 4 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "For each ward, first validate: (pop > 0) and (cases is not None) and (cases != 999) and (cases >= 0). Only divide inside the valid branch!",
            2: "Algorithm:\n1. If valid: rate = round((cases / pop) * 100000, 2); alert is 'Red' (>=50), 'Amber' (>=20), else 'Green'.\n2. Else: rate = None; alert = 'Data Error'.\n3. Apply the same guarded logic to Ward A (28, 40000) and Ward B (999, 0).",
            3: "if pop_b > 0 and cases_b is not None and cases_b != 999 and cases_b >= 0:\n    rate_b = round((cases_b / pop_b) * 100000, 2)\n    alert_b = 'Red' if rate_b >= 50.0 else ('Amber' if rate_b >= 20.0 else 'Green')\nelse:\n    rate_b = None\n    alert_b = 'Data Error'",
        },
        verifier=_verify_exam_03,
    )
)
