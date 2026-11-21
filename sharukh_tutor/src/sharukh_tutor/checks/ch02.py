"""Chapter 2 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch02_todo_01..06) and Coding Exam (ch02_exam_01..03).
"""

from __future__ import annotations

import math
from typing import Any

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 2.1: Type Casting Raw Survey Strings
# ---------------------------------------------------------------------------
def _verify_todo_01(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 3:
        return CheckResult(
            "ch02_todo_01",
            False,
            "failed",
            "To-Do 2.1 — Pass All Three Converted Variables",
            "Pass (age_int, hb_float, visits_int) into tutor.check('ch02_todo_01', (age_int, hb_float, visits_int)).",
        )
    age_int, hb_float, visits_int = val
    if age_int is None or hb_float is None or visits_int is None:
        return CheckResult(
            "ch02_todo_01",
            False,
            "unattempted",
            "To-Do 2.1 — Ready for Your Solution",
            "One or more variables are still None. Convert all three raw string inputs.",
        )

    errors = []
    if type(age_int) is not int or age_int != 34:
        errors.append(f"age_int: expected int 34, got {type(age_int).__name__} ({age_int!r})")
    if type(hb_float) is not float or not math.isclose(hb_float, 11.8, abs_tol=1e-6):
        errors.append(f"hb_float: expected float 11.8, got {type(hb_float).__name__} ({hb_float!r})")
    if type(visits_int) is not int or visits_int != 4:
        errors.append(f"visits_int: expected int 4, got {type(visits_int).__name__} ({visits_int!r})")

    if errors:
        return CheckResult(
            "ch02_todo_01",
            False,
            "failed",
            "To-Do 2.1 — Check Data Types After Conversion",
            "Remember: CSV files store every field as text (str) until you cast it explicitly.",
            details=errors,
        )
    return CheckResult(
        "ch02_todo_01",
        True,
        "passed",
        "To-Do 2.1 Passed — Raw Survey Strings Cast Cleanly!",
        "age_int=34 (int), hb_float=11.8 (float), visits_int=4 (int).",
        tip="Always verify types with type(...) right after loading raw survey or clinical CSV files.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_todo_01",
        chapter="ch02",
        title="To-Do 2.1: Casting Raw Survey Strings",
        book_section="Section 2.3.4 (Type Conversion)",
        points=10,
        hints={
            1: "Use int(...) for whole numbers (age, visits) and float(...) for decimal measurements (hemoglobin).",
            2: "Algorithm:\n1. age_int = int(raw_age)\n2. hb_float = float(raw_hb)\n3. visits_int = int(raw_visits)",
            3: "age_int = int(___)\nhb_float = float(___)\nvisits_int = int(___)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 2.2: Augmented Assignment Queue Tracker
# ---------------------------------------------------------------------------
def _verify_todo_02(val: Any) -> CheckResult:
    if type(val) is not int:
        return CheckResult(
            "ch02_todo_02",
            False,
            "failed",
            "To-Do 2.2 — Expected Integer Patient Count",
            f"Expected an int for waiting_patients, got {type(val).__name__} ({val!r}).",
        )
    if val != 20:
        return CheckResult(
            "ch02_todo_02",
            False,
            "failed",
            "To-Do 2.2 — Trace the Queue Step by Step",
            f"Expected 20 waiting patients, got {val}.",
            details=["Start at 12 -> += 7 (19) -> -= 4 (15) -> += 5 (20)."],
        )
    return CheckResult(
        "ch02_todo_02",
        True,
        "passed",
        "To-Do 2.2 Passed — Clinic Queue Rebinding Verified!",
        "waiting_patients transitioned cleanly: 12 -> 19 -> 15 -> 20.",
        tip="Augmented assignments (+=, -=) rebind the name to a new integer object cleanly.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_todo_02",
        chapter="ch02",
        title="To-Do 2.2: Clinic Queue Augmented Assignment",
        book_section="Section 2.3.3 (Rebinding & Augmented Assignment)",
        points=10,
        hints={
            1: "Start with waiting_patients = 12, then update it in order using += 7, -= 4, and += 5.",
            2: "Trace table:\nStart: 12\nAfter 7 arrive (+= 7): 19\nAfter 4 discharged (-= 4): 15\nAfter 5 arrive (+= 5): 20",
            3: "waiting_patients = 12\nwaiting_patients += 7\nwaiting_patients -= ___\nwaiting_patients += ___",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 2.3: Clinical Boolean Screening Flags
# ---------------------------------------------------------------------------
def _verify_todo_03(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 3:
        return CheckResult(
            "ch02_todo_03",
            False,
            "failed",
            "To-Do 2.3 — Pass All Three Boolean Flags",
            "Pass (has_fever, is_hypertensive, is_anemic) into tutor.check('ch02_todo_03', ...).",
        )
    has_fever, is_hypertensive, is_anemic = val
    if has_fever is None or is_hypertensive is None or is_anemic is None:
        return CheckResult(
            "ch02_todo_03",
            False,
            "unattempted",
            "To-Do 2.3 — Ready for Your Solution",
            "Replace all three None placeholders with comparison expressions returning True or False.",
        )

    errors = []
    if type(has_fever) is not bool or has_fever is not False:
        errors.append(f"has_fever: expected bool False (since 37.2 < 38.0), got {type(has_fever).__name__} ({has_fever!r})")
    if type(is_hypertensive) is not bool or is_hypertensive is not True:
        errors.append(f"is_hypertensive: expected bool True (since 146 >= 140), got {type(is_hypertensive).__name__} ({is_hypertensive!r})")
    if type(is_anemic) is not bool or is_anemic is not True:
        errors.append(f"is_anemic: expected bool True (since 10.2 < 11.0), got {type(is_anemic).__name__} ({is_anemic!r})")

    if errors:
        return CheckResult(
            "ch02_todo_03",
            False,
            "failed",
            "To-Do 2.3 — Check Boolean Threshold Conditions",
            "Make sure each flag is a real Python bool (True/False), NOT a string like 'True'.",
            details=errors,
        )
    return CheckResult(
        "ch02_todo_03",
        True,
        "passed",
        "To-Do 2.3 Passed — Clinical Screening Booleans Verified!",
        "has_fever=False, is_hypertensive=True, is_anemic=True.",
        tip="Notice how naming booleans with has_/is_ prefixes makes clinical logic read like English.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_todo_03",
        chapter="ch02",
        title="To-Do 2.3: Clinical Boolean Screening Flags",
        book_section="Section 2.3.2 (Primitive Data Types — bool)",
        points=10,
        hints={
            1: "Use comparison operators (>= and <) to produce True/False booleans directly from temp_c, sbp_mmhg, and hb_g_dl.",
            2: "Thresholds:\n- has_fever: temp_c >= 38.0\n- is_hypertensive: sbp_mmhg >= 140\n- is_anemic: hb_g_dl < 11.0",
            3: "has_fever = temp_c >= 38.0\nis_hypertensive = sbp_mmhg >= ___\nis_anemic = hb_g_dl < ___",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 2.4: Two-Step String-Float-to-Int Conversion Trap
# ---------------------------------------------------------------------------
def _verify_todo_04(val: Any) -> CheckResult:
    if isinstance(val, str):
        return CheckResult(
            "ch02_todo_04",
            False,
            "failed",
            "To-Do 2.4 — Still a String!",
            f"months_int is still a str ({val!r}). Convert it first with float(...), then int(...).",
        )
    if type(val) is float:
        return CheckResult(
            "ch02_todo_04",
            False,
            "failed",
            "To-Do 2.4 — Halfway There! Convert Float to Int",
            f"months_int is a float ({val!r}). Wrap float(raw_months) inside int(...) to get integer 24.",
        )
    if type(val) is not int or val != 24:
        return CheckResult(
            "ch02_todo_04",
            False,
            "failed",
            "To-Do 2.4 — Expected Integer 24",
            f"Expected int 24, got {type(val).__name__} ({val!r}).",
        )
    return CheckResult(
        "ch02_todo_04",
        True,
        "passed",
        "To-Do 2.4 Passed — Survey Decimal-String Trap Conquered!",
        "raw_months = '24.0' -> float('24.0') = 24.0 -> int(24.0) = 24.",
        tip="DHS and SPSS exports very frequently write whole ages as '24.0' — int(float(s)) is a classic data-cleaning idiom.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_todo_04",
        chapter="ch02",
        title="To-Do 2.4: Two-Step Decimal-String Conversion",
        book_section="Section 2.3.4 & Section 2.3.6 (ValueError Anatomy)",
        points=10,
        hints={
            1: "Calling int('24.0') directly raises ValueError because of the decimal point. Convert to float first, then int!",
            2: "Algorithm:\n1. First parse the decimal string: months_float = float(raw_months)\n2. Then cast to integer: months_int = int(months_float)",
            3: "months_int = int(float(___))",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 2.5: Floating-Point Precision & math.isclose()
# ---------------------------------------------------------------------------
def _verify_todo_05(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch02_todo_05",
            False,
            "failed",
            "To-Do 2.5 — Pass Both Comparison Booleans",
            "Pass (exact_eq, safe_eq) into tutor.check('ch02_todo_05', (exact_eq, safe_eq)).",
        )
    exact_eq, safe_eq = val
    if exact_eq is None or safe_eq is None:
        return CheckResult(
            "ch02_todo_05",
            False,
            "unattempted",
            "To-Do 2.5 — Ready for Your Solution",
            "Assign exact_eq using == and safe_eq using math.isclose(...).",
        )

    errors = []
    if exact_eq is not False:
        errors.append(f"exact_eq: expected False (because 0.1 + 0.2 == 0.30000000000000004), got {exact_eq!r}")
    if safe_eq is not True:
        errors.append(f"safe_eq: expected True from math.isclose(dose_a + dose_b, target_dose), got {safe_eq!r}")

    if errors:
        return CheckResult(
            "ch02_todo_05",
            False,
            "failed",
            "To-Do 2.5 — Check Floating-Point Comparisons",
            "Compare exact == versus math.isclose(...) on 0.1 + 0.2 vs 0.3.",
            details=errors,
        )
    return CheckResult(
        "ch02_todo_05",
        True,
        "passed",
        "To-Do 2.5 Passed — Binary Float Precision Mastered!",
        "exact_eq is False, while math.isclose(0.1 + 0.2, 0.3) is True!",
        tip="In scientific computing and ML unit tests, ALWAYS compare floats with math.isclose() or np.isclose(), never ==.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_todo_05",
        chapter="ch02",
        title="To-Do 2.5: Safe Float Comparison with math.isclose",
        book_section="Section 2.3.5 (Binary Floating-Point Precision)",
        points=10,
        hints={
            1: "Import math, compare (dose_a + dose_b) == target_dose for exact_eq, and use math.isclose(dose_a + dose_b, target_dose) for safe_eq.",
            2: "Algorithm:\n1. import math\n2. exact_eq = (dose_a + dose_b) == target_dose\n3. safe_eq = math.isclose(dose_a + dose_b, target_dose)",
            3: "import math\nexact_eq = (dose_a + dose_b) == target_dose\nsafe_eq = math.isclose(___, ___)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 2.6: Pediatric Weight-Based Dosage & Type Verification
# ---------------------------------------------------------------------------
def _verify_todo_06(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 3:
        return CheckResult(
            "ch02_todo_06",
            False,
            "failed",
            "To-Do 2.6 — Pass All Three Outputs",
            "Pass (daily_mg, exact_tablets, dose_is_float) into tutor.check('ch02_todo_06', ...).",
        )
    daily_mg, exact_tablets, dose_is_float = val
    if daily_mg is None or exact_tablets is None or dose_is_float is None:
        return CheckResult(
            "ch02_todo_06",
            False,
            "unattempted",
            "To-Do 2.6 — Ready for Your Solution",
            "Complete daily_mg, exact_tablets, and dose_is_float.",
        )

    errors = []
    if not _is_number(daily_mg) or not math.isclose(daily_mg, 832.5, abs_tol=1e-6):
        errors.append(f"daily_mg: expected 832.5 (18.5 * 15 * 3), got {daily_mg!r}")
    if not _is_number(exact_tablets) or not math.isclose(exact_tablets, 3.33, abs_tol=1e-6):
        errors.append(f"exact_tablets: expected 3.33 (round(832.5 / 250, 2)), got {exact_tablets!r}")
    if dose_is_float is not True:
        errors.append(f"dose_is_float: expected bool True (type(daily_mg) is float), got {dose_is_float!r}")

    if errors:
        return CheckResult(
            "ch02_todo_06",
            False,
            "failed",
            "To-Do 2.6 — Check Pediatric Dosage Calculation",
            "Verify daily mg (weight_kg * mg_per_kg * doses_per_day) and tablet conversion.",
            details=errors,
        )
    return CheckResult(
        "ch02_todo_06",
        True,
        "passed",
        "To-Do 2.6 Passed — Pediatric Dosage Verified!",
        "18.5 kg * 15 mg/kg * 3/day = 832.5 mg/day (3.33 tablets of 250 mg).",
        tip="Notice int * float always promotes cleanly to float in Python.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_todo_06",
        chapter="ch02",
        title="To-Do 2.6: Pediatric Weight-Based Dosage",
        book_section="Section 2.4 (Problem-Structuring Blueprint)",
        points=10,
        hints={
            1: "Multiply weight_kg * 15 * 3 for daily_mg, divide by 250 and round to 2 decimals for exact_tablets, and check isinstance(daily_mg, float).",
            2: "Algorithm:\n1. daily_mg = weight_kg * 15 * 3\n2. exact_tablets = round(daily_mg / 250, 2)\n3. dose_is_float = isinstance(daily_mg, float)",
            3: "daily_mg = weight_kg * 15 * 3\nexact_tablets = round(daily_mg / ___, 2)\ndose_is_float = isinstance(daily_mg, ___)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 2.1 (30 pts): DHS Messy Survey Record Normalization
# ---------------------------------------------------------------------------
def _verify_exam_01(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch02_exam_01",
            False,
            "failed",
            "Exam Task 2.1 — Expected Dictionary Submission",
            "Pass {'weight_kg': ..., 'height_m': ..., 'anc_visits': ..., 'bmi': ...}.",
        )
    req = ["weight_kg", "height_m", "anc_visits", "bmi"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch02_exam_01",
            False,
            "unattempted",
            "Exam Task 2.1 — Incomplete Submission",
            "Compute weight_kg, height_m, anc_visits, and bmi.",
            points_possible=30,
        )

    errors = []
    if type(val["weight_kg"]) is not float or not math.isclose(val["weight_kg"], 54.6, abs_tol=1e-6):
        errors.append(f"weight_kg: expected float 54.6, got {type(val['weight_kg']).__name__} ({val['weight_kg']!r})")
    if type(val["height_m"]) is not float or not math.isclose(val["height_m"], 1.56, abs_tol=1e-6):
        errors.append(f"height_m: expected float 1.56 (float('156.0') / 100), got {val['height_m']!r}")
    if type(val["anc_visits"]) is not int or val["anc_visits"] != 4:
        errors.append(f"anc_visits: expected int 4 (int(float('4.0'))), got {type(val['anc_visits']).__name__} ({val['anc_visits']!r})")
    expected_bmi = round(54.6 / (1.56 ** 2), 2)  # 22.44
    if not _is_number(val["bmi"]) or not math.isclose(val["bmi"], expected_bmi, abs_tol=1e-6):
        errors.append(f"bmi: expected {expected_bmi}, got {val['bmi']!r}")

    if errors:
        return CheckResult(
            "ch02_exam_01",
            False,
            "failed",
            "Exam Task 2.1 — Check Survey Parsing & Types",
            "One or more fields had the wrong type or value.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch02_exam_01",
        True,
        "passed",
        "Exam Task 2.1 Passed (30/30 pts) — DHS Record Normalized!",
        "weight_kg=54.6 (float), height_m=1.56 (float), anc_visits=4 (int), bmi=22.44.",
        points_earned=30,
        points_possible=30,
        tip="Clean type normalization at ingestion prevents cascading TypeError bugs downstream.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_exam_01",
        chapter="ch02",
        title="Exam 2.1: DHS Survey Record Normalization",
        book_section="Chapter 2 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Convert raw_weight_kg and raw_height_cm with float(...), divide height by 100, convert raw_anc_visits with int(float(...)), then compute BMI.",
            2: "Algorithm:\n1. weight_kg = float(raw_weight_kg)\n2. height_m = float(raw_height_cm) / 100\n3. anc_visits = int(float(raw_anc_visits))\n4. bmi = round(weight_kg / (height_m ** 2), 2)",
            3: "weight_kg = float(raw_weight_kg)\nheight_m = float(raw_height_cm) / 100\nanc_visits = int(float(___))\nbmi = round(weight_kg / (height_m ** 2), 2)",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 2.2 (35 pts): Multi-Day Outbreak Accumulator & Threshold Alert
# ---------------------------------------------------------------------------
def _verify_exam_02(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch02_exam_02",
            False,
            "failed",
            "Exam Task 2.2 — Expected Dictionary Submission",
            "Pass {'total_cases': ..., 'mean_daily_cases': ..., 'incidence_per_100k': ..., 'outbreak_alert': ...}.",
        )
    req = ["total_cases", "mean_daily_cases", "incidence_per_100k", "outbreak_alert"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch02_exam_02",
            False,
            "unattempted",
            "Exam Task 2.2 — Incomplete Submission",
            "Compute total_cases, mean_daily_cases, incidence_per_100k, and outbreak_alert.",
            points_possible=35,
        )

    errors = []
    if type(val["total_cases"]) is not int or val["total_cases"] != 100:
        errors.append(f"total_cases: expected int 100 (18+27+34+21), got {val['total_cases']!r}")
    if not _is_number(val["mean_daily_cases"]) or not math.isclose(val["mean_daily_cases"], 25.0, abs_tol=1e-6):
        errors.append(f"mean_daily_cases: expected 25.0 (100 / 4), got {val['mean_daily_cases']!r}")
    if not _is_number(val["incidence_per_100k"]) or not math.isclose(val["incidence_per_100k"], 40.0, abs_tol=1e-6):
        errors.append(f"incidence_per_100k: expected 40.0 ((100 / 250000) * 100000), got {val['incidence_per_100k']!r}")
    if type(val["outbreak_alert"]) is not bool or val["outbreak_alert"] is not True:
        errors.append(f"outbreak_alert: expected bool True (since 40.0 >= 35.0), got {val['outbreak_alert']!r}")

    if errors:
        return CheckResult(
            "ch02_exam_02",
            False,
            "failed",
            "Exam Task 2.2 — Check Outbreak Accumulation Metrics",
            "Verify your 4-day sum, mean, rate per 100k, and boolean alert flag.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch02_exam_02",
        True,
        "passed",
        "Exam Task 2.2 Passed (35/35 pts) — Surveillance Alert Triggered!",
        "100 cumulative cases (25.0/day) -> 40.0 per 100,000 -> outbreak_alert=True.",
        points_earned=35,
        points_possible=35,
        tip="You just built the accumulator + threshold pattern that scales to loops in Chapter 5!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_exam_02",
        chapter="ch02",
        title="Exam 2.2: 4-Day Outbreak Accumulator & Alert",
        book_section="Chapter 2 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Start total_cases = 0, add each day's count using +=, divide by 4 for mean_daily_cases, scale by 100000 / 250000, and check >= 35.0.",
            2: "Algorithm:\n1. total_cases = 0; add 18, 27, 34, 21 via +=\n2. mean_daily_cases = total_cases / 4\n3. incidence_per_100k = (total_cases / 250000) * 100000\n4. outbreak_alert = incidence_per_100k >= 35.0",
            3: "total_cases = 0\ntotal_cases += 18\n...\nmean_daily_cases = total_cases / 4\nincidence_per_100k = (total_cases / 250000) * 100000\noutbreak_alert = incidence_per_100k >= 35.0",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 2.3 (35 pts): Biostatistics 2x2 Odds & Odds Ratio
# ---------------------------------------------------------------------------
def _verify_exam_03(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch02_exam_03",
            False,
            "failed",
            "Exam Task 2.3 — Expected Dictionary Submission",
            "Pass {'odds_exposed': ..., 'odds_unexposed': ..., 'odds_ratio': ..., 'elevated_risk': ...}.",
        )
    req = ["odds_exposed", "odds_unexposed", "odds_ratio", "elevated_risk"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch02_exam_03",
            False,
            "unattempted",
            "Exam Task 2.3 — Incomplete Submission",
            "Compute odds_exposed, odds_unexposed, odds_ratio, and elevated_risk.",
            points_possible=35,
        )

    expected_oe = round(45 / 105, 4)    # 0.4286
    expected_ou = round(20 / 130, 4)    # 0.1538
    expected_or = round((45 * 130) / (105 * 20), 2)  # 2.79

    errors = []
    if not _is_number(val["odds_exposed"]) or not math.isclose(val["odds_exposed"], expected_oe, abs_tol=1e-4):
        errors.append(f"odds_exposed: expected {expected_oe} (round(45 / 105, 4)), got {val['odds_exposed']!r}")
    if not _is_number(val["odds_unexposed"]) or not math.isclose(val["odds_unexposed"], expected_ou, abs_tol=1e-4):
        errors.append(f"odds_unexposed: expected {expected_ou} (round(20 / 130, 4)), got {val['odds_unexposed']!r}")
    if not _is_number(val["odds_ratio"]) or not math.isclose(val["odds_ratio"], expected_or, abs_tol=1e-2):
        errors.append(
            f"odds_ratio: expected {expected_or} (round((a * d) / (b * c), 2)), got {val['odds_ratio']!r}. "
            "Check parentheses around (b * c) in the denominator!"
        )
    if type(val["elevated_risk"]) is not bool or val["elevated_risk"] is not True:
        errors.append(f"elevated_risk: expected bool True (odds_ratio > 1.0), got {val['elevated_risk']!r}")

    if errors:
        return CheckResult(
            "ch02_exam_03",
            False,
            "failed",
            "Exam Task 2.3 — Check 2x2 Odds Ratio Math",
            "Verify odds = cases / non-cases (not cases / total!) and OR = (a * d) / (b * c).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch02_exam_03",
        True,
        "passed",
        "Exam Task 2.3 Passed (35/35 pts) — Epidemiological Odds Ratio Verified!",
        "OR = 2.79: Unsupplemented mothers had 2.79x higher odds of anemia (elevated_risk=True).",
        points_earned=35,
        points_possible=35,
        tip="Remember: compute OR from exact unrounded counts (a*d)/(b*c) before rounding, avoiding intermediate rounding drift!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch02_exam_03",
        chapter="ch02",
        title="Exam 2.3: 2x2 Table Odds & Odds Ratio",
        book_section="Chapter 2 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Odds = diseased / non-diseased (a/b and c/d). Compute exact odds_ratio = round((a * d) / (b * c), 2), then check odds_ratio > 1.0.",
            2: "Algorithm:\n1. odds_exposed = round(a / b, 4)\n2. odds_unexposed = round(c / d, 4)\n3. odds_ratio = round((a * d) / (b * c), 2)\n4. elevated_risk = odds_ratio > 1.0",
            3: "odds_exposed = round(a / b, 4)\nodds_unexposed = round(c / d, 4)\nodds_ratio = round((a * d) / (b * c), 2)\nelevated_risk = odds_ratio > ___",
        },
        verifier=_verify_exam_03,
    )
)
