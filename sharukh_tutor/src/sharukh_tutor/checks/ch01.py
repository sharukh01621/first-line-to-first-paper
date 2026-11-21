"""Chapter 1 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch01_todo_01..06) and Coding Exam (ch01_exam_01..03).
"""

from __future__ import annotations

import math
from typing import Any

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 1.1: Clinic Intake Banner String
# ---------------------------------------------------------------------------
def _verify_todo_01(val: Any) -> CheckResult:
    expected = "Upazila Health Complex - Patient Intake Active"
    if not isinstance(val, str):
        return CheckResult(
            "ch01_todo_01",
            False,
            "failed",
            "To-Do 1.1 — Type Mismatch",
            f"Expected a text string (str), but received type '{type(val).__name__}'.",
            details=["Wrap your text inside matching quotation marks: \"...\" or '...'."],
            tip="In Python, text must always be enclosed in quotes so Python knows it is data, not a command.",
        )
    if val.strip() != expected:
        return CheckResult(
            "ch01_todo_01",
            False,
            "failed",
            "To-Do 1.1 — Text Does Not Match Exactly",
            f"Expected: {expected!r}\nReceived: {val!r}",
            details=["Check capitalization, spaces around the hyphen '-', and spelling."],
            tip="Computers compare characters strictly: 'health' and 'Health' are different strings.",
        )
    return CheckResult(
        "ch01_todo_01",
        True,
        "passed",
        "To-Do 1.1 Passed — First String Verified!",
        f"clinic_banner matches {expected!r} exactly.",
        tip="Clean string literals are the foundation of readable clinical logs and reports.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_todo_01",
        chapter="ch01",
        title="To-Do 1.1: Clinic Intake Banner",
        book_section="Section 1.3.1 (Strings & print)",
        points=10,
        hints={
            1: "Assign the exact banner text inside quotation marks to the variable named clinic_banner.",
            2: "Algorithm:\n1. Start with the variable name clinic_banner\n2. Add the assignment symbol =\n3. Place the exact text inside double quotes \"...\"",
            3: 'clinic_banner = "Upazila Health Complex - ..."  # Fill in the exact remaining words',
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 1.2: Fever Conversion (Celsius -> Fahrenheit)
# ---------------------------------------------------------------------------
def _verify_todo_02(val: Any) -> CheckResult:
    expected = (38.5 * 9 / 5) + 32  # 101.3
    if isinstance(val, str):
        return CheckResult(
            "ch01_todo_02",
            False,
            "failed",
            "To-Do 1.2 — Received Text Instead of a Number",
            f"temp_f is a string ({val!r}), not a numeric calculation.",
            details=["Remove quotation marks around the arithmetic expression so Python computes the number."],
        )
    if not _is_number(val):
        return CheckResult(
            "ch01_todo_02",
            False,
            "failed",
            "To-Do 1.2 — Type Mismatch",
            f"Expected a numeric value (float), got {type(val).__name__}.",
        )
    if math.isclose(val, 38.5 * (9 / 5 + 32), rel_tol=1e-3):
        return CheckResult(
            "ch01_todo_02",
            False,
            "failed",
            "To-Do 1.2 — Operator Precedence Trap!",
            f"Got {val:.2f}, which suggests + 32 was placed inside the parentheses.",
            details=["Multiply 38.5 by (9 / 5) FIRST, and add 32 AFTERWARD."],
        )
    if not math.isclose(val, expected, abs_tol=1e-2):
        return CheckResult(
            "ch01_todo_02",
            False,
            "failed",
            "To-Do 1.2 — Incorrect Fahrenheit Value",
            f"Expected {expected:.1f}°F, but got {val}.",
            details=["Formula: (temp_c * 9 / 5) + 32 where temp_c = 38.5."],
        )
    return CheckResult(
        "ch01_todo_02",
        True,
        "passed",
        "To-Do 1.2 Passed — Fever Conversion Accurate!",
        f"38.5°C correctly converts to {val:.1f}°F (clinically significant fever).",
        tip="Always verify unit conversions against a mental anchor: 37.0°C is 98.6°F, so 38.5°C must be slightly above 101°F.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_todo_02",
        chapter="ch01",
        title="To-Do 1.2: Fever Temperature Conversion",
        book_section="Section 1.3.2 (Arithmetic Expressions)",
        points=10,
        hints={
            1: "Use the standard formula F = (C * 9 / 5) + 32 with C = 38.5.",
            2: "Algorithm:\n1. Start with temp_c = 38.5\n2. Multiply temp_c by 9, divide by 5\n3. Add 32 and store the result in temp_f",
            3: "temp_c = 38.5\ntemp_f = (temp_c * ___ / ___) + ___",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 1.3: Patient BMI Rounded to 2 Decimals
# ---------------------------------------------------------------------------
def _verify_todo_03(val: Any) -> CheckResult:
    expected = round(62.0 / (1.65 ** 2), 2)  # 22.77
    if not _is_number(val):
        return CheckResult(
            "ch01_todo_03",
            False,
            "failed",
            "To-Do 1.3 — Type Mismatch",
            f"Expected a float for patient_bmi, got {type(val).__name__}.",
        )
    if math.isclose(val, 62.0 / 1.65, abs_tol=0.1):
        return CheckResult(
            "ch01_todo_03",
            False,
            "failed",
            "To-Do 1.3 — Forgot to Square Height!",
            f"Received {val:.2f}, which equals weight / height (missing the square).",
            details=["Use height_m ** 2 (or height_m * height_m) in the denominator."],
        )
    if math.isclose(val, 62.0 / (1.65 * 2), abs_tol=0.1):
        return CheckResult(
            "ch01_todo_03",
            False,
            "failed",
            "To-Do 1.3 — Multiplied by 2 Instead of Squaring!",
            f"Received {val:.2f}. In Python, '*' multiplies while '**' raises to a power.",
            details=["Replace height_m * 2 with height_m ** 2."],
        )
    if not math.isclose(val, expected, abs_tol=1e-6):
        if math.isclose(val, 62.0 / (1.65 ** 2), abs_tol=1e-3):
            return CheckResult(
                "ch01_todo_03",
                False,
                "failed",
                "To-Do 1.3 — Almost There! Round to 2 Decimal Places",
                f"Your raw BMI ({val:.6f}) is right, but the prompt asks for 2 decimal places ({expected}).",
                details=["Wrap your final calculation in round(..., 2)."],
            )
        return CheckResult(
            "ch01_todo_03",
            False,
            "failed",
            "To-Do 1.3 — Incorrect BMI Value",
            f"Expected {expected}, but got {val}.",
            details=["Check inputs: weight_kg = 62.0, height_m = 1.65."],
        )
    return CheckResult(
        "ch01_todo_03",
        True,
        "passed",
        "To-Do 1.3 Passed — Patient BMI Computed Correctly!",
        f"patient_bmi = {val} kg/m^2 (normal WHO BMI range: 18.5-24.9).",
        tip="Remember: '**' is exponentiation in Python; '^' is NOT exponentiation (it is bitwise XOR!).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_todo_03",
        chapter="ch01",
        title="To-Do 1.3: Patient BMI Calculation",
        book_section="Section 1.3.2 & Section 1.4 (Problem Blueprint)",
        points=10,
        hints={
            1: "BMI equals weight in kilograms divided by height in meters squared, then rounded to 2 decimals.",
            2: "Algorithm:\n1. weight_kg = 62.0 and height_m = 1.65\n2. Compute raw_bmi = weight_kg / (height_m ** 2)\n3. Round using round(raw_bmi, 2)",
            3: "weight_kg = 62.0\nheight_m = 1.65\npatient_bmi = round(weight_kg / (height_m ** ___), ___)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 1.4: Vaccine Cold-Box Capacity (// and %)
# ---------------------------------------------------------------------------
def _verify_todo_04(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch01_todo_04",
            False,
            "failed",
            "To-Do 1.4 — Pass Both Variables",
            "Pass (full_boxes, leftover_vials) into tutor.check('ch01_todo_04', (full_boxes, leftover_vials)).",
        )
    full_boxes, leftover_vials = val
    if isinstance(full_boxes, float):
        return CheckResult(
            "ch01_todo_04",
            False,
            "failed",
            "To-Do 1.4 — Used Regular Division (/) Instead of Floor Division (//)",
            f"full_boxes is {full_boxes} (a float). You cannot pack a fractional full box!",
            details=["Use // (floor division) to get the whole number of full boxes."],
        )
    if full_boxes != 11 or leftover_vials != 11:
        return CheckResult(
            "ch01_todo_04",
            False,
            "failed",
            "To-Do 1.4 — Check Quotient and Remainder",
            f"Expected (11, 11), got ({full_boxes!r}, {leftover_vials!r}).",
            details=["275 // 24 = 11 full boxes, and 275 % 24 = 11 leftover vials (check: 11 * 24 + 11 = 275)."],
        )
    return CheckResult(
        "ch01_todo_04",
        True,
        "passed",
        "To-Do 1.4 Passed — Cold-Chain Logistics Verified!",
        "11 full cold boxes + 11 leftover vials = 275 total vaccine vials.",
        tip="Verification trick: always check that (full_boxes * 24) + leftover_vials == 275!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_todo_04",
        chapter="ch01",
        title="To-Do 1.4: Vaccine Cold-Box Packing",
        book_section="Section 1.3.2 (Floor Division // and Modulo %)",
        points=10,
        hints={
            1: "Python has two special operators for packing problems: // gives whole groups, and % gives the remainder.",
            2: "Algorithm:\n1. total_vials = 275, box_capacity = 24\n2. full_boxes = total_vials // box_capacity\n3. leftover_vials = total_vials % box_capacity",
            3: "total_vials = 275\nbox_capacity = 24\nfull_boxes = total_vials ___ box_capacity\nleftover_vials = total_vials ___ box_capacity",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 1.5: Crude Dengue Incidence Rate per 100,000
# ---------------------------------------------------------------------------
def _verify_todo_05(val: Any) -> CheckResult:
    expected = round((186 / 240000) * 100000, 2)  # 77.5
    if not _is_number(val):
        return CheckResult(
            "ch01_todo_05",
            False,
            "failed",
            "To-Do 1.5 — Type Mismatch",
            f"Expected a numeric rate, got {type(val).__name__}.",
        )
    if math.isclose(val, 186 / 240000, abs_tol=1e-4):
        return CheckResult(
            "ch01_todo_05",
            False,
            "failed",
            "To-Do 1.5 — Forgot to Scale per 100,000 Population",
            f"Got {val:.6f} (the raw proportion). Multiply by 100000 to express per 100,000 residents.",
        )
    if not math.isclose(val, expected, abs_tol=1e-6):
        return CheckResult(
            "ch01_todo_05",
            False,
            "failed",
            "To-Do 1.5 — Incorrect Incidence Rate",
            f"Expected {expected}, got {val}.",
            details=["Formula: round((186 / 240000) * 100000, 2)."],
        )
    return CheckResult(
        "ch01_todo_05",
        True,
        "passed",
        "To-Do 1.5 Passed — Epidemiological Incidence Rate Verified!",
        f"Weekly dengue incidence = {val} cases per 100,000 population.",
        tip="Standardizing counts per 100,000 lets epidemiologists compare rural and urban districts fairly.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_todo_05",
        chapter="ch01",
        title="To-Do 1.5: Weekly Dengue Incidence Rate",
        book_section="Section 1.3.2 (Scaling Rates & Parentheses)",
        points=10,
        hints={
            1: "Divide confirmed cases (186) by total population (240000), multiply by 100000, and round to 2 decimal places.",
            2: "Algorithm:\n1. cases = 186\n2. population = 240000\n3. incidence_per_100k = round((cases / population) * 100000, 2)",
            3: "cases = 186\npopulation = 240000\nincidence_per_100k = round((___ / ___) * 100000, 2)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 1.6: Debug & Fix Broken Mean Systolic BP Calculation
# ---------------------------------------------------------------------------
def _verify_todo_06(val: Any) -> CheckResult:
    expected = (124 + 130 + 127) / 3  # 127.0
    if not _is_number(val):
        return CheckResult(
            "ch01_todo_06",
            False,
            "failed",
            "To-Do 1.6 — Type Mismatch",
            f"Expected a numeric mean systolic BP, got {type(val).__name__}.",
        )
    if math.isclose(val, 124 + 130 + 127 / 3, abs_tol=0.1):
        return CheckResult(
            "ch01_todo_06",
            False,
            "failed",
            "To-Do 1.6 — Missing Parentheses Around the Sum!",
            f"Got {val:.2f}! Without parentheses, Python only divided 127 by 3 before adding 124 and 130.",
            details=["Wrap the numerator in parentheses: (sbp_1 + sbp_2 + sbp_3) / 3."],
        )
    if not math.isclose(val, expected, abs_tol=1e-6):
        return CheckResult(
            "ch01_todo_06",
            False,
            "failed",
            "To-Do 1.6 — Incorrect Mean Systolic BP",
            f"Expected {expected}, got {val}.",
            details=["Check that all three readings (124, 130, 127) are numbers (not quoted strings) and summed before dividing by 3."],
        )
    return CheckResult(
        "ch01_todo_06",
        True,
        "passed",
        "To-Do 1.6 Passed — All 3 Bugs Squashed!",
        f"Mean systolic blood pressure = {val} mmHg.",
        tip="Missing parentheses around a numerator is a silent logic bug—Python runs without error, so sanity checks are essential!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_todo_06",
        chapter="ch01",
        title="To-Do 1.6: Debug the Broken Blood Pressure Average",
        book_section="Section 1.3.4 (Break It On Purpose — Error Anatomy)",
        points=10,
        hints={
            1: "Look closely for three bugs: a number accidentally wrapped in quotes, a misspelled variable name, and missing parentheses around the sum.",
            2: "Checklist:\n1. sbp_2 should be 130 (integer), not '130' (string)\n2. Spelling of sbp_3 must match everywhere\n3. Use (sbp_1 + sbp_2 + sbp_3) / 3",
            3: "sbp_1 = 124\nsbp_2 = 130\nsbp_3 = 127\nmean_sbp = (sbp_1 + sbp_2 + sbp_3) / ___",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 1.1 (30 pts): ORS Supply & Carton Logistics
# ---------------------------------------------------------------------------
def _verify_exam_01(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch01_exam_01",
            False,
            "failed",
            "Exam Task 1.1 — Expected Result Dictionary",
            "Pass {'base_packets': base_packets, 'total_packets': total_packets, 'full_cartons': full_cartons, 'extra_packets': extra_packets}.",
        )
    required = ["base_packets", "total_packets", "full_cartons", "extra_packets"]
    missing = [k for k in required if k not in val or val[k] is None]
    if missing:
        return CheckResult(
            "ch01_exam_01",
            False,
            "unattempted",
            "Exam Task 1.1 — Incomplete Submission",
            f"Missing or None values for: {', '.join(missing)}.",
        )

    errors = []
    if val["base_packets"] != 420:
        errors.append(f"base_packets: expected 420 (140 patients * 3 packets), got {val['base_packets']!r}")
    if val["total_packets"] == 482:
        errors.append(
            "total_packets: got 482! Classic binary floating-point trap: 420 * 1.15 is 482.99999999999994, "
            "so int(...) chops off the decimal to 482. Use round(base_packets * buffer_multiplier) to get 483!"
        )
    elif not _is_number(val["total_packets"]) or not math.isclose(val["total_packets"], 483.0, abs_tol=1e-6):
        errors.append(f"total_packets: expected 483 (round(420 * 1.15)), got {val['total_packets']!r}")
    if val["full_cartons"] != 9:
        errors.append(f"full_cartons: expected 9 (483 // 50), got {val['full_cartons']!r}")
    if val["extra_packets"] != 33:
        errors.append(f"extra_packets: expected 33 (483 % 50), got {val['extra_packets']!r}")

    if errors:
        return CheckResult(
            "ch01_exam_01",
            False,
            "failed",
            "Exam Task 1.1 — Check ORS Supply Calculations",
            "One or more logistics values did not match the clinic specification.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch01_exam_01",
        True,
        "passed",
        "Exam Task 1.1 Passed (30/30 pts) — ORS Supply Logistics Verified!",
        "420 base packets -> 483 with 15% buffer -> 9 full cartons + 33 extra packets.",
        points_earned=30,
        points_possible=30,
        tip="In real supply-chain code, we would order 10 cartons so the 33 extra packets are covered.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_exam_01",
        chapter="ch01",
        title="Exam 1.1: Pediatric ORS Supply Logistics",
        book_section="Chapter 1 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Multiply 140 patients by 3 packets, add a 15% buffer with round(base_packets * 1.15), then pack into 50-packet cartons using // and %.",
            2: "Algorithm:\n1. base_packets = 140 * 3\n2. total_packets = round(base_packets * 1.15)\n3. full_cartons = total_packets // 50\n4. extra_packets = total_packets % 50",
            3: "base_packets = 140 * 3\ntotal_packets = round(base_packets * 1.15)\nfull_cartons = total_packets // ___\nextra_packets = total_packets % ___",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 1.2 (35 pts): Maternal Health Clinic Summary Card
# ---------------------------------------------------------------------------
def _verify_exam_02(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch01_exam_02",
            False,
            "failed",
            "Exam Task 1.2 — Expected Result Dictionary",
            "Pass {'anc_coverage_pct': ..., 'anemia_prevalence_pct': ..., 'summary_line': ...}.",
        )
    expected_anc = round((218 / 320) * 100, 1)  # 68.1
    expected_anemia = round((96 / 320) * 100, 1)  # 30.0
    expected_line = "ANC 4+ Coverage: 68.1% | Anemia Prevalence: 30.0%"

    errors = []
    if not _is_number(val.get("anc_coverage_pct")) or not math.isclose(val["anc_coverage_pct"], expected_anc, abs_tol=1e-6):
        errors.append(f"anc_coverage_pct: expected {expected_anc}, got {val.get('anc_coverage_pct')!r}")
    if not _is_number(val.get("anemia_prevalence_pct")) or not math.isclose(val["anemia_prevalence_pct"], expected_anemia, abs_tol=1e-6):
        errors.append(f"anemia_prevalence_pct: expected {expected_anemia}, got {val.get('anemia_prevalence_pct')!r}")
    if val.get("summary_line") != expected_line:
        errors.append(f"summary_line:\n  expected: {expected_line!r}\n  got:      {val.get('summary_line')!r}")

    if errors:
        return CheckResult(
            "ch01_exam_02",
            False,
            "failed",
            "Exam Task 1.2 — Check Maternal Health Summary Metrics",
            "One or more metrics or the summary string format did not match.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch01_exam_02",
        True,
        "passed",
        "Exam Task 1.2 Passed (35/35 pts) — Maternal Health Summary Verified!",
        f"Report generated: {expected_line}",
        points_earned=35,
        points_possible=35,
        tip="Using f-strings like f'ANC 4+ Coverage: {anc_coverage_pct}%' keeps reports synced with calculations.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_exam_02",
        chapter="ch01",
        title="Exam 1.2: Maternal Health Clinic Summary Card",
        book_section="Chapter 1 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Compute both percentages using (count / 320) * 100 rounded to 1 decimal place, then build the formatted summary_line.",
            2: "Algorithm:\n1. anc_coverage_pct = round((218 / 320) * 100, 1)\n2. anemia_prevalence_pct = round((96 / 320) * 100, 1)\n3. Combine them with an f-string matching the exact template.",
            3: 'summary_line = f"ANC 4+ Coverage: {anc_coverage_pct}% | Anemia Prevalence: {anemia_prevalence_pct}%"',
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 1.3 (35 pts): Two-District Case Fatality Rate & Risk Ratio
# ---------------------------------------------------------------------------
def _verify_exam_03(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch01_exam_03",
            False,
            "failed",
            "Exam Task 1.3 — Expected Result Dictionary",
            "Pass {'cfr_rural_pct': ..., 'cfr_urban_pct': ..., 'cfr_ratio': ...}.",
        )
    errors = []
    if not _is_number(val.get("cfr_rural_pct")) or not math.isclose(val["cfr_rural_pct"], 2.0, abs_tol=1e-6):
        errors.append(f"cfr_rural_pct: expected 2.0 ((9 / 450) * 100), got {val.get('cfr_rural_pct')!r}")
    if not _is_number(val.get("cfr_urban_pct")) or not math.isclose(val["cfr_urban_pct"], 1.0, abs_tol=1e-6):
        errors.append(f"cfr_urban_pct: expected 1.0 ((6 / 600) * 100), got {val.get('cfr_urban_pct')!r}")
    if not _is_number(val.get("cfr_ratio")) or not math.isclose(val["cfr_ratio"], 2.0, abs_tol=1e-6):
        errors.append(f"cfr_ratio: expected 2.0 (cfr_rural_pct / cfr_urban_pct), got {val.get('cfr_ratio')!r}")

    if errors:
        return CheckResult(
            "ch01_exam_03",
            False,
            "failed",
            "Exam Task 1.3 — Check Epidemiological CFR Calculations",
            "Verify your denominators (450 rural cases, 600 urban cases) and ratio direction (rural / urban).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch01_exam_03",
        True,
        "passed",
        "Exam Task 1.3 Passed (35/35 pts) — Epidemiological Disparity Quantified!",
        "Rural CFR (2.0%) is 2.0x higher than Urban CFR (1.0%) — a key health-equity signal.",
        points_earned=35,
        points_possible=35,
        tip="You just wrote your first epidemiological disparity calculation — the exact metric we revisit with confidence intervals in Chapter 20!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch01_exam_03",
        chapter="ch01",
        title="Exam 1.3: Rural vs. Urban Case Fatality Disparity",
        book_section="Chapter 1 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Case Fatality Rate (%) = (deaths / confirmed_cases) * 100. Compute it separately for rural and urban, then divide rural by urban.",
            2: "Algorithm:\n1. cfr_rural_pct = (9 / 450) * 100\n2. cfr_urban_pct = (6 / 600) * 100\n3. cfr_ratio = cfr_rural_pct / cfr_urban_pct",
            3: "cfr_rural_pct = (9 / 450) * 100\ncfr_urban_pct = (6 / 600) * 100\ncfr_ratio = ___ / ___",
        },
        verifier=_verify_exam_03,
    )
)
