"""Chapter 5 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch05_todo_01..06) and Coding Exam (ch05_exam_01..03).
"""

from __future__ import annotations

import math
from typing import Any

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 5.1: 7-Day Vaccine Cold-Chain Tally (Accumulator + Counter)
# ---------------------------------------------------------------------------
def _verify_todo_01(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch05_todo_01",
            False,
            "failed",
            "To-Do 5.1 — Pass Both Loop Results",
            "Pass (total_doses, high_volume_days) into tutor.check('ch05_todo_01', ...).",
        )
    total_doses, high_volume_days = val
    if total_doses is None or high_volume_days is None:
        return CheckResult(
            "ch05_todo_01",
            False,
            "unattempted",
            "To-Do 5.1 — Ready for Your Solution",
            "Replace the None placeholders using a for loop over daily_doses.",
        )

    errors = []
    if type(total_doses) is not int or total_doses != 350:
        errors.append(
            f"total_doses: expected int 350 (45+52+48+60+55+40+50), "
            f"got {type(total_doses).__name__} ({total_doses!r})"
        )
    if type(high_volume_days) is not int or high_volume_days != 4:
        errors.append(
            f"high_volume_days: expected int 4 (days with doses >= 50: 52, 60, 55, 50), "
            f"got {type(high_volume_days).__name__} ({high_volume_days!r}). "
            "Check whether you used >= 50 instead of > 50!"
        )

    if errors:
        return CheckResult(
            "ch05_todo_01",
            False,
            "failed",
            "To-Do 5.1 — Check Your Accumulator & Counter",
            "Initialize both total_doses = 0 and high_volume_days = 0 BEFORE the for loop.",
            details=errors,
        )
    return CheckResult(
        "ch05_todo_01",
        True,
        "passed",
        "To-Do 5.1 Passed — Accumulator & Counter Patterns Verified!",
        "total_doses=350 across 7 days, with high_volume_days=4 (>= 50 doses).",
        tip="Combining an accumulator (+= doses) and conditional counter (+= 1) in one pass is a staple epidemiology idiom.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_todo_01",
        chapter="ch05",
        title="To-Do 5.1: 7-Day Vaccine Cold-Chain Tally",
        book_section="Section 5.3.2 (The 4 Universal Loop Patterns)",
        points=10,
        hints={
            1: "Initialize total_doses = 0 and high_volume_days = 0 before the loop, then iterate 'for doses in daily_doses:'.",
            2: "Algorithm:\n1. total_doses = 0; high_volume_days = 0\n2. for doses in daily_doses:\n     total_doses += doses\n     if doses >= 50:\n         high_volume_days += 1",
            3: "total_doses = 0\nhigh_volume_days = 0\nfor doses in daily_doses:\n    total_doses += ___\n    if doses >= ___:\n        high_volume_days += 1",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 5.2: Filtering Febrile Patients (Filter Pattern)
# ---------------------------------------------------------------------------
def _verify_todo_02(val: Any) -> CheckResult:
    if val is None:
        return CheckResult(
            "ch05_todo_02",
            False,
            "unattempted",
            "To-Do 5.2 — Ready for Your Solution",
            "Build the list febrile_temps using a for loop and .append().",
        )
    if not isinstance(val, list):
        return CheckResult(
            "ch05_todo_02",
            False,
            "failed",
            "To-Do 5.2 — Expected a Python List",
            f"Expected a list of febrile temperatures, got {type(val).__name__} ({val!r}).",
        )
    expected = [38.4, 39.1, 38.0]
    if val != expected:
        return CheckResult(
            "ch05_todo_02",
            False,
            "failed",
            "To-Do 5.2 — Check Your Fever Cutoff (>= 38.0)",
            f"Expected {expected}, got {val!r}.",
            details=[
                "WHO clinical fever cutoff includes 38.0 C itself (use >= 38.0, not > 38.0).",
                "Make sure you append each qualifying temperature to an initially empty list [].",
            ],
        )
    return CheckResult(
        "ch05_todo_02",
        True,
        "passed",
        "To-Do 5.2 Passed — Filter Pattern Mastered!",
        "febrile_temps = [38.4, 39.1, 38.0] captured cleanly in original order.",
        tip="Start every Filter loop with an empty list `results = []` before the loop header, then `.append()` matching records.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_todo_02",
        chapter="ch05",
        title="To-Do 5.2: Filtering Febrile Patients",
        book_section="Section 5.3.2 (The 4 Universal Loop Patterns — Filter)",
        points=10,
        hints={
            1: "Start with febrile_temps = [] before the loop, then append each temp that satisfies temp >= 38.0.",
            2: "Algorithm:\n1. febrile_temps = []\n2. for temp in temps_c:\n     if temp >= 38.0:\n         febrile_temps.append(temp)",
            3: "febrile_temps = []\nfor temp in temps_c:\n    if temp >= ___:\n        febrile_temps.append(___)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 5.3: Skipping DHS Missing Codes with continue
# ---------------------------------------------------------------------------
def _verify_todo_03(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch05_todo_03",
            False,
            "failed",
            "To-Do 5.3 — Pass Both Cleaned Metrics",
            "Pass (valid_count, mean_age) into tutor.check('ch05_todo_03', (valid_count, mean_age)).",
        )
    valid_count, mean_age = val
    if valid_count is None or mean_age is None:
        return CheckResult(
            "ch05_todo_03",
            False,
            "unattempted",
            "To-Do 5.3 — Ready for Your Solution",
            "Skip None and 999 with continue, then compute valid_count and mean_age.",
        )

    errors = []
    if type(valid_count) is not int or valid_count != 4:
        errors.append(
            f"valid_count: expected int 4 (valid ages: 24, 31, 28, 37), "
            f"got {type(valid_count).__name__} ({valid_count!r})"
        )
    if not _is_number(mean_age) or not math.isclose(mean_age, 30.0, abs_tol=1e-6):
        errors.append(
            f"mean_age: expected 30.0 (round(120 / 4, 1)), got {mean_age!r}. "
            "Make sure you divided by valid_count (4), NOT len(raw_ages) (7), and skipped 999!"
        )

    if errors:
        return CheckResult(
            "ch05_todo_03",
            False,
            "failed",
            "To-Do 5.3 — Check Missing-Code Filtering",
            "Use `if age is None or age == 999: continue` before updating your sum and count.",
            details=errors,
        )
    return CheckResult(
        "ch05_todo_03",
        True,
        "passed",
        "To-Do 5.3 Passed — DHS Missing Codes Skipped Cleanly!",
        "valid_count=4, mean_age=30.0 years (999 and None excluded from both numerator and denominator).",
        tip="Always guard missing codes BEFORE adding to your accumulator—including 999 would inflate the mean to 445+ years!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_todo_03",
        chapter="ch05",
        title="To-Do 5.3: Skipping DHS Missing Codes with continue",
        book_section="Section 5.3.3 (continue & break Flow Control)",
        points=10,
        hints={
            1: "Inside the loop, check `if age is None or age == 999: continue` so missing sentinels are never summed or counted.",
            2: "Algorithm:\n1. age_sum = 0; valid_count = 0\n2. for age in raw_ages:\n     if age is None or age == 999:\n         continue\n     age_sum += age\n     valid_count += 1\n3. mean_age = round(age_sum / valid_count, 1)",
            3: "age_sum = 0\nvalid_count = 0\nfor age in raw_ages:\n    if age is None or age == ___:\n        continue\n    age_sum += age\n    valid_count += 1\nmean_age = round(age_sum / ___, 1)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 5.4: First Outbreak Threshold Breach with break & enumerate()
# ---------------------------------------------------------------------------
def _verify_todo_04(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch05_todo_04",
            False,
            "failed",
            "To-Do 5.4 — Pass Both Breach Values",
            "Pass (breach_day, breach_cases) into tutor.check('ch05_todo_04', ...).",
        )
    breach_day, breach_cases = val
    if breach_day is None or breach_cases is None:
        return CheckResult(
            "ch05_todo_04",
            False,
            "unattempted",
            "To-Do 5.4 — Ready for Your Solution",
            "Use enumerate(daily_cases, start=1) and break on the first day >= 50.",
        )

    errors = []
    if type(breach_day) is not int or breach_day != 4:
        extra = ""
        if breach_day == 3:
            extra = " (Looks 0-indexed! Pass start=1 to enumerate().)"
        elif breach_day == 6:
            extra = " (Did you forget `break`? Without break, later days overwrite Day 4!)"
        errors.append(f"breach_day: expected int 4, got {breach_day!r}{extra}")
    if type(breach_cases) is not int or breach_cases != 52:
        errors.append(
            f"breach_cases: expected int 52 (Day 4 cases), got {breach_cases!r}"
        )

    if errors:
        return CheckResult(
            "ch05_todo_04",
            False,
            "failed",
            "To-Do 5.4 — Check Early Exit with break",
            "Stop the loop immediately with `break` on the FIRST day where cases >= 50.",
            details=errors,
        )
    return CheckResult(
        "ch05_todo_04",
        True,
        "passed",
        "To-Do 5.4 Passed — First Outbreak Breach Caught with break!",
        "Outbreak threshold (>= 50) first breached on Day 4 with 52 cases.",
        tip="Without `break`, a loop keeps scanning and overwrites your first alert with the last alert (Day 6: 58 cases).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_todo_04",
        chapter="ch05",
        title="To-Do 5.4: First Outbreak Threshold Breach (break)",
        book_section="Section 5.3.3 & Section 5.3.4 (break & enumerate)",
        points=10,
        hints={
            1: "Loop with `for day, cases in enumerate(daily_cases, start=1):` and exit with `break` as soon as `cases >= alert_threshold`.",
            2: "Algorithm:\n1. breach_day = None; breach_cases = None\n2. for day, cases in enumerate(daily_cases, start=1):\n     if cases >= alert_threshold:\n         breach_day = day\n         breach_cases = cases\n         break",
            3: "for day, cases in enumerate(daily_cases, start=1):\n    if cases >= alert_threshold:\n        breach_day = ___\n        breach_cases = ___\n        ___",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 5.5: Epidemic Doubling Simulation (while Loop)
# ---------------------------------------------------------------------------
def _verify_todo_05(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch05_todo_05",
            False,
            "failed",
            "To-Do 5.5 — Pass Both Simulation Outputs",
            "Pass (days_elapsed, cases) into tutor.check('ch05_todo_05', (days_elapsed, cases)).",
        )
    days_elapsed, cases = val
    if days_elapsed is None or cases is None:
        return CheckResult(
            "ch05_todo_05",
            False,
            "unattempted",
            "To-Do 5.5 — Ready for Your Solution",
            "Write a while loop that doubles cases every 2 days while cases <= bed_limit.",
        )

    errors = []
    if type(days_elapsed) is not int or days_elapsed != 8:
        errors.append(
            f"days_elapsed: expected int 8 (Day 0:15 -> Day 2:30 -> Day 4:60 -> Day 6:120 -> Day 8:240), "
            f"got {days_elapsed!r}"
        )
    if type(cases) is not int or cases != 240:
        errors.append(
            f"cases: expected int 240 (first doubling exceeding 200 beds), got {cases!r}"
        )

    if errors:
        return CheckResult(
            "ch05_todo_05",
            False,
            "failed",
            "To-Do 5.5 — Trace Your while Loop Step by Step",
            "Double cases (*= 2) and add 2 to days_elapsed (+= 2) while cases <= 200.",
            details=errors,
        )
    return CheckResult(
        "ch05_todo_05",
        True,
        "passed",
        "To-Do 5.5 Passed — Epidemic Doubling while Loop Verified!",
        "15 -> 30 (Day 2) -> 60 (Day 4) -> 120 (Day 6) -> 240 (Day 8 > 200 bed limit).",
        tip="Whenever you write a while loop, always verify two things: (1) the stopping condition and (2) that the state moves toward it every iteration.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_todo_05",
        chapter="ch05",
        title="To-Do 5.5: Epidemic Doubling Simulation (while Loop)",
        book_section="Section 5.3.5 (while Loops & Capacity Simulation)",
        points=10,
        hints={
            1: "Use `while cases <= bed_limit:` and inside the body update both `cases *= 2` and `days_elapsed += 2`.",
            2: "Trace table:\nStart: cases=15, days=0\nIter 1: cases=30, days=2\nIter 2: cases=60, days=4\nIter 3: cases=120, days=6\nIter 4: cases=240, days=8 (240 > 200 -> stop!)",
            3: "while cases <= bed_limit:\n    cases *= ___\n    days_elapsed += ___",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 5.6: Multi-District Nested Loop Summary
# ---------------------------------------------------------------------------
def _verify_todo_06(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch05_todo_06",
            False,
            "failed",
            "To-Do 5.6 — Pass Both District Summary Outputs",
            "Pass (district_totals, top_district) into tutor.check('ch05_todo_06', ...).",
        )
    district_totals, top_district = val
    if district_totals is None or top_district is None:
        return CheckResult(
            "ch05_todo_06",
            False,
            "unattempted",
            "To-Do 5.6 — Ready for Your Solution",
            "Use nested loops to compute district_totals and identify top_district (1-based).",
        )

    errors = []
    if district_totals != [45, 66, 40]:
        errors.append(
            f"district_totals: expected [45, 66, 40], got {district_totals!r}. "
            "Make sure you reset `d_total = 0` inside the outer loop for each district!"
        )
    if type(top_district) is not int or top_district != 2:
        extra = " (Remember: districts are 1-based, so District 2 has 66 cases!)" if top_district == 1 else ""
        errors.append(f"top_district: expected int 2, got {top_district!r}{extra}")

    if errors:
        return CheckResult(
            "ch05_todo_06",
            False,
            "failed",
            "To-Do 5.6 — Check Nested Loop Totals & Top District",
            "Sum each inner clinic list separately and track which 1-based district has the highest sum.",
            details=errors,
        )
    return CheckResult(
        "ch05_todo_06",
        True,
        "passed",
        "To-Do 5.6 Passed — Nested Loop Multi-District Summary Verified!",
        "district_totals=[45, 66, 40], top_district=2 (66 cases).",
        tip="Notice where `d_total = 0` lives: INSIDE the outer loop (so each district starts fresh) and OUTSIDE the inner loop!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_todo_06",
        chapter="ch05",
        title="To-Do 5.6: Multi-District Clinic Totals (Nested Loops)",
        book_section="Section 5.3.6 (Nested Loops)",
        points=10,
        hints={
            1: "In the outer loop over `enumerate(district_clinics, start=1)`, reset `d_total = 0`, sum all clinic counts in an inner loop, append `d_total`, and track the running max district.",
            2: "Algorithm:\n1. district_totals = []; top_district = 1; best_total = -1\n2. for d_num, clinics in enumerate(district_clinics, start=1):\n     d_total = 0\n     for count in clinics:\n         d_total += count\n     district_totals.append(d_total)\n     if d_total > best_total:\n         best_total = d_total; top_district = d_num",
            3: "district_totals = []\ntop_district = 1\nbest_total = -1\nfor d_num, clinics in enumerate(district_clinics, start=1):\n    d_total = 0\n    for count in clinics:\n        d_total += ___\n    district_totals.append(d_total)\n    if d_total > best_total:\n        best_total = d_total\n        top_district = ___",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 5.1 (30 pts): Dirty Hemoglobin Mean Skipping 999 / None
# ---------------------------------------------------------------------------
def _verify_exam_01(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch05_exam_01",
            False,
            "failed",
            "Exam Task 5.1 — Expected Dictionary Submission",
            "Pass {'valid_count': ..., 'missing_count': ..., 'mean_hb': ..., 'anemic_count': ...}.",
        )
    req = ["valid_count", "missing_count", "mean_hb", "anemic_count"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch05_exam_01",
            False,
            "unattempted",
            "Exam Task 5.1 — Incomplete Submission",
            "Compute valid_count, missing_count, mean_hb, and anemic_count.",
            points_possible=30,
        )

    errors = []
    if type(val["valid_count"]) is not int or val["valid_count"] != 6:
        errors.append(
            f"valid_count: expected int 6 (10.5, 11.8, 9.2, 12.4, 10.1, 11.0), got {val['valid_count']!r}"
        )
    if type(val["missing_count"]) is not int or val["missing_count"] != 4:
        errors.append(
            f"missing_count: expected int 4 (two 999s and two Nones), got {val['missing_count']!r}"
        )
    if not _is_number(val["mean_hb"]) or not math.isclose(val["mean_hb"], 10.83, abs_tol=1e-4):
        errors.append(
            f"mean_hb: expected 10.83 (round(65.0 / 6, 2)), got {val['mean_hb']!r}"
        )
    if type(val["anemic_count"]) is not int or val["anemic_count"] != 3:
        extra = " (WHO anemia cutoff is strictly < 11.0, so 11.0 is NOT anemic!)" if val["anemic_count"] == 4 else ""
        errors.append(
            f"anemic_count: expected int 3 (10.5, 9.2, 10.1), got {val['anemic_count']!r}{extra}"
        )

    if errors:
        return CheckResult(
            "ch05_exam_01",
            False,
            "failed",
            "Exam Task 5.1 — Check Missing-Value Filtering & Anemia Cutoff",
            "Skip both None and 999 with continue, then compute the cleaned mean and anemic count.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch05_exam_01",
        True,
        "passed",
        "Exam Task 5.1 Passed (30/30 pts) — Dirty Hemoglobin Dataset Cleaned!",
        "valid_count=6, missing_count=4, mean_hb=10.83 g/dL, anemic_count=3.",
        points_earned=30,
        points_possible=30,
        tip="Always guard `if valid_count > 0:` before dividing so all-missing cohorts never crash with ZeroDivisionError!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_exam_01",
        chapter="ch05",
        title="Exam 5.1: Dirty Maternal Hemoglobin Cleaning & Summary",
        book_section="Chapter 5 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Loop over raw_hb_readings. If hb is None or hb == 999, increment missing_count and continue; otherwise update hb_sum, valid_count, and anemic_count (< 11.0).",
            2: "Algorithm:\n1. Init valid_count=0, missing_count=0, hb_sum=0.0, anemic_count=0\n2. For hb in raw_hb_readings:\n     if hb is None or hb == 999: missing_count += 1; continue\n     hb_sum += hb; valid_count += 1\n     if hb < 11.0: anemic_count += 1\n3. mean_hb = round(hb_sum / valid_count, 2) if valid_count > 0 else None",
            3: "for hb in raw_hb_readings:\n    if hb is None or hb == 999:\n        missing_count += 1\n        continue\n    hb_sum += ___\n    valid_count += 1\n    if hb < 11.0:\n        anemic_count += 1",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 5.2 (35 pts): Monsoon Dengue Peak-Week & High-Alert Surveillance
# ---------------------------------------------------------------------------
def _verify_exam_02(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch05_exam_02",
            False,
            "failed",
            "Exam Task 5.2 — Expected Dictionary Submission",
            "Pass {'total_cases': ..., 'peak_week': ..., 'peak_cases': ..., 'high_alert_weeks': ..., 'first_alert_week': ...}.",
        )
    req = [
        "total_cases",
        "peak_week",
        "peak_cases",
        "high_alert_weeks",
        "first_alert_week",
    ]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch05_exam_02",
            False,
            "unattempted",
            "Exam Task 5.2 — Incomplete Submission",
            "Compute total_cases, peak_week, peak_cases, high_alert_weeks, and first_alert_week.",
            points_possible=35,
        )

    errors = []
    if type(val["total_cases"]) is not int or val["total_cases"] != 868:
        errors.append(f"total_cases: expected int 868, got {val['total_cases']!r}")
    if type(val["peak_week"]) is not int or val["peak_week"] != 5:
        extra = " (Check start=1 in enumerate()!)" if val["peak_week"] == 4 else ""
        errors.append(f"peak_week: expected int 5 (Week 5 had 178 cases), got {val['peak_week']!r}{extra}")
    if type(val["peak_cases"]) is not int or val["peak_cases"] != 178:
        errors.append(f"peak_cases: expected int 178, got {val['peak_cases']!r}")
    if type(val["high_alert_weeks"]) is not int or val["high_alert_weeks"] != 4:
        errors.append(
            f"high_alert_weeks: expected int 4 (Weeks 4, 5, 6, 7 >= 100), got {val['high_alert_weeks']!r}"
        )
    if type(val["first_alert_week"]) is not int or val["first_alert_week"] != 4:
        extra = (
            " (Looks like later alert weeks overwrote the first one—check `if first_alert_week is None:` or use a break loop!)"
            if val["first_alert_week"] == 7
            else ""
        )
        errors.append(
            f"first_alert_week: expected int 4 (Week 4: 134 cases), got {val['first_alert_week']!r}{extra}"
        )

    if errors:
        return CheckResult(
            "ch05_exam_02",
            False,
            "failed",
            "Exam Task 5.2 — Check Monsoon Dengue Surveillance Metrics",
            "Verify your running sum, 1-based peak week, high-alert count (>= 100), and first alert week.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch05_exam_02",
        True,
        "passed",
        "Exam Task 5.2 Passed (35/35 pts) — Monsoon Dengue Surveillance Verified!",
        "868 total cases | Peak: Week 5 (178 cases) | 4 high-alert weeks starting Week 4.",
        points_earned=35,
        points_possible=35,
        tip="Using `enumerate(weekly_dengue, start=1)` eliminates off-by-one week numbering bugs in epidemiology time series.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_exam_02",
        chapter="ch05",
        title="Exam 5.2: Monsoon Dengue Peak-Week & High-Alert Surveillance",
        book_section="Chapter 5 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Loop with `enumerate(weekly_dengue, start=1)` to update total_cases, running peak_cases/peak_week, high_alert_weeks (>= 100), and first_alert_week (only when still None).",
            2: "Algorithm:\n1. total_cases=0, peak_week=1, peak_cases=weekly_dengue[0], high_alert_weeks=0, first_alert_week=None\n2. for week, cases in enumerate(weekly_dengue, start=1):\n     total_cases += cases\n     if cases > peak_cases: peak_cases = cases; peak_week = week\n     if cases >= 100:\n         high_alert_weeks += 1\n         if first_alert_week is None: first_alert_week = week",
            3: "for week, cases in enumerate(weekly_dengue, start=1):\n    total_cases += cases\n    if cases > peak_cases:\n        peak_cases = cases\n        peak_week = ___\n    if cases >= high_alert_threshold:\n        high_alert_weeks += 1\n        if first_alert_week is None:\n            first_alert_week = ___",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 5.3 (35 pts): while-Loop Hospital Surge-Capacity Simulation
# ---------------------------------------------------------------------------
def _verify_exam_03(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch05_exam_03",
            False,
            "failed",
            "Exam Task 5.3 — Expected Dictionary Submission",
            "Pass {'surge_day': ..., 'final_occupied_beds': ..., 'overflow_beds': ..., 'total_admitted': ...}.",
        )
    req = ["surge_day", "final_occupied_beds", "overflow_beds", "total_admitted"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch05_exam_03",
            False,
            "unattempted",
            "Exam Task 5.3 — Incomplete Submission",
            "Simulate daily admissions/discharges in a while loop until occupied_beds > 250.",
            points_possible=35,
        )

    errors = []
    if type(val["surge_day"]) is not int or val["surge_day"] != 8:
        errors.append(
            f"surge_day: expected int 8 (occupied beds reach 260 on Day 8), got {val['surge_day']!r}"
        )
    if type(val["final_occupied_beds"]) is not int or val["final_occupied_beds"] != 260:
        errors.append(
            f"final_occupied_beds: expected int 260 (40->50->65->85->110->140->175->215->260), "
            f"got {val['final_occupied_beds']!r}"
        )
    if type(val["overflow_beds"]) is not int or val["overflow_beds"] != 10:
        errors.append(
            f"overflow_beds: expected int 10 (260 - 250), got {val['overflow_beds']!r}"
        )
    if type(val["total_admitted"]) is not int or val["total_admitted"] != 300:
        errors.append(
            f"total_admitted: expected int 300 (20+25+30+35+40+45+50+55), "
            f"got {val['total_admitted']!r}. Check that daily_admissions += 5 happens AFTER adding to total_admitted!"
        )

    if errors:
        return CheckResult(
            "ch05_exam_03",
            False,
            "failed",
            "Exam Task 5.3 — Trace the Surge-Capacity while Loop",
            "Check the exact order of updates inside `while occupied_beds <= bed_capacity:`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch05_exam_03",
        True,
        "passed",
        "Exam Task 5.3 Passed (35/35 pts) — Hospital Surge Capacity Simulated!",
        "Capacity (250 beds) exceeded on Day 8 with 260 occupied beds (overflow=10, total_admitted=300).",
        points_earned=35,
        points_possible=35,
        tip="In discrete-time simulations, statement order inside the while body matters: add today's admissions BEFORE incrementing tomorrow's admission rate!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch05_exam_03",
        chapter="ch05",
        title="Exam 5.3: Hospital Surge-Capacity Simulation (while Loop)",
        book_section="Chapter 5 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Loop `while occupied_beds <= bed_capacity:`. On each day: increment day, update occupied_beds += daily_admissions - daily_discharges, add daily_admissions to total_admitted, THEN increase daily_admissions += 5.",
            2: "Trace first 3 days:\nDay 1: occ=40+20-10=50, total_adm=20, next_adm=25\nDay 2: occ=50+25-10=65, total_adm=45, next_adm=30\nDay 3: occ=65+30-10=85, total_adm=75, next_adm=35\n...continues until Day 8 (occ=260 > 250).",
            3: "while occupied_beds <= bed_capacity:\n    surge_day += 1\n    occupied_beds += daily_admissions - daily_discharges\n    total_admitted += ___\n    daily_admissions += ___\noverflow_beds = occupied_beds - bed_capacity",
        },
        verifier=_verify_exam_03,
    )
)
