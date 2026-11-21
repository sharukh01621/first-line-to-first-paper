"""Chapter 3 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch03_todo_01..06) and Coding Exam (ch03_exam_01..03).
"""

from __future__ import annotations

import math
from typing import Any

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 3.1: Cleaning Messy Clinic District Names
# ---------------------------------------------------------------------------
def _verify_todo_01(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch03_todo_01",
            False,
            "failed",
            "To-Do 3.1 — Pass Both Cleaned District and Character Count",
            "Pass (clean_district, char_count) into tutor.check('ch03_todo_01', (clean_district, char_count)).",
        )
    clean_district, char_count = val
    if clean_district is None or char_count is None:
        return CheckResult(
            "ch03_todo_01",
            False,
            "unattempted",
            "To-Do 3.1 — Ready for Your Solution",
            "Replace both None placeholders by cleaning raw_district and measuring len(clean_district).",
        )

    errors = []
    if not isinstance(clean_district, str) or clean_district != "Sylhet Sadar":
        hint_extra = ""
        if isinstance(clean_district, str) and clean_district.startswith(" "):
            hint_extra = " (Did you forget .strip() or forget to assign the result?)"
        elif isinstance(clean_district, str) and "_" in clean_district:
            hint_extra = " (Remember to replace '_' with a space ' '.)"
        errors.append(
            f"clean_district: expected 'Sylhet Sadar', got {clean_district!r}{hint_extra}"
        )
    if type(char_count) is not int or char_count != 12:
        errors.append(
            f"char_count: expected int 12 (len('Sylhet Sadar')), got {type(char_count).__name__} ({char_count!r})"
        )

    if errors:
        return CheckResult(
            "ch03_todo_01",
            False,
            "failed",
            "To-Do 3.1 — Check Your String Cleaning Chain",
            "Remember: strings are immutable, so chain .strip().replace('_', ' ').title() and assign it to clean_district.",
            details=errors,
        )
    return CheckResult(
        "ch03_todo_01",
        True,
        "passed",
        "To-Do 3.1 Passed — District Name Cleaned & Measured!",
        "clean_district='Sylhet Sadar' (12 characters).",
        tip="Chaining .strip().replace('_', ' ').title() standardizes messy facility names in one readable line.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_todo_01",
        chapter="ch03",
        title="To-Do 3.1: Cleaning Messy Clinic District Names",
        book_section="Section 3.3.2 (String Immutability & Method Chaining)",
        points=10,
        hints={
            1: "Chain .strip() to remove outer spaces, .replace('_', ' ') to swap the underscore, and .title() to capitalize each word.",
            2: "Algorithm:\n1. clean_district = raw_district.strip().replace('_', ' ').title()\n2. char_count = len(clean_district)",
            3: "clean_district = raw_district.strip().replace(___, ___).title()\nchar_count = len(___)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 3.2: Indexing & Slicing Structured Patient IDs
# ---------------------------------------------------------------------------
def _verify_todo_02(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 4:
        return CheckResult(
            "ch03_todo_02",
            False,
            "failed",
            "To-Do 3.2 — Pass All Four Extracted Fields",
            "Pass (country_code, district_code, cohort_year, serial_str) into tutor.check('ch03_todo_02', ...).",
        )
    country_code, district_code, cohort_year, serial_str = val
    if any(x is None for x in val):
        return CheckResult(
            "ch03_todo_02",
            False,
            "unattempted",
            "To-Do 3.2 — Ready for Your Solution",
            "Extract all four components from patient_id = 'BD-DHK-2026-0482'.",
        )

    errors = []
    if country_code != "BD":
        errors.append(f"country_code: expected 'BD' (patient_id[:2]), got {country_code!r}")
    if district_code != "DHK":
        errors.append(f"district_code: expected 'DHK' (patient_id[3:6]), got {district_code!r}")
    if type(cohort_year) is not int or cohort_year != 2026:
        errors.append(
            f"cohort_year: expected int 2026 (int(patient_id[7:11])), got {type(cohort_year).__name__} ({cohort_year!r})"
        )
    if type(serial_str) is not str or serial_str != "0482":
        errors.append(
            f"serial_str: expected str '0482' (patient_id[-4:]), got {type(serial_str).__name__} ({serial_str!r})"
        )

    if errors:
        return CheckResult(
            "ch03_todo_02",
            False,
            "failed",
            "To-Do 3.2 — Check Your Slice Indices & Types",
            "Remember the half-open slice rule [start:stop]: start is included, stop is excluded!",
            details=errors,
        )
    return CheckResult(
        "ch03_todo_02",
        True,
        "passed",
        "To-Do 3.2 Passed — Structured Patient ID Slicing Mastered!",
        "country_code='BD', district_code='DHK', cohort_year=2026 (int), serial_str='0482' (str).",
        tip="Notice why serial_str stays a str: casting '0482' to int drops the leading zero!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_todo_02",
        chapter="ch03",
        title="To-Do 3.2: Slicing Structured Patient IDs",
        book_section="Section 3.3.1 (Indexing & Half-Open Slicing)",
        points=10,
        hints={
            1: "In 'BD-DHK-2026-0482', indices 0..1 are 'BD', 3..5 are 'DHK', 7..10 are '2026', and the last 4 characters are [-4:].",
            2: "Algorithm:\n1. country_code = patient_id[:2]\n2. district_code = patient_id[3:6]\n3. cohort_year = int(patient_id[7:11])\n4. serial_str = patient_id[-4:]",
            3: "country_code = patient_id[:2]\ndistrict_code = patient_id[___:___]\ncohort_year = int(patient_id[7:11])\nserial_str = patient_id[___:]",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 3.3: Parsing Delimited Clinical Lab Records
# ---------------------------------------------------------------------------
def _verify_todo_03(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 3:
        return CheckResult(
            "ch03_todo_03",
            False,
            "failed",
            "To-Do 3.3 — Pass All Three Parsed Lab Fields",
            "Pass (lab_id, temp_c, is_dengue_pos) into tutor.check('ch03_todo_03', ...).",
        )
    lab_id, temp_c, is_dengue_pos = val
    if any(x is None for x in val):
        return CheckResult(
            "ch03_todo_03",
            False,
            "unattempted",
            "To-Do 3.3 — Ready for Your Solution",
            "Split raw_record on '|' and parse lab_id, temp_c, and is_dengue_pos.",
        )

    errors = []
    if lab_id != "PAT_109":
        errors.append(f"lab_id: expected 'PAT_109', got {lab_id!r}")
    if type(temp_c) is not float or not math.isclose(temp_c, 38.4, abs_tol=1e-6):
        errors.append(
            f"temp_c: expected float 38.4, got {type(temp_c).__name__} ({temp_c!r})"
        )
    if type(is_dengue_pos) is not bool or is_dengue_pos is not True:
        errors.append(
            f"is_dengue_pos: expected bool True, got {type(is_dengue_pos).__name__} ({is_dengue_pos!r})"
        )

    if errors:
        return CheckResult(
            "ch03_todo_03",
            False,
            "failed",
            "To-Do 3.3 — Check Delimited Record Parsing",
            "Strip whitespace from each split field, remove 'C' before float(), and check for 'pos' as a bool.",
            details=errors,
        )
    return CheckResult(
        "ch03_todo_03",
        True,
        "passed",
        "To-Do 3.3 Passed — Pipe-Delimited Lab Record Parsed Cleanly!",
        "lab_id='PAT_109', temp_c=38.4 (float), is_dengue_pos=True (bool).",
        tip="Hospital HL7 and laboratory text exports frequently use '|' delimiters to avoid colliding with commas in clinical notes.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_todo_03",
        chapter="ch03",
        title="To-Do 3.3: Parsing Pipe-Delimited Lab Records",
        book_section="Section 3.3.3 (Splitting & Joining Records)",
        points=10,
        hints={
            1: "Use raw_record.split('|') to break the string into 3 parts, then .strip() each part and remove 'C' before calling float(...).",
            2: "Algorithm:\n1. parts = raw_record.split('|')\n2. lab_id = parts[0].strip()\n3. temp_c = float(parts[1].replace('C', '').strip())\n4. is_dengue_pos = 'pos' in parts[2].strip().lower()",
            3: "parts = raw_record.split('|')\nlab_id = parts[0].strip()\ntemp_c = float(parts[1].replace(___, '').strip())\nis_dengue_pos = 'pos' in parts[2].lower()",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 3.4: Overcoming String Immutability via Rebuild
# ---------------------------------------------------------------------------
def _verify_todo_04(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 2:
        return CheckResult(
            "ch03_todo_04",
            False,
            "failed",
            "To-Do 3.4 — Pass Both Rebuilt Facility Codes",
            "Pass (fixed_code, replaced_facility) into tutor.check('ch03_todo_04', ...).",
        )
    fixed_code, replaced_facility = val
    if fixed_code is None or replaced_facility is None:
        return CheckResult(
            "ch03_todo_04",
            False,
            "unattempted",
            "To-Do 3.4 — Ready for Your Solution",
            "Create fixed_code ('UHC-0042') and replaced_facility ('DHK-0042') without mutating in place.",
        )

    errors = []
    if fixed_code != "UHC-0042":
        errors.append(f"fixed_code: expected 'UHC-0042', got {fixed_code!r}")
    if replaced_facility != "DHK-0042":
        errors.append(
            f"replaced_facility: expected 'DHK-0042' ('DHK' + fixed_code[3:]), got {replaced_facility!r}"
        )

    if errors:
        return CheckResult(
            "ch03_todo_04",
            False,
            "failed",
            "To-Do 3.4 — Check String Rebuild Logic",
            "Because strings cannot be edited in place, construct brand-new strings with .upper(), .replace(), and slicing.",
            details=errors,
        )
    return CheckResult(
        "ch03_todo_04",
        True,
        "passed",
        "To-Do 3.4 Passed — String Immutability Conquered!",
        "fixed_code='UHC-0042' and replaced_facility='DHK-0042'.",
        tip="Whenever you feel tempted to write s[0] = 'X', slice and concatenate a brand-new string instead!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_todo_04",
        chapter="ch03",
        title="To-Do 3.4: Rebuilding Immutable Facility Codes",
        book_section="Section 3.3.2 & Section 3.3.6 (Immutability & TypeError)",
        points=10,
        hints={
            1: "Convert raw_code with .upper().replace('_', '-') for fixed_code, then build replaced_facility using 'DHK' + fixed_code[3:].",
            2: "Algorithm:\n1. fixed_code = raw_code.upper().replace('_', '-')\n2. replaced_facility = 'DHK' + fixed_code[3:]",
            3: "fixed_code = raw_code.upper().replace('_', ___)\nreplaced_facility = 'DHK' + fixed_code[___:]",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 3.5: Publication-Grade f-String Formatting
# ---------------------------------------------------------------------------
def _verify_todo_05(val: Any) -> CheckResult:
    expected = "Rajshahi: n=12,450 | Pos=386 (3.10%) | Mean Hb=11.2 g/dL"
    if val is None:
        return CheckResult(
            "ch03_todo_05",
            False,
            "unattempted",
            "To-Do 3.5 — Ready for Your Solution",
            "Format report_line using an f-string with :, (thousands), :.2% (percent), and :.1f (1 decimal).",
        )
    if not isinstance(val, str):
        return CheckResult(
            "ch03_todo_05",
            False,
            "failed",
            "To-Do 3.5 — Expected a Formatted String",
            f"Expected str, got {type(val).__name__} ({val!r}).",
        )
    if val != expected:
        details = [f"Expected: {expected!r}", f"Got:      {val!r}"]
        if "12450" in val:
            details.append("Hint: Use {screened:,} to insert the thousands comma in 12,450.")
        if "0.03" in val or "3.1%" in val:
            details.append("Hint: Use {prevalence:.2%} so 0.031004... formats as 3.10%.")
        if "11.25" in val or "11.246" in val:
            details.append("Hint: Use {mean_hb:.1f} so 11.246 formats to 1 decimal place (11.2).")
        return CheckResult(
            "ch03_todo_05",
            False,
            "failed",
            "To-Do 3.5 — Check Exact f-String Specifiers & Spacing",
            "Compare your formatted string character-for-character against the expected format.",
            details=details,
        )
    return CheckResult(
        "ch03_todo_05",
        True,
        "passed",
        "To-Do 3.5 Passed — Publication-Ready f-String Verified!",
        expected,
        tip="Notice how :.2% automatically multiplies a proportion by 100 and appends the '%' sign — zero manual * 100 needed!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_todo_05",
        chapter="ch03",
        title="To-Do 3.5: Publication-Grade f-String Summary",
        book_section="Section 3.3.4 (Publication-Grade f-Strings)",
        points=10,
        hints={
            1: "Use {screened:,} for comma-separated counts, {prevalence:.2%} for two-decimal percentage, and {mean_hb:.1f} for 1-decimal hemoglobin.",
            2: "Template:\nreport_line = f\"{district}: n={screened:,} | Pos={positive} ({prevalence:.2%}) | Mean Hb={mean_hb:.1f} g/dL\"",
            3: "report_line = f\"{district}: n={screened:___} | Pos={positive} ({prevalence:___}) | Mean Hb={mean_hb:___} g/dL\"",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 3.6: Sanitizing Simulated User Input & Zero-Padded Case Tags
# ---------------------------------------------------------------------------
def _verify_todo_06(val: Any) -> CheckResult:
    if not isinstance(val, (tuple, list)) or len(val) != 4:
        return CheckResult(
            "ch03_todo_06",
            False,
            "failed",
            "To-Do 3.6 — Pass All Four Sanitized Outputs",
            "Pass (clean_name, age_int, temp_c, case_tag) into tutor.check('ch03_todo_06', ...).",
        )
    clean_name, age_int, temp_c, case_tag = val
    if any(x is None for x in val):
        return CheckResult(
            "ch03_todo_06",
            False,
            "unattempted",
            "To-Do 3.6 — Ready for Your Solution",
            "Sanitize raw_name, raw_age, raw_temp, and format case_tag with 4-digit zero padding.",
        )

    errors = []
    if clean_name != "Fatema Akter":
        errors.append(f"clean_name: expected 'Fatema Akter', got {clean_name!r}")
    if type(age_int) is not int or age_int != 28:
        errors.append(f"age_int: expected int 28, got {type(age_int).__name__} ({age_int!r})")
    if type(temp_c) is not float or not math.isclose(temp_c, 38.6, abs_tol=1e-6):
        errors.append(f"temp_c: expected float 38.6, got {type(temp_c).__name__} ({temp_c!r})")
    if case_tag != "CASE-0007":
        errors.append(f"case_tag: expected 'CASE-0007' (using :04d), got {case_tag!r}")

    if errors:
        return CheckResult(
            "ch03_todo_06",
            False,
            "failed",
            "To-Do 3.6 — Check Input Sanitization & Zero-Padding",
            "Always .strip() raw user input before casting, and use :04d to zero-pad sequence integers.",
            details=errors,
        )
    return CheckResult(
        "ch03_todo_06",
        True,
        "passed",
        "To-Do 3.6 Passed — Clinic Intake Input Sanitized!",
        "Fatema Akter | Age: 28 | Temp: 38.6 C | Tag: CASE-0007.",
        tip="Zero-padded IDs like CASE-0007 sort in true chronological order alphabetically (CASE-0007 before CASE-0010)!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_todo_06",
        chapter="ch03",
        title="To-Do 3.6: Sanitizing Clinic Input & Zero-Padded IDs",
        book_section="Section 3.3.5 (Interactive User Input & Validation Hygiene)",
        points=10,
        hints={
            1: "Clean raw_name with .strip().title(), cast raw_age with int(...), strip 'C' from raw_temp before float(...), and format int(raw_seq.strip()) with :04d.",
            2: "Algorithm:\n1. clean_name = raw_name.strip().title()\n2. age_int = int(raw_age.strip())\n3. temp_c = float(raw_temp.replace('C', '').strip())\n4. case_tag = f'CASE-{int(raw_seq.strip()):04d}'",
            3: "clean_name = raw_name.strip().title()\nage_int = int(raw_age.strip())\ntemp_c = float(raw_temp.replace('C', '').strip())\ncase_tag = f'CASE-{int(raw_seq.strip()):___}'",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 3.1 (30 pts): Clinical Specimen Barcode Parser
# ---------------------------------------------------------------------------
def _verify_exam_01(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch03_exam_01",
            False,
            "failed",
            "Exam Task 3.1 — Expected Dictionary Submission",
            "Pass {'clean_barcode': ..., 'division_code': ..., 'sample_year': ..., 'sample_number': ..., 'is_malaria_panel': ...}.",
        )
    req = [
        "clean_barcode",
        "division_code",
        "sample_year",
        "sample_number",
        "is_malaria_panel",
    ]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch03_exam_01",
            False,
            "unattempted",
            "Exam Task 3.1 — Incomplete Submission",
            "Compute clean_barcode, division_code, sample_year, sample_number, and is_malaria_panel.",
            points_possible=30,
        )

    errors = []
    if val["clean_barcode"] != "BD-CTG-2026-0094-MAL":
        errors.append(
            f"clean_barcode: expected 'BD-CTG-2026-0094-MAL', got {val['clean_barcode']!r}"
        )
    if val["division_code"] != "CTG":
        errors.append(f"division_code: expected 'CTG', got {val['division_code']!r}")
    if type(val["sample_year"]) is not int or val["sample_year"] != 2026:
        errors.append(
            f"sample_year: expected int 2026, got {type(val['sample_year']).__name__} ({val['sample_year']!r})"
        )
    if type(val["sample_number"]) is not int or val["sample_number"] != 94:
        errors.append(
            f"sample_number: expected int 94, got {type(val['sample_number']).__name__} ({val['sample_number']!r})"
        )
    if type(val["is_malaria_panel"]) is not bool or val["is_malaria_panel"] is not True:
        errors.append(
            f"is_malaria_panel: expected bool True, got {type(val['is_malaria_panel']).__name__} ({val['is_malaria_panel']!r})"
        )

    if errors:
        return CheckResult(
            "ch03_exam_01",
            False,
            "failed",
            "Exam Task 3.1 — Check Specimen Barcode Extraction",
            "Normalize with .strip().upper() first, then split on '-' (or slice) and cast year/number to int.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch03_exam_01",
        True,
        "passed",
        "Exam Task 3.1 Passed (30/30 pts) — Specimen Barcode Parsed!",
        "BD-CTG-2026-0094-MAL -> Division='CTG', Year=2026, Sample #94, Malaria Panel=True.",
        points_earned=30,
        points_possible=30,
        tip="Using .split('-') after .strip().upper() makes barcode parsers resilient even if field widths change later.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_exam_01",
        chapter="ch03",
        title="Exam 3.1: Clinical Specimen Barcode Parser",
        book_section="Chapter 3 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "First create clean_barcode = raw_barcode.strip().upper(), then split on '-' into 5 tokens and cast the year and sample number to int.",
            2: "Algorithm:\n1. clean_barcode = raw_barcode.strip().upper()\n2. tokens = clean_barcode.split('-')\n3. division_code = tokens[1]\n4. sample_year = int(tokens[2]); sample_number = int(tokens[3])\n5. is_malaria_panel = clean_barcode.endswith('MAL')",
            3: "clean_barcode = raw_barcode.strip().upper()\ntokens = clean_barcode.split('-')\ndivision_code = tokens[1]\nsample_year = int(tokens[___])\nsample_number = int(tokens[___])\nis_malaria_panel = clean_barcode.endswith('MAL')",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 3.2 (35 pts): Dirty Field-Survey Intake Cleaner
# ---------------------------------------------------------------------------
def _verify_exam_02(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch03_exam_02",
            False,
            "failed",
            "Exam Task 3.2 — Expected Dictionary Submission",
            "Pass {'patient_name': ..., 'age_years': ..., 'bmi': ..., 'sbp_mmhg': ..., 'dbp_mmhg': ...}.",
        )
    req = ["patient_name", "age_years", "bmi", "sbp_mmhg", "dbp_mmhg"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch03_exam_02",
            False,
            "unattempted",
            "Exam Task 3.2 — Incomplete Submission",
            "Parse patient_name, age_years, bmi, sbp_mmhg, and dbp_mmhg from raw_payload.",
            points_possible=35,
        )

    expected_bmi = round(72.4 / (1.68 ** 2), 2)  # 25.65
    errors = []
    if val["patient_name"] != "Rahim Uddin":
        errors.append(f"patient_name: expected 'Rahim Uddin', got {val['patient_name']!r}")
    if type(val["age_years"]) is not int or val["age_years"] != 45:
        errors.append(
            f"age_years: expected int 45, got {type(val['age_years']).__name__} ({val['age_years']!r})"
        )
    if not _is_number(val["bmi"]) or not math.isclose(val["bmi"], expected_bmi, abs_tol=1e-4):
        errors.append(f"bmi: expected {expected_bmi} (72.4 / 1.68**2), got {val['bmi']!r}")
    if type(val["sbp_mmhg"]) is not int or val["sbp_mmhg"] != 148:
        errors.append(
            f"sbp_mmhg: expected int 148, got {type(val['sbp_mmhg']).__name__} ({val['sbp_mmhg']!r})"
        )
    if type(val["dbp_mmhg"]) is not int or val["dbp_mmhg"] != 92:
        errors.append(
            f"dbp_mmhg: expected int 92, got {type(val['dbp_mmhg']).__name__} ({val['dbp_mmhg']!r})"
        )

    if errors:
        return CheckResult(
            "ch03_exam_02",
            False,
            "failed",
            "Exam Task 3.2 — Check Field-Survey Payload Cleaning",
            "Split on ';', strip units ('yrs', 'kg', 'cm', 'mmHg') before casting, and split blood pressure on '/'.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch03_exam_02",
        True,
        "passed",
        "Exam Task 3.2 Passed (35/35 pts) — Messy Survey Payload Cleaned!",
        "Rahim Uddin (45 yrs) | BMI=25.65 | BP=148/92 mmHg.",
        points_earned=35,
        points_possible=35,
        tip="Two-level splitting (first on ';', then on '/' for BP) is a classic pattern for parsing messy clinical EHR strings.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_exam_02",
        chapter="ch03",
        title="Exam 3.2: Dirty Field-Survey Intake Cleaner",
        book_section="Chapter 3 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Split raw_payload on ';' into 5 parts. Strip units ('yrs', 'kg', 'cm', 'mmHg') before casting to int/float, and split BP on '/'.",
            2: "Algorithm:\n1. fields = raw_payload.split(';')\n2. patient_name = fields[0].strip().replace('_', ' ').title()\n3. age_years = int(fields[1].replace('yrs', '').strip())\n4. weight_kg = float(fields[2].replace('kg', '').strip()); height_m = float(fields[3].replace('cm', '').strip()) / 100\n5. bp = fields[4].replace('mmHg', '').strip().split('/'); sbp_mmhg, dbp_mmhg = int(bp[0]), int(bp[1])",
            3: "fields = raw_payload.split(';')\npatient_name = fields[0].strip().replace('_', ' ').title()\nage_years = int(fields[1].replace('yrs', '').strip())\nweight_kg = float(fields[2].replace('kg', '').strip())\nheight_m = float(fields[3].replace('cm', '').strip()) / 100\nbmi = round(weight_kg / (height_m ** 2), 2)\nbp = fields[4].replace('mmHg', '').strip().split('/')\nsbp_mmhg = int(bp[0])\ndbp_mmhg = int(bp[1])",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 3.3 (35 pts): Publication-Ready Surveillance Bulletin Generator
# ---------------------------------------------------------------------------
def _verify_exam_03(val: Any) -> CheckResult:
    if not isinstance(val, dict):
        return CheckResult(
            "ch03_exam_03",
            False,
            "failed",
            "Exam Task 3.3 — Expected Dictionary Submission",
            "Pass {'report_id': ..., 'incidence_per_100k': ..., 'cfr_pct_str': ..., 'bulletin_line': ...}.",
        )
    req = ["report_id", "incidence_per_100k", "cfr_pct_str", "bulletin_line"]
    if any(val.get(k) is None for k in req):
        return CheckResult(
            "ch03_exam_03",
            False,
            "unattempted",
            "Exam Task 3.3 — Incomplete Submission",
            "Compute report_id, incidence_per_100k, cfr_pct_str, and bulletin_line.",
            points_possible=35,
        )

    expected_id = "UHC-SYL-03-R0014"
    expected_inc = round((512 / 340500) * 100000, 2)  # 150.37
    expected_cfr = "1.76%"
    expected_line = "[UHC-SYL-03-R0014] Pop=340,500 | Cases=512 (150.4/100k) | CFR=1.76%"

    errors = []
    if val["report_id"] != expected_id:
        errors.append(f"report_id: expected {expected_id!r}, got {val['report_id']!r}")
    if not _is_number(val["incidence_per_100k"]) or not math.isclose(
        val["incidence_per_100k"], expected_inc, abs_tol=1e-2
    ):
        errors.append(
            f"incidence_per_100k: expected {expected_inc} (round((512/340500)*100000, 2)), got {val['incidence_per_100k']!r}"
        )
    if val["cfr_pct_str"] != expected_cfr:
        errors.append(
            f"cfr_pct_str: expected {expected_cfr!r} (f'{{deaths / confirmed_cases:.2%}}'), got {val['cfr_pct_str']!r}"
        )
    if val["bulletin_line"] != expected_line:
        errors.append(f"bulletin_line expected: {expected_line!r}")
        errors.append(f"bulletin_line got:      {val['bulletin_line']!r}")

    if errors:
        return CheckResult(
            "ch03_exam_03",
            False,
            "failed",
            "Exam Task 3.3 — Check Surveillance Bulletin Formatting",
            "Verify zero-padding (:04d), thousands commas (:,), percentage (:.2%), and 1-decimal incidence (:.1f).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch03_exam_03",
        True,
        "passed",
        "Exam Task 3.3 Passed (35/35 pts) — Publication Bulletin Generated!",
        expected_line,
        points_earned=35,
        points_possible=35,
        tip="Reusable f-string templates let you generate consistent, audit-ready surveillance reports across hundreds of districts.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch03_exam_03",
        chapter="ch03",
        title="Exam 3.3: Publication-Ready Surveillance Bulletin",
        book_section="Chapter 3 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Format report_id with :04d, compute incidence_per_100k = round((confirmed_cases / population) * 100000, 2), format cfr_pct_str with :.2%, and assemble bulletin_line.",
            2: "Algorithm:\n1. report_id = f'{facility_code.upper()}-R{report_seq:04d}'\n2. incidence_per_100k = round((confirmed_cases / population) * 100000, 2)\n3. cfr_pct_str = f'{deaths / confirmed_cases:.2%}'\n4. bulletin_line = f'[{report_id}] Pop={population:,} | Cases={confirmed_cases:,} ({incidence_per_100k:.1f}/100k) | CFR={cfr_pct_str}'",
            3: "report_id = f'{facility_code.upper()}-R{report_seq:04d}'\nincidence_per_100k = round((confirmed_cases / population) * 100000, 2)\ncfr_pct_str = f'{deaths / confirmed_cases:___}'\nbulletin_line = f'[{report_id}] Pop={population:,} | Cases={confirmed_cases:,} ({incidence_per_100k:.1f}/100k) | CFR={cfr_pct_str}'",
        },
        verifier=_verify_exam_03,
    )
)
