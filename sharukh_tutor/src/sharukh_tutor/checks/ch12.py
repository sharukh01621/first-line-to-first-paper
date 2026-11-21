"""Chapter 12 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch12_todo_01..06) and Coding Exam (ch12_exam_01..03)
for Object-Oriented Programming (OOP) for Data Scientists & Researchers.
Every verifier inspects a submitted class (`isinstance(cls, type)`), tests
instantiation, methods, dunder protocols, edge cases, instance isolation,
and non-mutation, while letting `NotImplementedError` propagate cleanly for
unattempted starter stubs.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise

# Fallback reference to the learner's MaternalPatient class when grading
# ch12_exam_03 (EpidemicRegistry) after ch12_exam_01.
_LAST_MATERNAL_PATIENT_CLS: Any = None


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _check_class(
    ex_id: str,
    title: str,
    cls: Any,
    expected_name: str,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    if not isinstance(cls, type):
        hint_extra = (
            f"It looks like you passed an instance of {type(cls).__name__} "
            f"instead of the class itself."
            if not isinstance(cls, (int, float, str, list, dict, tuple, bool))
            else f"Received {type(cls).__name__} ({cls!r}) instead of a class."
        )
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Pass the Class Name Without Parentheses",
            (
                f"Expected a class (like {expected_name}), "
                f"but received {type(cls).__name__}."
            ),
            details=[
                hint_extra,
                f"Pass the class itself: tutor.check('{ex_id}', {expected_name})",
                f"Do NOT call {expected_name}(...) inside tutor.check().",
            ],
            points_earned=0,
            points_possible=points_possible,
        )
    return None


def _none_return_trap(
    ex_id: str,
    title: str,
    call_str: str,
    points_possible: int = 10,
) -> CheckResult:
    return CheckResult(
        ex_id,
        False,
        "failed",
        f"{title} — Method Returned None (print() vs. return Trap!)",
        f"Calling {call_str} returned None instead of the computed value.",
        details=[
            "Remember: print() only displays output on screen; methods must use `return ...` to hand back values.",
            f"Verify that {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Methods are functions attached to objects — they still need `return` to pass values back to callers.",
    )


def _is_default_repr(text: str) -> bool:
    return " object at 0x" in text


# ---------------------------------------------------------------------------
# GUIDED TO-DO 12.1: PatientRecord (class, __init__, .bmi(), __repr__)
# ---------------------------------------------------------------------------
def _verify_todo_01(cls: Any) -> CheckResult:
    bad = _check_class("ch12_todo_01", "To-Do 12.1", cls, "PatientRecord", 10)
    if bad is not None:
        return bad

    try:
        p1 = cls("P-101", 64.5, 168.0)
        p2 = cls("P-102", 54.6, 156.0)
        p3 = cls("P-103", 70.0, 175.0)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_01",
            False,
            "failed",
            "To-Do 12.1 — Check __init__ Signature",
            f"PatientRecord('P-101', 64.5, 168.0) raised TypeError: {exc}",
            details=[
                "Define: def __init__(self, patient_id, weight_kg, height_cm):",
                "Remember that `self` must always be the first parameter of every instance method.",
            ],
        )
    except Exception as exc:
        return CheckResult(
            "ch12_todo_01",
            False,
            "failed",
            "To-Do 12.1 — Error Instantiating PatientRecord",
            f"Creating PatientRecord('P-101', 64.5, 168.0) raised {type(exc).__name__}: {exc}",
        )

    errors = []
    # Check stored instance attributes & independence between p1 and p2
    for attr, exp1, exp2 in [
        ("patient_id", "P-101", "P-102"),
        ("weight_kg", 64.5, 54.6),
        ("height_cm", 168.0, 156.0),
    ]:
        if not hasattr(p1, attr):
            errors.append(f"Missing instance attribute `self.{attr}` set inside `__init__`.")
        else:
            v1 = getattr(p1, attr)
            v2 = getattr(p2, attr)
            if v1 != exp1 or v2 != exp2:
                errors.append(
                    f"Attribute `{attr}` mismatch: p1.{attr}={v1!r} (expected {exp1!r}), "
                    f"p2.{attr}={v2!r} (expected {exp2!r})."
                )

    if errors:
        return CheckResult(
            "ch12_todo_01",
            False,
            "failed",
            "To-Do 12.1 — Store Attributes on `self` Inside `__init__`",
            "Assign `self.patient_id`, `self.weight_kg`, and `self.height_cm` inside `__init__`.",
            details=errors,
        )

    # Check .bmi() method
    if not hasattr(p1, "bmi") or not callable(p1.bmi):
        return CheckResult(
            "ch12_todo_01",
            False,
            "failed",
            "To-Do 12.1 — Missing `.bmi()` Method",
            "PatientRecord needs a `.bmi(self)` method returning BMI rounded to 2 decimal places.",
        )

    bmi_cases = [
        (p1, 22.85, "PatientRecord('P-101', 64.5, 168.0).bmi()"),
        (p2, 22.44, "PatientRecord('P-102', 54.6, 156.0).bmi()"),
        (p3, 22.86, "PatientRecord('P-103', 70.0, 175.0).bmi()"),
    ]
    for obj, expected, label in bmi_cases:
        try:
            got = obj.bmi()
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch12_todo_01",
                False,
                "failed",
                "To-Do 12.1 — Runtime Error in `.bmi()`",
                f"{label} raised {type(exc).__name__}: {exc}",
                details=["Make sure `.bmi(self)` reads `self.weight_kg` and `self.height_cm`."],
            )
        if got is None:
            return _none_return_trap("ch12_todo_01", "To-Do 12.1", label, 10)
        if _is_number(got) and got < 0.05:
            errors.append(
                f"{label} returned {got!r} — did you forget to convert `self.height_cm` to meters (`self.height_cm / 100`)?"
            )
        elif not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
            errors.append(f"{label}: expected {expected}, got {got!r}")

    # Check __repr__
    try:
        r1 = repr(p1)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"repr(p1) raised {type(exc).__name__}: {exc}")
    else:
        if not isinstance(r1, str) or _is_default_repr(r1):
            errors.append(
                f"Define a custom `__repr__(self)` method — currently repr(p1) returned {r1!r}."
            )
        elif "PatientRecord" not in r1 or "P-101" not in r1:
            errors.append(
                f"`__repr__` should include both 'PatientRecord' and the patient_id ('P-101'), got {r1!r}."
            )

    if errors:
        return CheckResult(
            "ch12_todo_01",
            False,
            "failed",
            "To-Do 12.1 — Check `.bmi()` Formula & `__repr__`",
            "Compute `round(self.weight_kg / ((self.height_cm / 100) ** 2), 2)` and return an informative `__repr__` string.",
            details=errors,
        )

    return CheckResult(
        "ch12_todo_01",
        True,
        "passed",
        "To-Do 12.1 Passed — PatientRecord Class Verified!",
        "Stored independent instance attributes, computed BMI across 3 patients, and implemented a readable `__repr__`.",
        tip="A crisp `__repr__` turns opaque `<PatientRecord object at 0x...>` logs into immediately inspectable clinical records.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_todo_01",
        chapter="ch12",
        title="To-Do 12.1: PatientRecord Class with .bmi() and __repr__",
        book_section="Section 12.3.1 (Classes, __init__, self & __repr__)",
        points=10,
        hints={
            1: "Store `patient_id`, `weight_kg`, and `height_cm` on `self` in `__init__`, convert cm to m in `.bmi()`, and return a string containing `'PatientRecord'` and `self.patient_id` in `__repr__`.",
            2: "Blueprint:\n1. `__init__(self, patient_id, weight_kg, height_cm)` assigns the 3 attributes.\n2. `bmi(self)` computes `height_m = self.height_cm / 100` and returns `round(self.weight_kg / (height_m ** 2), 2)`.\n3. `__repr__(self)` returns `f\"PatientRecord(patient_id={self.patient_id!r}, weight_kg={self.weight_kg}, height_cm={self.height_cm})\"`.",
            3: "class PatientRecord:\n    def __init__(self, patient_id, weight_kg, height_cm):\n        self.patient_id = patient_id\n        self.weight_kg = weight_kg\n        self.height_cm = height_cm\n\n    def bmi(self):\n        height_m = self.height_cm / ___\n        return round(self.weight_kg / (height_m ** 2), 2)\n\n    def __repr__(self):\n        return f'PatientRecord(patient_id={self.patient_id!r}, weight_kg={self.weight_kg}, height_cm={self.height_cm})'",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 12.2: ClinicCohort (composition, __len__, .mean_bmi())
# ---------------------------------------------------------------------------
class _MockPatient:
    """Lightweight duck-typed patient helper used to test ClinicCohort."""

    def __init__(self, patient_id: str, bmi_val: float) -> None:
        self.patient_id = patient_id
        self._bmi = bmi_val

    def bmi(self) -> float:
        return self._bmi


def _verify_todo_02(cls: Any) -> CheckResult:
    bad = _check_class("ch12_todo_02", "To-Do 12.2", cls, "ClinicCohort", 10)
    if bad is not None:
        return bad

    try:
        c1 = cls("Dhaka Urban Clinic")
        c2 = cls("Sylhet Rural Clinic")
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Check `__init__(self, name)` Signature",
            f"ClinicCohort('Dhaka Urban Clinic') raised TypeError: {exc}",
            details=["Define: def __init__(self, name): and initialize an empty patient list on `self`."],
        )
    except Exception as exc:
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Error Instantiating ClinicCohort",
            f"ClinicCohort('Dhaka Urban Clinic') raised {type(exc).__name__}: {exc}",
        )

    errors = []
    if not hasattr(c1, "name") or c1.name != "Dhaka Urban Clinic":
        errors.append("Store `self.name = name` inside `__init__`.")

    # Verify __len__ on empty cohort
    try:
        empty_len = len(c1)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Missing `__len__(self)` Dunder Method",
            f"len(cohort) raised TypeError: {exc}",
            details=["Implement `def __len__(self): return len(self.patients)` so Python's `len(cohort)` works."],
        )
    if empty_len != 0:
        errors.append(f"Empty cohort should have len(cohort) == 0, got {empty_len!r}.")

    # Verify mean_bmi() on empty cohort -> None (no ZeroDivisionError)
    if not hasattr(c1, "mean_bmi") or not callable(c1.mean_bmi):
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Missing `.mean_bmi()` Method",
            "Implement `.mean_bmi(self)` returning the mean patient BMI (rounded to 2 decimals) or None if empty.",
        )

    try:
        empty_mean = c1.mean_bmi()
    except NotImplementedError:
        raise
    except ZeroDivisionError:
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — ZeroDivisionError on Empty Cohort",
            "Calling `.mean_bmi()` on an empty ClinicCohort raised ZeroDivisionError.",
            details=["Add an empty guard: `if not self.patients: return None` before dividing!"],
        )
    except Exception as exc:
        errors.append(f"c1.mean_bmi() on empty cohort raised {type(exc).__name__}: {exc}")
    else:
        if empty_mean is not None:
            errors.append(f"Empty cohort `.mean_bmi()` should return None, got {empty_mean!r}.")

    # Add patients to c1 and verify c2 stays completely isolated
    if not hasattr(c1, "add_patient") or not callable(c1.add_patient):
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Missing `.add_patient(self, patient)` Method",
            "Define `add_patient(self, patient)` to append a patient object to the cohort.",
        )

    samples = [
        _MockPatient("P-101", 22.85),
        _MockPatient("P-102", 22.44),
        _MockPatient("P-103", 18.75),
    ]
    for sample in samples:
        try:
            c1.add_patient(sample)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch12_todo_02",
                False,
                "failed",
                "To-Do 12.2 — Runtime Error in `.add_patient()`",
                f"c1.add_patient(patient) raised {type(exc).__name__}: {exc}",
            )

    if len(c1) != 3:
        errors.append(f"After adding 3 patients, expected len(c1) == 3, got {len(c1)!r}.")

    # Catch the classic mutable class-attribute trap!
    if len(c2) != 0:
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Mutable Class Attribute Leak Detected!",
            (
                f"Adding 3 patients to `c1` also mutated `c2` (len(c2) == {len(c2)} instead of 0)!"
            ),
            details=[
                "Never define `patients = []` directly in the class body — all instances would share the SAME list!",
                "Initialize `self.patients = []` INSIDE `__init__(self, name)` so each cohort gets its own list.",
            ],
            tip="This is the OOP cousin of Chapter 7's aliasing trap: mutable containers always belong inside `__init__`.",
        )

    try:
        mean_val = c1.mean_bmi()
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"c1.mean_bmi() raised {type(exc).__name__}: {exc}")
    else:
        if mean_val is None:
            return _none_return_trap("ch12_todo_02", "To-Do 12.2", "c1.mean_bmi()", 10)
        expected_mean = round((22.85 + 22.44 + 18.75) / 3, 2)  # 21.35
        if not _is_number(mean_val) or not math.isclose(mean_val, expected_mean, abs_tol=1e-2):
            errors.append(
                f"c1.mean_bmi() for BMIs [22.85, 22.44, 18.75]: expected {expected_mean}, got {mean_val!r}."
            )

    if errors:
        return CheckResult(
            "ch12_todo_02",
            False,
            "failed",
            "To-Do 12.2 — Check Cohort Container & Mean BMI",
            "Track patients inside `self.patients = []`, support `len(cohort)`, and return `round(mean, 2)` or `None`.",
            details=errors,
        )

    return CheckResult(
        "ch12_todo_02",
        True,
        "passed",
        "To-Do 12.2 Passed — ClinicCohort Composition & `__len__` Verified!",
        "Verified empty guard (None), `len(cohort)`, 2-decimal `.mean_bmi()` (21.35), and full instance isolation.",
        tip="Composition ('a cohort HAS-A list of patients') is cleaner and safer than deep inheritance hierarchies.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_todo_02",
        chapter="ch12",
        title="To-Do 12.2: ClinicCohort Container (__len__ & .mean_bmi())",
        book_section="Section 12.3.3 (Dunder Protocols & Composition)",
        points=10,
        hints={
            1: "Inside `__init__(self, name)`, set `self.name = name` and `self.patients = []`. Implement `add_patient(self, patient)`, `__len__(self)`, and `mean_bmi(self)` (returning `None` if empty).",
            2: "Blueprint:\n1. `add_patient`: `self.patients.append(patient)`\n2. `__len__`: `return len(self.patients)`\n3. `mean_bmi`: if `not self.patients`: `return None`; compute `sum(p.bmi() for p in self.patients) / len(self.patients)` rounded to 2 decimals.",
            3: "class ClinicCohort:\n    def __init__(self, name):\n        self.name = name\n        self.patients = []\n\n    def add_patient(self, patient):\n        self.patients.append(patient)\n\n    def __len__(self):\n        return len(self.patients)\n\n    def mean_bmi(self):\n        if not self.patients:\n            return ___\n        total = sum(p.bmi() for p in self.patients)\n        return round(total / len(self.patients), 2)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 12.3: @dataclass LabResult
# ---------------------------------------------------------------------------
def _verify_todo_03(cls: Any) -> CheckResult:
    bad = _check_class("ch12_todo_03", "To-Do 12.3", cls, "LabResult", 10)
    if bad is not None:
        return bad

    # Instantiate first so starter stub's NotImplementedError propagates cleanly
    try:
        r_anemic = cls("P-101", 10.4)
        r_border = cls("P-102", 11.0)
        r_normal = cls("P-103", 12.5)
        r_same = cls(patient_id="P-101", hb_g_dl=10.4)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_03",
            False,
            "failed",
            "To-Do 12.3 — Check @dataclass Fields (`patient_id`, `hb_g_dl`)",
            f"Instantiating LabResult raised TypeError: {exc}",
            details=[
                "Decorate the class with `@dataclass` and declare typed fields in order:",
                "    patient_id: str",
                "    hb_g_dl: float",
            ],
        )
    except Exception as exc:
        return CheckResult(
            "ch12_todo_03",
            False,
            "failed",
            "To-Do 12.3 — Error Instantiating LabResult",
            f"LabResult('P-101', 10.4) raised {type(exc).__name__}: {exc}",
        )

    errors = []
    if not dataclasses.is_dataclass(cls):
        errors.append(
            "LabResult must be decorated with `@dataclass` (from `dataclasses import dataclass`)."
        )
    else:
        field_names = [f.name for f in dataclasses.fields(cls)]
        if field_names != ["patient_id", "hb_g_dl"]:
            errors.append(
                f"Expected dataclass fields ['patient_id', 'hb_g_dl'], found {field_names!r}."
            )

    # Value equality from @dataclass
    if not (r_anemic == r_same) or (r_anemic == r_border):
        errors.append(
            "Expected `@dataclass` value equality (`LabResult('P-101', 10.4) == LabResult('P-101', 10.4)` -> True)."
        )

    if not hasattr(r_anemic, "is_anemic") or not callable(r_anemic.is_anemic):
        return CheckResult(
            "ch12_todo_03",
            False,
            "failed",
            "To-Do 12.3 — Missing `.is_anemic()` Method",
            "Add `def is_anemic(self) -> bool:` returning `self.hb_g_dl < 11.0`.",
            details=errors,
        )

    for obj, expected, label in [
        (r_anemic, True, "LabResult('P-101', 10.4).is_anemic()"),
        (cls("P-104", 10.9), True, "LabResult('P-104', 10.9).is_anemic()"),
        (r_border, False, "LabResult('P-102', 11.0).is_anemic()"),
        (r_normal, False, "LabResult('P-103', 12.5).is_anemic()"),
    ]:
        try:
            got = obj.is_anemic()
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got is None:
            return _none_return_trap("ch12_todo_03", "To-Do 12.3", label, 10)
        if not isinstance(got, bool):
            errors.append(f"{label} should return a bool (True/False), got {type(got).__name__} ({got!r}).")
        elif got is not expected:
            extra = " (use strict `< 11.0`, not `<= 11.0`!)" if obj.hb_g_dl == 11.0 else ""
            errors.append(f"{label}: expected {expected}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch12_todo_03",
            False,
            "failed",
            "To-Do 12.3 — Check `@dataclass` Definition & `.is_anemic()` Cutoff",
            "Use `@dataclass` with `patient_id: str`, `hb_g_dl: float`, and `.is_anemic()` returning `self.hb_g_dl < 11.0`.",
            details=errors,
        )

    return CheckResult(
        "ch12_todo_03",
        True,
        "passed",
        "To-Do 12.3 Passed — `@dataclass LabResult` Verified!",
        "Auto-generated `__init__`, `__repr__`, and value equality `__eq__`, plus exact `< 11.0` anemia detection.",
        tip="`@dataclass` eliminates boilerplate for structured scientific records, hyperparameters, and experiment configs.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_todo_03",
        chapter="ch12",
        title="To-Do 12.3: @dataclass LabResult with .is_anemic()",
        book_section="Section 12.3.2 (Structured Records with @dataclass)",
        points=10,
        hints={
            1: "Import `dataclass` from `dataclasses`, decorate `LabResult`, declare `patient_id: str` and `hb_g_dl: float`, and return `self.hb_g_dl < 11.0` in `is_anemic(self)`.",
            2: "Blueprint:\n1. `@dataclass`\n2. `class LabResult:`\n3. `patient_id: str` and `hb_g_dl: float`\n4. `def is_anemic(self): return self.hb_g_dl < 11.0`",
            3: "from dataclasses import dataclass\n\n@dataclass\nclass LabResult:\n    patient_id: str\n    hb_g_dl: float\n\n    def is_anemic(self):\n        return self.hb_g_dl < ___",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 12.4: MeanImputer (Scikit-Learn style .fit / .transform)
# ---------------------------------------------------------------------------
def _verify_todo_04(cls: Any) -> CheckResult:
    bad = _check_class("ch12_todo_04", "To-Do 12.4", cls, "MeanImputer", 10)
    if bad is not None:
        return bad

    try:
        imp1 = cls()
        imp2 = cls()
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_04",
            False,
            "failed",
            "To-Do 12.4 — MeanImputer() Should Take No Required Arguments",
            f"Calling MeanImputer() raised TypeError: {exc}",
            details=["Define `def __init__(self):` initializing `self.mean_ = None`."],
        )
    except Exception as exc:
        return CheckResult(
            "ch12_todo_04",
            False,
            "failed",
            "To-Do 12.4 — Error Instantiating MeanImputer",
            f"MeanImputer() raised {type(exc).__name__}: {exc}",
        )

    for method in ("fit", "transform"):
        if not hasattr(imp1, method) or not callable(getattr(imp1, method)):
            return CheckResult(
                "ch12_todo_04",
                False,
                "failed",
                f"To-Do 12.4 — Missing `.{method}()` Method",
                f"MeanImputer must implement both `.fit(self, values)` and `.transform(self, values)`.",
            )

    train_data = [10.0, None, 11.0, 999, 11.5]
    train_snapshot = list(train_data)
    test_data = [9.5, None, 999, 12.2]
    test_snapshot = list(test_data)

    try:
        imp1.fit(train_data)
        transformed = imp1.transform(test_data)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch12_todo_04",
            False,
            "failed",
            "To-Do 12.4 — Runtime Error in `.fit()` / `.transform()`",
            f"Calling fit/transform raised {type(exc).__name__}: {exc}",
            details=["Check that you skip `v is None or v == 999` BEFORE summing or averaging."],
        )

    if transformed is None:
        return _none_return_trap("ch12_todo_04", "To-Do 12.4", "imputer.transform(test_data)", 10)

    errors = []
    if train_data != train_snapshot:
        errors.append("`.fit(values)` mutated the caller's training list! Never modify input lists in place.")
    if test_data != test_snapshot:
        errors.append("`.transform(values)` mutated the caller's input list! Always build and return a NEW list.")

    # Expected train mean: (10.0 + 11.0 + 11.5) / 3 = 10.8333... -> 10.83
    expected_test = [9.5, 10.83, 10.83, 12.2]
    if not isinstance(transformed, list) or len(transformed) != len(expected_test):
        errors.append(f"transform({test_snapshot}): expected list {expected_test}, got {transformed!r}")
    else:
        for got_v, exp_v in zip(transformed, expected_test):
            if not _is_number(got_v) or not math.isclose(got_v, exp_v, abs_tol=1e-2):
                errors.append(
                    f"transform({test_snapshot}) after fitting on {train_snapshot}: "
                    f"expected {expected_test}, got {transformed!r}"
                )
                break

    # Verify independent second imputer with a different training distribution
    train2 = [12.0, 14.0, 999]  # mean = 13.0
    try:
        imp2.fit(train2)
        out2 = imp2.transform([None, 11.5])
        out1_again = imp1.transform([None])
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Second MeanImputer instance raised {type(exc).__name__}: {exc}")
    else:
        if out2 != [13.0, 11.5]:
            errors.append(f"Second imputer fit([12.0, 14.0, 999]).transform([None, 11.5]): expected [13.0, 11.5], got {out2!r}")
        if not (isinstance(out1_again, list) and len(out1_again) == 1 and math.isclose(out1_again[0], 10.83, abs_tol=1e-2)):
            errors.append(
                f"Fitting `imp2` overwrote `imp1`'s learned mean! Got {out1_again!r} instead of [10.83]. "
                "Store learned state on `self`, never as a class variable."
            )

    if errors:
        return CheckResult(
            "ch12_todo_04",
            False,
            "failed",
            "To-Do 12.4 — Check Fit/Transform Logic & Non-Mutation",
            "Fit the mean on valid values (skipping None and 999), and return a new list replacing None/999 with round(mean, 2).",
            details=errors,
        )

    return CheckResult(
        "ch12_todo_04",
        True,
        "passed",
        "To-Do 12.4 Passed — Scikit-Learn Style `MeanImputer` Verified!",
        "Learned train-set mean (10.83), applied it cleanly to unseen test data without leakage or mutation, and kept instances isolated.",
        tip="Fitting on training data and transforming test data with the SAME learned `self.mean_` is the core anti-leakage pattern in Scikit-Learn!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_todo_04",
        chapter="ch12",
        title="To-Do 12.4: Scikit-Learn Style MeanImputer (.fit / .transform)",
        book_section="Section 12.3.4 (The ML Bridge — Scikit-Learn Estimator API)",
        points=10,
        hints={
            1: "In `fit(self, values)`, collect valid entries (`v is not None and v != 999`), store `self.mean_ = round(sum(valid) / len(valid), 2)`, and return `self`. In `transform(self, values)`, return a new list replacing `None` or `999` with `self.mean_`.",
            2: "Blueprint:\n1. `__init__`: `self.mean_ = None`\n2. `fit(self, values)`: filter out `None` and `999`, compute `self.mean_ = round(sum(valid) / len(valid), 2)`, `return self`\n3. `transform(self, values)`: return `[self.mean_ if (v is None or v == 999) else v for v in values]`",
            3: "class MeanImputer:\n    def __init__(self):\n        self.mean_ = None\n\n    def fit(self, values):\n        valid = [v for v in values if v is not None and v != 999]\n        self.mean_ = round(sum(valid) / len(valid), ___)\n        return self\n\n    def transform(self, values):\n        return [\n            self.mean_ if (v is None or v == 999) else v\n            for v in values\n        ]",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 12.5: MinMaxScaler (.fit / .transform with constant guard)
# ---------------------------------------------------------------------------
def _verify_todo_05(cls: Any) -> CheckResult:
    bad = _check_class("ch12_todo_05", "To-Do 12.5", cls, "MinMaxScaler", 10)
    if bad is not None:
        return bad

    try:
        scaler = cls()
        const_scaler = cls()
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_05",
            False,
            "failed",
            "To-Do 12.5 — MinMaxScaler() Takes No Required Arguments",
            f"Calling MinMaxScaler() raised TypeError: {exc}",
            details=["Define `def __init__(self):` initializing `self.min_ = None` and `self.max_ = None`."],
        )
    except Exception as exc:
        return CheckResult(
            "ch12_todo_05",
            False,
            "failed",
            "To-Do 12.5 — Error Instantiating MinMaxScaler",
            f"MinMaxScaler() raised {type(exc).__name__}: {exc}",
        )

    for method in ("fit", "transform"):
        if not hasattr(scaler, method) or not callable(getattr(scaler, method)):
            return CheckResult(
                "ch12_todo_05",
                False,
                "failed",
                f"To-Do 12.5 — Missing `.{method}()` Method",
                f"MinMaxScaler must implement both `.fit(self, values)` and `.transform(self, values)`.",
            )

    train_vals = [10.0, 20.0, 30.0, 50.0]
    train_snap = list(train_vals)
    test_vals = [15.0, 35.0, 40.0]
    test_snap = list(test_vals)

    try:
        scaler.fit(train_vals)
        scaled_train = scaler.transform(train_vals)
        scaled_test = scaler.transform(test_vals)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch12_todo_05",
            False,
            "failed",
            "To-Do 12.5 — Runtime Error in `.fit()` / `.transform()`",
            f"MinMaxScaler raised {type(exc).__name__}: {exc}",
        )

    if scaled_train is None or scaled_test is None:
        return _none_return_trap("ch12_todo_05", "To-Do 12.5", "scaler.transform(...)", 10)

    errors = []
    if train_vals != train_snap or test_vals != test_snap:
        errors.append("Do NOT mutate input lists inside `.fit()` or `.transform()`.")

    if scaled_train != [0.0, 0.25, 0.5, 1.0]:
        errors.append(
            f"fit([10, 20, 30, 50]).transform([10, 20, 30, 50]): expected [0.0, 0.25, 0.5, 1.0], got {scaled_train!r}"
        )
    if scaled_test != [0.125, 0.625, 0.75]:
        errors.append(
            f"fit([10, 20, 30, 50]).transform([15, 35, 40]): expected [0.125, 0.625, 0.75], got {scaled_test!r} "
            "(make sure `.transform()` uses the fitted `self.min_` and `self.max_`, NOT the test list's own min/max!)."
        )

    # Rounding to 4 decimals check
    try:
        r_scaler = cls()
        r_scaler.fit([12.0, 15.0, 21.0])
        r_out = r_scaler.transform([14.0])
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Rounding test raised {type(exc).__name__}: {exc}")
    else:
        if r_out != [0.2222]:
            errors.append(f"fit([12, 15, 21]).transform([14]): expected [0.2222] (4 decimals), got {r_out!r}")

    # Constant feature guard: max == min -> 0.0 without ZeroDivisionError
    try:
        const_scaler.fit([7.5, 7.5, 7.5])
        const_out = const_scaler.transform([7.5, 7.5])
    except NotImplementedError:
        raise
    except ZeroDivisionError:
        errors.append(
            "Constant feature [7.5, 7.5, 7.5] raised ZeroDivisionError — guard `if self.max_ == self.min_: return [0.0 for _ in values]`!"
        )
    except Exception as exc:
        errors.append(f"Constant feature test raised {type(exc).__name__}: {exc}")
    else:
        if const_out != [0.0, 0.0]:
            errors.append(f"Constant feature transform([7.5, 7.5]): expected [0.0, 0.0], got {const_out!r}")

    if errors:
        return CheckResult(
            "ch12_todo_05",
            False,
            "failed",
            "To-Do 12.5 — Check Min-Max Scaling, Rounding & Constant Guard",
            "Scale each x via `round((x - self.min_) / (self.max_ - self.min_), 4)`, returning `0.0` when `max_ == min_`.",
            details=errors,
        )

    return CheckResult(
        "ch12_todo_05",
        True,
        "passed",
        "To-Do 12.5 Passed — `MinMaxScaler` Verified!",
        "Passed train scaling, unseen test scaling using fitted bounds, 4-decimal rounding, constant-feature guard, and non-mutation.",
        tip="Guarding against zero-range features prevents mysterious NaN/ZeroDivisionError crashes in clinical feature pipelines.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_todo_05",
        chapter="ch12",
        title="To-Do 12.5: MinMaxScaler (.fit / .transform)",
        book_section="Section 12.3.4 (The ML Bridge — Scikit-Learn Estimator API)",
        points=10,
        hints={
            1: "Store `self.min_ = min(values)` and `self.max_ = max(values)` in `fit()`. In `transform()`, if `self.max_ == self.min_` return `[0.0 for _ in values]`, else scale each element and round to 4 decimals.",
            2: "Blueprint:\n1. `fit`: `self.min_ = min(values); self.max_ = max(values); return self`\n2. `transform`: compute `span = self.max_ - self.min_`; if `span == 0`, return `[0.0 for _ in values]`; else return `[round((x - self.min_) / span, 4) for x in values]`.",
            3: "class MinMaxScaler:\n    def __init__(self):\n        self.min_ = None\n        self.max_ = None\n\n    def fit(self, values):\n        self.min_ = min(values)\n        self.max_ = max(values)\n        return self\n\n    def transform(self, values):\n        span = self.max_ - self.min_\n        if span == 0:\n            return [0.0 for _ in values]\n        return [round((x - self.min_) / ___, 4) for x in values]",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 12.6: LinearLayer (Mini PyTorch nn.Module: forward & __call__)
# ---------------------------------------------------------------------------
def _verify_todo_06(cls: Any) -> CheckResult:
    bad = _check_class("ch12_todo_06", "To-Do 12.6", cls, "LinearLayer", 10)
    if bad is not None:
        return bad

    try:
        layer1 = cls(2.5, -1.0)
        layer2 = cls(0.5, 3.0)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_todo_06",
            False,
            "failed",
            "To-Do 12.6 — Check `__init__(self, weight, bias)` Signature",
            f"LinearLayer(2.5, -1.0) raised TypeError: {exc}",
            details=["Define `def __init__(self, weight, bias):` storing `self.weight` and `self.bias`."],
        )
    except Exception as exc:
        return CheckResult(
            "ch12_todo_06",
            False,
            "failed",
            "To-Do 12.6 — Error Instantiating LinearLayer",
            f"LinearLayer(2.5, -1.0) raised {type(exc).__name__}: {exc}",
        )

    errors = []
    if not hasattr(layer1, "weight") or not hasattr(layer1, "bias"):
        errors.append("Store `self.weight = weight` and `self.bias = bias` inside `__init__`.")

    if not hasattr(layer1, "forward") or not callable(layer1.forward):
        return CheckResult(
            "ch12_todo_06",
            False,
            "failed",
            "To-Do 12.6 — Missing `.forward(self, x)` Method",
            "Define `def forward(self, x): return self.weight * x + self.bias`.",
        )

    if not callable(layer1):
        return CheckResult(
            "ch12_todo_06",
            False,
            "failed",
            "To-Do 12.6 — Missing `__call__(self, x)` Dunder Method",
            "A LinearLayer instance is not callable yet (`layer(x)` fails).",
            details=[
                "Implement `def __call__(self, x): return self.forward(x)` so calling `layer(x)` works just like PyTorch `nn.Module`!",
            ],
        )

    cases = [
        (layer1, 4.0, 9.0, "LinearLayer(2.5, -1.0)"),
        (layer1, 0.0, -1.0, "LinearLayer(2.5, -1.0)"),
        (layer1, -2.0, -6.0, "LinearLayer(2.5, -1.0)"),
        (layer2, 10.0, 8.0, "LinearLayer(0.5, 3.0)"),
    ]
    for layer, x_val, expected, desc in cases:
        try:
            f_val = layer.forward(x_val)
            c_val = layer(x_val)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{desc} on x={x_val} raised {type(exc).__name__}: {exc}")
            continue
        if f_val is None:
            return _none_return_trap("ch12_todo_06", "To-Do 12.6", f"{desc}.forward({x_val})", 10)
        if c_val is None:
            return _none_return_trap("ch12_todo_06", "To-Do 12.6", f"{desc}({x_val})", 10)
        if not _is_number(f_val) or not math.isclose(f_val, expected, abs_tol=1e-6):
            errors.append(f"{desc}.forward({x_val}): expected {expected}, got {f_val!r}")
        if not _is_number(c_val) or not math.isclose(c_val, expected, abs_tol=1e-6):
            errors.append(f"{desc}({x_val}): expected {expected}, got {c_val!r}")

    # Verify __call__ actually delegates to self.forward(x) (PyTorch pattern)
    if not errors:
        called_forward = []
        orig_forward = layer1.forward

        def _spy_forward(x: Any) -> Any:
            called_forward.append(x)
            return orig_forward(x)

        layer1.forward = _spy_forward
        try:
            spy_res = layer1(4.0)
        finally:
            layer1.forward = orig_forward

        if not called_forward or not math.isclose(spy_res, 9.0, abs_tol=1e-6):
            errors.append(
                "`__call__(self, x)` should delegate directly to `return self.forward(x)` rather than duplicating the formula!"
            )

    if errors:
        return CheckResult(
            "ch12_todo_06",
            False,
            "failed",
            "To-Do 12.6 — Check `.forward(x)` & `__call__(x)` Delegation",
            "Compute `self.weight * x + self.bias` in `forward(self, x)`, and return `self.forward(x)` from `__call__(self, x)`.",
            details=errors,
        )

    return CheckResult(
        "ch12_todo_06",
        True,
        "passed",
        "To-Do 12.6 Passed — Mini PyTorch `LinearLayer` Verified!",
        "Verified `layer.forward(x)`, callable syntax `layer(x)`, and true `__call__ -> forward` delegation across multiple layers.",
        tip="Now you know the exact secret behind PyTorch's `y_pred = model(x)`: `nn.Module.__call__` runs hooks and delegates straight to your `forward()`!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_todo_06",
        chapter="ch12",
        title="To-Do 12.6: Mini PyTorch LinearLayer (forward & __call__)",
        book_section="Section 12.3.4 (The ML Bridge — PyTorch nn.Module Pattern)",
        points=10,
        hints={
            1: "Store `self.weight` and `self.bias` in `__init__`. Compute `self.weight * x + self.bias` in `forward(self, x)`, and have `__call__(self, x)` return `self.forward(x)`.",
            2: "Blueprint:\n1. `__init__(self, weight, bias)`: assign `self.weight`, `self.bias`\n2. `forward(self, x)`: `return self.weight * x + self.bias`\n3. `__call__(self, x)`: `return self.forward(x)`",
            3: "class LinearLayer:\n    def __init__(self, weight, bias):\n        self.weight = weight\n        self.bias = bias\n\n    def forward(self, x):\n        return self.weight * x + self.bias\n\n    def __call__(self, x):\n        return self.___(x)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 12.1 (30 pts): MaternalPatient (validation, anemia_grade, repr)
# ---------------------------------------------------------------------------
def _verify_exam_01(cls: Any) -> CheckResult:
    global _LAST_MATERNAL_PATIENT_CLS
    bad = _check_class("ch12_exam_01", "Exam Task 12.1", cls, "MaternalPatient", 30)
    if bad is not None:
        return bad

    _LAST_MATERNAL_PATIENT_CLS = cls

    # Nominal instantiation first so starter stub NotImplementedError propagates
    try:
        nominal = cls("M-101", "Dhaka", 10.4)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_exam_01",
            False,
            "failed",
            "Exam Task 12.1 — Check `__init__(self, patient_id, district, hb_g_dl)`",
            f"MaternalPatient('M-101', 'Dhaka', 10.4) raised TypeError: {exc}",
            points_earned=0,
            points_possible=30,
        )
    except Exception as exc:
        return CheckResult(
            "ch12_exam_01",
            False,
            "failed",
            "Exam Task 12.1 — Runtime Error Instantiating Nominal Patient",
            f"MaternalPatient('M-101', 'Dhaka', 10.4) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    errors = []
    for attr, expected in [("patient_id", "M-101"), ("district", "Dhaka"), ("hb_g_dl", 10.4)]:
        if not hasattr(nominal, attr):
            errors.append(f"Missing instance attribute `self.{attr}`.")
        elif getattr(nominal, attr) != expected:
            errors.append(f"`nominal.{attr}`: expected {expected!r}, got {getattr(nominal, attr)!r}.")

    # Check anemia_grade method
    if not hasattr(nominal, "anemia_grade") or not callable(nominal.anemia_grade):
        return CheckResult(
            "ch12_exam_01",
            False,
            "failed",
            "Exam Task 12.1 — Missing `.anemia_grade()` Method",
            "Implement `.anemia_grade(self)` returning 'Severe', 'Moderate', 'Mild', or 'Normal'.",
            points_earned=0,
            points_possible=30,
        )

    grade_cases = [
        (2.0, "Severe"),
        (6.9, "Severe"),
        (7.0, "Moderate"),
        (9.9, "Moderate"),
        (10.0, "Mild"),
        (10.9, "Mild"),
        (11.0, "Normal"),
        (13.5, "Normal"),
        (25.0, "Normal"),
    ]
    for hb, expected in grade_cases:
        label = f"MaternalPatient('M-X', 'Dhaka', {hb}).anemia_grade()"
        try:
            p = cls("M-X", "Dhaka", hb)
            got = p.anemia_grade()
        except NotImplementedError:
            raise
        except ValueError:
            errors.append(
                f"Valid boundary hb_g_dl={hb} raised ValueError — valid range is [2.0, 25.0] INCLUSIVE!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got is None:
            return _none_return_trap("ch12_exam_01", "Exam Task 12.1", label, 30)
        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    # Validation checks: out-of-range hb_g_dl MUST raise ValueError
    for bad_hb in (1.9, 0.0, -4.5, 25.1, 999.0):
        try:
            cls("M-BAD", "Sylhet", bad_hb)
        except NotImplementedError:
            raise
        except ValueError:
            pass
        except Exception as exc:
            errors.append(
                f"Invalid hb_g_dl={bad_hb} raised {type(exc).__name__}, expected ValueError."
            )
        else:
            errors.append(
                f"Invalid hb_g_dl={bad_hb} did NOT raise ValueError! Guard `if not (2.0 <= hb_g_dl <= 25.0): raise ValueError(...)`."
            )

    # Check __repr__
    r = repr(nominal)
    if not isinstance(r, str) or _is_default_repr(r):
        errors.append(f"Implement a custom `__repr__(self)` — currently repr() returned {r!r}.")
    elif "MaternalPatient" not in r or "M-101" not in r:
        errors.append(f"`__repr__` should include 'MaternalPatient' and patient_id ('M-101'), got {r!r}.")

    if errors:
        return CheckResult(
            "ch12_exam_01",
            False,
            "failed",
            "Exam Task 12.1 — Check Validation Bounds, WHO Grades & `__repr__`",
            "Validate `2.0 <= hb_g_dl <= 25.0` (else ValueError), grade Severe/Moderate/Mild/Normal, and customize `__repr__`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch12_exam_01",
        True,
        "passed",
        "Exam Task 12.1 Passed (30/30 pts) — Validated `MaternalPatient` Verified!",
        "Passed inclusive [2.0, 25.0] validation, all 4 WHO anemia grade boundaries, invalid-value ValueError guards, and `__repr__`.",
        points_earned=30,
        points_possible=30,
        tip="Validating physical bounds inside `__init__` guarantees no impossible sentinel like `999` can ever sneak into downstream analytics.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_exam_01",
        chapter="ch12",
        title="Exam 12.1: Validated MaternalPatient Class",
        book_section="Chapter 12 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "In `__init__`, raise `ValueError` if `not (2.0 <= hb_g_dl <= 25.0)`, then store `patient_id`, `district`, and `hb_g_dl`. In `anemia_grade()`, test `< 7.0`, `< 10.0`, `< 11.0`, else `'Normal'`.",
            2: "Blueprint:\n1. Validate `if not (2.0 <= hb_g_dl <= 25.0): raise ValueError(...)`\n2. Store `self.patient_id`, `self.district`, `self.hb_g_dl`\n3. `anemia_grade`: `< 7.0 -> 'Severe'`, `< 10.0 -> 'Moderate'`, `< 11.0 -> 'Mild'`, else `'Normal'`\n4. `__repr__`: include `MaternalPatient` and `self.patient_id`.",
            3: "class MaternalPatient:\n    def __init__(self, patient_id, district, hb_g_dl):\n        if not (2.0 <= hb_g_dl <= 25.0):\n            raise ValueError(f'Invalid hb_g_dl: {hb_g_dl}')\n        self.patient_id = patient_id\n        self.district = district\n        self.hb_g_dl = hb_g_dl\n\n    def anemia_grade(self):\n        if self.hb_g_dl < 7.0:\n            return 'Severe'\n        if self.hb_g_dl < 10.0:\n            return 'Moderate'\n        if self.hb_g_dl < ___:\n            return 'Mild'\n        return 'Normal'",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 12.2 (35 pts): ZScoreScaler (fit, transform, unfitted guard)
# ---------------------------------------------------------------------------
def _verify_exam_02(cls: Any) -> CheckResult:
    bad = _check_class("ch12_exam_02", "Exam Task 12.2", cls, "ZScoreScaler", 35)
    if bad is not None:
        return bad

    try:
        unfitted = cls()
        scaler = cls()
        const_scaler = cls()
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_exam_02",
            False,
            "failed",
            "Exam Task 12.2 — ZScoreScaler() Takes No Required Arguments",
            f"ZScoreScaler() raised TypeError: {exc}",
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch12_exam_02",
            False,
            "failed",
            "Exam Task 12.2 — Error Instantiating ZScoreScaler",
            f"ZScoreScaler() raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    for method in ("fit", "transform"):
        if not hasattr(scaler, method) or not callable(getattr(scaler, method)):
            return CheckResult(
                "ch12_exam_02",
                False,
                "failed",
                f"Exam Task 12.2 — Missing `.{method}()` Method",
                "ZScoreScaler must implement both `.fit(self, values)` and `.transform(self, values)`.",
                points_earned=0,
                points_possible=35,
            )

    errors = []

    # 1. Unfitted .transform() MUST raise RuntimeError
    try:
        unfitted.transform([10.0, 12.0])
    except NotImplementedError:
        raise
    except RuntimeError:
        pass
    except Exception as exc:
        errors.append(
            f"Calling `.transform()` before `.fit()` raised {type(exc).__name__} instead of RuntimeError! "
            "Check `if self.mean_ is None or self.std_ is None: raise RuntimeError(...)`."
        )
    else:
        errors.append(
            "Calling `.transform()` before `.fit()` did NOT raise RuntimeError!"
        )

    # 2. Nominal fit & transform (population std)
    train_vals = [10.0, 12.0, 14.0, 16.0]
    train_snap = list(train_vals)
    test_vals = [10.0, 13.0, 16.0]
    test_snap = list(test_vals)

    expected_mean = 13.0
    expected_std = math.sqrt(5.0)  # population std = sqrt(20 / 4) = 2.2360679...
    sample_std = math.sqrt(20.0 / 3.0)

    try:
        scaler.fit(train_vals)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch12_exam_02",
            False,
            "failed",
            "Exam Task 12.2 — Runtime Error in `.fit()`",
            f"scaler.fit([10.0, 12.0, 14.0, 16.0]) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if train_vals != train_snap:
        errors.append("`.fit(values)` mutated the caller's training list!")

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "std_"):
        errors.append("Store fitted parameters on `self.mean_` and `self.std_` inside `.fit(values)`.")
    else:
        if not _is_number(scaler.mean_) or not math.isclose(scaler.mean_, expected_mean, abs_tol=1e-4):
            errors.append(f"After fit({train_snap}), expected `mean_ == 13.0`, got {scaler.mean_!r}.")
        if _is_number(scaler.std_) and math.isclose(scaler.std_, sample_std, abs_tol=1e-3):
            errors.append(
                f"Got `std_ == {scaler.std_:.4f}` (sample std dividing by N - 1) — "
                f"use POPULATION std dividing by N (`{expected_std:.4f}`)!"
            )
        elif not _is_number(scaler.std_) or not math.isclose(scaler.std_, expected_std, abs_tol=1e-4):
            errors.append(
                f"After fit({train_snap}), expected population `std_ == {expected_std:.4f}`, got {scaler.std_!r}."
            )

    try:
        z_out = scaler.transform(test_vals)
        z_unseen = scaler.transform([15.0, 11.0])
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"scaler.transform(...) raised {type(exc).__name__}: {exc}")
    else:
        if z_out is None:
            return _none_return_trap("ch12_exam_02", "Exam Task 12.2", "scaler.transform(test_vals)", 35)
        if test_vals != test_snap:
            errors.append("`.transform(values)` mutated the caller's input list!")
        if z_out != [-1.3416, 0.0, 1.3416]:
            errors.append(f"transform([10.0, 13.0, 16.0]): expected [-1.3416, 0.0, 1.3416], got {z_out!r}")
        if z_unseen != [0.8944, -0.8944]:
            errors.append(
                f"transform([15.0, 11.0]) using fitted mean_/std_: expected [0.8944, -0.8944], got {z_unseen!r}"
            )

    # 3. Constant-feature edge case (std_ == 0 -> [0.0, ...])
    try:
        const_scaler.fit([5.0, 5.0, 5.0])
        const_out = const_scaler.transform([5.0, 8.0])
    except NotImplementedError:
        raise
    except ZeroDivisionError:
        errors.append(
            "Constant feature [5.0, 5.0, 5.0] raised ZeroDivisionError — return `[0.0 for _ in values]` when `self.std_ == 0`!"
        )
    except Exception as exc:
        errors.append(f"Constant feature test raised {type(exc).__name__}: {exc}")
    else:
        if const_out != [0.0, 0.0]:
            errors.append(f"Constant feature transform([5.0, 8.0]): expected [0.0, 0.0], got {const_out!r}")

    if errors:
        return CheckResult(
            "ch12_exam_02",
            False,
            "failed",
            "Exam Task 12.2 — Check Unfitted Guard, Population Std & Scaling",
            "Store `mean_` and population `std_` in `fit()`, raise `RuntimeError` if unfitted, guard `std_ == 0`, and round to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch12_exam_02",
        True,
        "passed",
        "Exam Task 12.2 Passed (35/35 pts) — Production `ZScoreScaler` Verified!",
        "Passed unfitted RuntimeError guard, population std_, 4-decimal z-scores on unseen data, zero-variance guard, and non-mutation.",
        points_earned=35,
        points_possible=35,
        tip="Raising `RuntimeError` on unfitted `.transform()` mirrors Scikit-Learn's `NotFittedError` — catching pipeline ordering bugs instantly.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_exam_02",
        chapter="ch12",
        title="Exam 12.2: Production ZScoreScaler (.fit / .transform)",
        book_section="Chapter 12 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Initialize `self.mean_ = None` and `self.std_ = None`. In `fit(values)`, compute population std dividing squared deviations by `len(values)`. In `transform(values)`, raise `RuntimeError` if unfitted, return `[0.0 ...]` if `std_ == 0`, else `round((x - mean_) / std_, 4)`.",
            2: "Blueprint:\n1. `fit`: `mean = sum(values) / len(values)`; `var = sum((x - mean) ** 2 for x in values) / len(values)`; `self.mean_ = mean`; `self.std_ = var ** 0.5`; `return self`\n2. `transform`: `if self.mean_ is None or self.std_ is None: raise RuntimeError(...)`; `if self.std_ == 0: return [0.0 for _ in values]`; return `[round((x - self.mean_) / self.std_, 4) for x in values]`.",
            3: "class ZScoreScaler:\n    def __init__(self):\n        self.mean_ = None\n        self.std_ = None\n\n    def fit(self, values):\n        mean = sum(values) / len(values)\n        var = sum((x - mean) ** 2 for x in values) / len(values)\n        self.mean_ = mean\n        self.std_ = var ** 0.5\n        return self\n\n    def transform(self, values):\n        if self.mean_ is None or self.std_ is None:\n            raise ___('ZScoreScaler must be fitted before transform().')\n        if self.std_ == 0:\n            return [0.0 for _ in values]\n        return [round((x - self.mean_) / self.std_, 4) for x in values]",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 12.3 (35 pts): EpidemicRegistry (Composition + Type Guard)
# ---------------------------------------------------------------------------
def _resolve_maternal_patient_cls(registry_cls: Any) -> Any:
    for attr_name in ("add", "__init__"):
        fn = getattr(registry_cls, attr_name, None)
        globs = getattr(fn, "__globals__", None)
        if isinstance(globs, dict) and isinstance(globs.get("MaternalPatient"), type):
            return globs["MaternalPatient"]
    return _LAST_MATERNAL_PATIENT_CLS


def _verify_exam_03(cls: Any) -> CheckResult:
    bad = _check_class("ch12_exam_03", "Exam Task 12.3", cls, "EpidemicRegistry", 35)
    if bad is not None:
        return bad

    # Instantiate first so starter stub's NotImplementedError propagates cleanly
    try:
        reg = cls()
        reg_isolated = cls()
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — EpidemicRegistry() Takes No Required Arguments",
            f"Calling EpidemicRegistry() raised TypeError: {exc}",
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — Error Instantiating EpidemicRegistry",
            f"EpidemicRegistry() raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    for method in ("add", "anemia_prevalence_pct", "count_by_district"):
        if not hasattr(reg, method) or not callable(getattr(reg, method)):
            return CheckResult(
                "ch12_exam_03",
                False,
                "failed",
                f"Exam Task 12.3 — Missing `.{method}()` Method",
                "EpidemicRegistry must implement `.add(patient)`, `__len__()`, `.anemia_prevalence_pct()`, and `.count_by_district()`.",
                points_earned=0,
                points_possible=35,
            )

    errors = []

    # Empty state checks
    try:
        if len(reg) != 0:
            errors.append(f"Empty registry should have len(reg) == 0, got {len(reg)!r}.")
    except TypeError as exc:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — Missing `__len__(self)` Method",
            f"len(reg) raised TypeError: {exc}",
            details=["Implement `def __len__(self): return len(self.patients)`."],
            points_earned=0,
            points_possible=35,
        )

    try:
        empty_prev = reg.anemia_prevalence_pct()
    except ZeroDivisionError:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — ZeroDivisionError on Empty Registry",
            "`.anemia_prevalence_pct()` raised ZeroDivisionError on an empty registry.",
            details=["Return `None` immediately if the registry has no patients!"],
            points_earned=0,
            points_possible=35,
        )
    if empty_prev is not None:
        errors.append(f"Empty registry `.anemia_prevalence_pct()` should return None, got {empty_prev!r}.")

    empty_counts = reg.count_by_district()
    if empty_counts != {}:
        errors.append(f"Empty registry `.count_by_district()` should return {{}}, got {empty_counts!r}.")

    # Type safety: non-MaternalPatient inputs MUST raise TypeError
    for invalid_obj in ("M-01", {"patient_id": "M-01", "hb_g_dl": 10.2}, 42):
        try:
            reg.add(invalid_obj)
        except NotImplementedError:
            raise
        except TypeError:
            pass
        except Exception as exc:
            errors.append(
                f"reg.add({invalid_obj!r}) raised {type(exc).__name__}, expected TypeError."
            )
        else:
            errors.append(
                f"reg.add({invalid_obj!r}) did NOT raise TypeError! Guard `if not isinstance(patient, MaternalPatient): raise TypeError(...)`."
            )

    # Resolve MaternalPatient class from learner's namespace
    mp_cls = _resolve_maternal_patient_cls(cls)
    if mp_cls is None:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — MaternalPatient Class Not Found in Scope",
            "Complete Exam Task 12.1 (`MaternalPatient`) in the same notebook before grading `EpidemicRegistry`.",
            points_earned=0,
            points_possible=35,
        )

    patient_specs = [
        ("M-01", "Dhaka", 6.5),    # Severe (anemic)
        ("M-02", "Dhaka", 9.8),    # Moderate (anemic)
        ("M-03", "Sylhet", 10.6),  # Mild (anemic)
        ("M-04", "Dhaka", 10.9),   # Mild (anemic)
        ("M-05", "Sylhet", 11.0),  # Normal (boundary!)
        ("M-06", "Khulna", 12.4),  # Normal
    ]
    patients = []
    for pid, dist, hb in patient_specs:
        try:
            patients.append(mp_cls(pid, dist, hb))
        except NotImplementedError:
            return CheckResult(
                "ch12_exam_03",
                False,
                "failed",
                "Exam Task 12.3 — Complete Task 12.1 (MaternalPatient) First",
                "`EpidemicRegistry` stores `MaternalPatient` objects, but `MaternalPatient` still raises NotImplementedError.",
                details=["Finish Exam Task 12.1 first, then re-run `tutor.check_exam('ch12', exam_submissions)`."],
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch12_exam_03",
                False,
                "failed",
                "Exam Task 12.3 — Could Not Instantiate MaternalPatient",
                f"MaternalPatient({pid!r}, {dist!r}, {hb}) raised {type(exc).__name__}: {exc}",
                details=["Verify Exam Task 12.1 passes before checking Exam Task 12.3."],
                points_earned=0,
                points_possible=35,
            )

    for p in patients:
        try:
            reg.add(p)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"reg.add({p!r}) raised {type(exc).__name__}: {exc}")

    if len(reg) != 6:
        errors.append(f"After adding 6 MaternalPatient objects, expected len(reg) == 6, got {len(reg)!r}.")

    if len(reg_isolated) != 0:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — Mutable Class Attribute Leak Detected!",
            "Adding patients to `reg` also modified a second `EpidemicRegistry()` instance!",
            details=["Initialize `self.patients = []` INSIDE `__init__(self)`, never at the class body level."],
            points_earned=0,
            points_possible=35,
        )

    # 4 of 6 anemic -> 66.666...% -> 66.7
    try:
        prev_pct = reg.anemia_prevalence_pct()
    except Exception as exc:
        errors.append(f"reg.anemia_prevalence_pct() raised {type(exc).__name__}: {exc}")
    else:
        if prev_pct is None:
            return _none_return_trap("ch12_exam_03", "Exam Task 12.3", "reg.anemia_prevalence_pct()", 35)
        if not _is_number(prev_pct) or not math.isclose(prev_pct, 66.7, abs_tol=1e-1):
            errors.append(
                f"For 4 anemic out of 6 patients (6.5, 9.8, 10.6, 10.9 anemic; 11.0, 12.4 normal): "
                f"expected 66.7%, got {prev_pct!r}"
            )

    try:
        dist_counts = reg.count_by_district()
    except Exception as exc:
        errors.append(f"reg.count_by_district() raised {type(exc).__name__}: {exc}")
    else:
        expected_counts = {"Dhaka": 3, "Sylhet": 2, "Khulna": 1}
        if dist_counts != expected_counts:
            errors.append(f"count_by_district(): expected {expected_counts!r}, got {dist_counts!r}")

    if errors:
        return CheckResult(
            "ch12_exam_03",
            False,
            "failed",
            "Exam Task 12.3 — Check Type Guard, Prevalence Rounding & District Counts",
            "Reject non-MaternalPatient with TypeError, return 1-decimal prevalence (or None if empty), and return district frequency dict.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch12_exam_03",
        True,
        "passed",
        "Exam Task 12.3 Passed (35/35 pts) — Composed `EpidemicRegistry` Verified!",
        "Verified empty guards, strict `isinstance(..., MaternalPatient)` TypeError guard, `len(reg)`, 66.7% anemia prevalence, and district counts.",
        points_earned=35,
        points_possible=35,
        tip="Combining validated domain records (`MaternalPatient`) inside a typed container (`EpidemicRegistry`) is clean, production-grade OOP composition.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch12_exam_03",
        chapter="ch12",
        title="Exam 12.3: Composed EpidemicRegistry Container",
        book_section="Chapter 12 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Initialize `self.patients = []` in `__init__(self)`. In `add(self, patient)`, raise `TypeError` if `not isinstance(patient, MaternalPatient)`. Implement `__len__`, `anemia_prevalence_pct()` (1 decimal or `None`), and `count_by_district()` returning a dict.",
            2: "Blueprint:\n1. `add`: `if not isinstance(patient, MaternalPatient): raise TypeError(...)`; `self.patients.append(patient)`\n2. `__len__`: `return len(self.patients)`\n3. `anemia_prevalence_pct`: if empty return `None`; count `p.anemia_grade() != 'Normal'`; return `round((anemic / len(self.patients)) * 100, 1)`\n4. `count_by_district`: tally `counts[p.district] = counts.get(p.district, 0) + 1`.",
            3: "class EpidemicRegistry:\n    def __init__(self):\n        self.patients = []\n\n    def __len__(self):\n        return len(self.patients)\n\n    def add(self, patient):\n        if not isinstance(patient, MaternalPatient):\n            raise ___('Expected a MaternalPatient instance')\n        self.patients.append(patient)\n\n    def anemia_prevalence_pct(self):\n        if not self.patients:\n            return None\n        anemic = sum(1 for p in self.patients if p.anemia_grade() != 'Normal')\n        return round((anemic / len(self.patients)) * 100, 1)\n\n    def count_by_district(self):\n        counts = {}\n        for p in self.patients:\n            counts[p.district] = counts.get(p.district, 0) + 1\n        return counts",
        },
        verifier=_verify_exam_03,
    )
)
