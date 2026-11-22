"""Chapter 35 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch35_todo_01..06) and the Chapter 35 Coding Exam
(ch35_exam_01..03) for Publishing Reproducible Research — Model Cards,
Inference Contracts, Artifact Manifests, and TRIPOD+AI Checklists.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn/Joblib inside the check
function, runs all disk writes strictly inside tempfile.TemporaryDirectory(),
and lets NotImplementedError propagate cleanly for starter stubs.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise

_REQUIRED_PAYLOAD_KEYS = {
    "maternal_age_years",
    "anc_visits",
    "maternal_hb_g_dl",
    "residence",
    "wealth_quintile",
}
_VALID_RESIDENCE = {"Rural", "Urban"}
_VALID_WEALTH = {"Poorest", "Poorer", "Middle", "Richer", "Richest"}
_TRIPOD_ITEMS = [
    "calibration_reported",
    "code_released",
    "data_source_described",
    "leakage_prevented",
    "limitations_stated",
    "missing_data_handled",
    "seeds_fixed",
    "subgroup_fairness_audited",
]
_MODEL_CARD_HEADINGS = [
    "## Model Details",
    "## Intended Use",
    "## Training Data",
    "## Evaluation Metrics",
    "## Fairness & Limitations",
]


def _is_number(val: Any) -> bool:
    return (
        isinstance(val, (int, float))
        and not isinstance(val, bool)
        and math.isfinite(float(val))
    )


def _is_bool_like(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (bool, np.bool_))


def _check_callable(
    ex_id: str,
    title: str,
    fn: Any,
    expected_name: str,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    if not callable(fn):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Pass the Function/Class Name Without Parentheses",
            (
                f"Expected a callable (like {expected_name}), "
                f"but received {type(fn).__name__} ({fn!r})."
            ),
            details=[
                f"Pass the object itself: tutor.check('{ex_id}', {expected_name})",
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
        f"{title} — Function Returned None (print() vs. return Trap!)",
        f"Calling {call_str} returned None instead of the expected result.",
        details=[
            "Remember: print() only displays text; functions must end with `return ...`.",
            f"Ensure {call_str} explicitly returns the required dictionary or string.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Inference contracts and reproducibility helpers must return structured values for automated audits.",
    )


def _check_dict_keys(
    ex_id: str,
    title: str,
    got: Any,
    expected_keys: list[str],
    points_possible: int = 10,
) -> Optional[CheckResult]:
    if not isinstance(got, dict):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Expected a Dictionary",
            f"Expected a dict with keys {expected_keys}, got {type(got).__name__} ({got!r}).",
            points_earned=0,
            points_possible=points_possible,
        )
    missing = [k for k in expected_keys if k not in got]
    extra = [k for k in got if k not in expected_keys]
    if missing or extra:
        details = []
        if missing:
            details.append(f"Missing key(s): {missing}")
        if extra:
            details.append(f"Unexpected key(s): {extra}")
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Check Dictionary Keys",
            f"Expected exact keys {expected_keys}, got {list(got.keys())}.",
            details=details,
            points_earned=0,
            points_possible=points_possible,
        )
    return None


def _ref_validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dictionary.")
    missing = sorted(_REQUIRED_PAYLOAD_KEYS - set(payload.keys()))
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    age = payload["maternal_age_years"]
    anc = payload["anc_visits"]
    hb = payload["maternal_hb_g_dl"]
    res = payload["residence"]
    wealth = payload["wealth_quintile"]

    if not _is_number(age) or not (12 <= float(age) <= 60):
        raise ValueError(f"maternal_age_years out of bounds [12, 60]: {age!r}")
    if not _is_number(anc) or not (0 <= float(anc) <= 25):
        raise ValueError(f"anc_visits out of bounds [0, 25]: {anc!r}")
    if not _is_number(hb) or not (2.0 <= float(hb) <= 25.0):
        raise ValueError(f"maternal_hb_g_dl out of bounds [2.0, 25.0]: {hb!r}")
    if not isinstance(res, str):
        raise ValueError(f"residence must be a string, got {type(res).__name__}")
    res_clean = res.strip().title()
    if res_clean not in _VALID_RESIDENCE:
        raise ValueError(f"Invalid residence: {res!r}")
    if not isinstance(wealth, str):
        raise ValueError(f"wealth_quintile must be a string, got {type(wealth).__name__}")
    wealth_clean = wealth.strip().title()
    if wealth_clean not in _VALID_WEALTH:
        raise ValueError(f"Invalid wealth_quintile: {wealth!r}")

    return {
        "maternal_age_years": float(age),
        "anc_visits": float(anc),
        "maternal_hb_g_dl": float(hb),
        "residence": res_clean,
        "wealth_quintile": wealth_clean,
    }


def _make_ref_fitted_pipeline() -> Any:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from ..datasets import load_data

    df = load_data("mch_survey_sample", as_pandas=True).copy()
    df["residence"] = df["residence"].astype(str).str.strip().str.title()
    df["wealth_quintile"] = (
        df["wealth_quintile"].astype(str).str.strip().str.title()
    )
    num_cols = ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]
    cat_cols = ["residence", "wealth_quintile"]
    df[num_cols] = df[num_cols].replace(999, np.nan)
    X = df[num_cols + cat_cols]
    y = df["child_stunted"]
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=500))])
    pipe.fit(X_tr, y_tr)
    return pipe


class _RefClinicalService:
    """Lightweight deterministic reference service for exam verifiers."""

    def __init__(self, pipeline: Any, threshold: float = 0.35, model_version: str = "v1.0"):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.model_version = str(model_version)
        self.calls = 0

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        self.calls += 1
        clean = _ref_validate_payload(payload)
        row = pd.DataFrame([clean])
        prob = float(self.pipeline.predict_proba(row)[0, 1])
        label = int(prob >= self.threshold)
        if prob >= self.threshold:
            tier = "High"
        elif prob >= self.threshold - 0.15:
            tier = "Borderline"
        else:
            tier = "Low"
        return {
            "probability": round(prob, 4),
            "predicted_label": label,
            "risk_tier": tier,
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# GUIDED TO-DO 35.1: validate_clinical_payload(payload)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_todo_01", "To-Do 35.1", fn, "validate_clinical_payload", 10)
    if bad is not None:
        return bad

    raw_valid = {
        "maternal_age_years": 26,
        "anc_visits": 4,
        "maternal_hb_g_dl": 11.4,
        "residence": "  rural ",
        "wealth_quintile": " poorest ",
    }
    raw_snapshot = copy.deepcopy(raw_valid)

    try:
        clean = fn(raw_valid)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch35_todo_01",
            False,
            "failed",
            "To-Do 35.1 — Rejected a Valid Clinical Payload",
            f"validate_clinical_payload(valid_payload) raised {type(exc).__name__}: {exc}",
            details=[
                "Strip whitespace and title-case `residence` and `wealth_quintile` before checking membership.",
                "Ensure numeric bounds are inclusive: age [12, 60], anc_visits [0, 25], hb [2.0, 25.0].",
            ],
            points_earned=0,
            points_possible=10,
        )

    if clean is None:
        return _none_return_trap("ch35_todo_01", "To-Do 35.1", "validate_clinical_payload(payload)", 10)

    key_err = _check_dict_keys(
        "ch35_todo_01",
        "To-Do 35.1",
        clean,
        ["maternal_age_years", "anc_visits", "maternal_hb_g_dl", "residence", "wealth_quintile"],
        10,
    )
    if key_err is not None:
        return key_err

    errors: list[str] = []
    if raw_valid != raw_snapshot:
        errors.append("Caller's `payload` dict was mutated in place! Build and return a fresh normalized copy.")
    if clean is raw_valid:
        errors.append("Returned the exact same dict object instead of a clean copy.")
    if clean.get("residence") != "Rural":
        errors.append(f"Expected normalized residence 'Rural', got {clean.get('residence')!r}.")
    if clean.get("wealth_quintile") != "Poorest":
        errors.append(f"Expected normalized wealth_quintile 'Poorest', got {clean.get('wealth_quintile')!r}.")
    if not _is_number(clean.get("maternal_age_years")) or not math.isclose(float(clean["maternal_age_years"]), 26.0):
        errors.append(f"Unexpected maternal_age_years: {clean.get('maternal_age_years')!r}.")
    if not _is_number(clean.get("anc_visits")) or not math.isclose(float(clean["anc_visits"]), 4.0):
        errors.append(f"Unexpected anc_visits: {clean.get('anc_visits')!r}.")
    if not _is_number(clean.get("maternal_hb_g_dl")) or not math.isclose(float(clean["maternal_hb_g_dl"]), 11.4):
        errors.append(f"Unexpected maternal_hb_g_dl: {clean.get('maternal_hb_g_dl')!r}.")

    # Boundary acceptance test
    boundary_cases = [
        {
            "maternal_age_years": 12,
            "anc_visits": 0,
            "maternal_hb_g_dl": 2.0,
            "residence": "Urban",
            "wealth_quintile": "Richest",
        },
        {
            "maternal_age_years": 60,
            "anc_visits": 25,
            "maternal_hb_g_dl": 25.0,
            "residence": "RURAL",
            "wealth_quintile": "middle",
        },
    ]
    for b_case in boundary_cases:
        try:
            b_out = fn(b_case)
            if not isinstance(b_out, dict):
                errors.append(f"Boundary payload {b_case} did not return a dict.")
        except Exception as exc:
            errors.append(f"Inclusive boundary payload {b_case} should be accepted, but raised {type(exc).__name__}: {exc}")

    # Invalid cases MUST raise ValueError
    invalid_cases = [
        ("non-dict input", "not a dict"),
        (
            "missing required key 'residence'",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "wealth_quintile": "Middle",
            },
        ),
        (
            "age below 12",
            {
                "maternal_age_years": 10,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "age above 60",
            {
                "maternal_age_years": 65,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "sentinel anc_visits=999",
            {
                "maternal_age_years": 25,
                "anc_visits": 999,
                "maternal_hb_g_dl": 11.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "negative anc_visits=-1",
            {
                "maternal_age_years": 25,
                "anc_visits": -1,
                "maternal_hb_g_dl": 11.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "impossible hemoglobin 99.0 g/dL",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": 99.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "hemoglobin below 2.0",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": 1.2,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "boolean passed for numeric field",
            {
                "maternal_age_years": True,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "string passed for numeric field",
            {
                "maternal_age_years": "25",
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": "Rural",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "invalid residence category",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": "PeriUrban",
                "wealth_quintile": "Middle",
            },
        ),
        (
            "non-string residence",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": 1,
                "wealth_quintile": "Middle",
            },
        ),
        (
            "invalid wealth_quintile category",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": 11.0,
                "residence": "Urban",
                "wealth_quintile": "UltraRich",
            },
        ),
        (
            "NaN hemoglobin",
            {
                "maternal_age_years": 25,
                "anc_visits": 3,
                "maternal_hb_g_dl": float("nan"),
                "residence": "Urban",
                "wealth_quintile": "Middle",
            },
        ),
    ]

    for desc, bad_payload in invalid_cases:
        try:
            res = fn(bad_payload)
            errors.append(f"Failed to raise ValueError for {desc} (returned {res!r}).")
        except ValueError:
            pass
        except Exception as exc:
            errors.append(f"Expected ValueError for {desc}, but got {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch35_todo_01",
            False,
            "failed",
            "To-Do 35.1 — Check Payload Schema, Bounds & Non-Mutation",
            "Validate dict type, required keys, non-bool finite numeric ranges, and stripped/title-cased categories, raising ValueError on violations.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch35_todo_01",
        True,
        "passed",
        "To-Do 35.1 Passed — Clinical Payload Validation Contract Verified!",
        "Validated exact required schema, physiological ranges (rejecting sentinel 999, Hb=99.0, bools, and NaN), normalized string casing, and preserved caller immutability.",
        tip="In Python `isinstance(True, int)` is True — always reject `isinstance(val, bool)` before numeric range checks in clinical APIs!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_todo_01",
        chapter="ch35",
        title="To-Do 35.1: Clinical Payload Contract (validate_clinical_payload)",
        book_section="Section 35.3.1 (Strict Inference Input Contracts)",
        points=10,
        hints={
            1: "Check `isinstance(payload, dict)` and verify all 5 required keys exist. Check numeric fields are `(int, float)` AND `not isinstance(x, bool)` AND `math.isfinite(x)` within `[12, 60]`, `[0, 25]`, and `[2.0, 25.0]`.",
            2: "Check `isinstance(residence, str)` and `isinstance(wealth_quintile, str)`, normalize via `.strip().title()`, verify membership in `{'Rural', 'Urban'}` and `{'Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'}`, and raise `ValueError` on any violation.",
            3: "def validate_clinical_payload(payload):\n    if not isinstance(payload, dict):\n        raise ValueError('Payload must be a dict')\n    req = {'maternal_age_years', 'anc_visits', 'maternal_hb_g_dl', 'residence', 'wealth_quintile'}\n    if req - set(payload.keys()):\n        raise ValueError('Missing required keys')\n    # check numeric bounds (excluding bool) & strip/title strings...\n    return {'maternal_age_years': float(age), 'anc_visits': ___, ...}",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 35.2: ClinicalRiskService
# ---------------------------------------------------------------------------
def _verify_todo_02(cls: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    if not isinstance(cls, type):
        return CheckResult(
            "ch35_todo_02",
            False,
            "failed",
            "To-Do 35.2 — Pass the Class Name `ClinicalRiskService`",
            f"Expected a class (type), got {type(cls).__name__} ({cls!r}).",
            details=["Pass the class without parentheses: tutor.check('ch35_todo_02', ClinicalRiskService)"],
            points_earned=0,
            points_possible=10,
        )

    class _FixedProbPipeline:
        def __init__(self, prob: float):
            self.prob = prob

        def predict_proba(self, X: Any) -> Any:
            if not isinstance(X, pd.DataFrame) or len(X) != 1:
                raise ValueError("Expected a 1-row pandas DataFrame.")
            return np.array([[1.0 - self.prob, self.prob]])

    errors: list[str] = []
    valid_payload = {
        "maternal_age_years": 24,
        "anc_visits": 2,
        "maternal_hb_g_dl": 10.2,
        "residence": " rural ",
        "wealth_quintile": " poorest ",
    }
    snap = copy.deepcopy(valid_payload)

    # Test deterministic risk tiers & boundaries
    tier_cases = [
        (0.42156, 0.35, "v1.0", 0.4216, 1, "High"),
        (0.35, 0.35, "v1.0", 0.35, 1, "High"),
        (0.25444, 0.35, "v1.0", 0.2544, 0, "Borderline"),
        (0.20, 0.35, "v1.0", 0.2, 0, "Borderline"),
        (0.11234, 0.35, "v1.0", 0.1123, 0, "Low"),
        (0.55, 0.50, "v2.3", 0.55, 1, "High"),
    ]
    expected_keys = [
        "probability",
        "predicted_label",
        "risk_tier",
        "threshold",
        "model_version",
    ]

    for p_val, thr, ver, exp_prob, exp_lbl, exp_tier in tier_cases:
        try:
            svc = cls(_FixedProbPipeline(p_val), threshold=thr, model_version=ver)
            out = svc.predict_one(valid_payload)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch35_todo_02",
                False,
                "failed",
                "To-Do 35.2 — Runtime Error in `ClinicalRiskService`",
                f"ClinicalRiskService(pipe, threshold={thr}, model_version={ver!r}).predict_one(...) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if out is None:
            return _none_return_trap("ch35_todo_02", "To-Do 35.2", "service.predict_one(payload)", 10)

        key_err = _check_dict_keys("ch35_todo_02", "To-Do 35.2", out, expected_keys, 10)
        if key_err is not None:
            return key_err

        if not _is_number(out["probability"]) or not math.isclose(float(out["probability"]), exp_prob, abs_tol=1e-4):
            errors.append(f"[p={p_val}] probability: expected {exp_prob}, got {out['probability']!r}.")
        if not isinstance(out["predicted_label"], int) or isinstance(out["predicted_label"], bool) or out["predicted_label"] != exp_lbl:
            errors.append(f"[p={p_val}, thr={thr}] predicted_label: expected int {exp_lbl}, got {out['predicted_label']!r}.")
        if out["risk_tier"] != exp_tier:
            errors.append(f"[p={p_val}, thr={thr}] risk_tier: expected {exp_tier!r}, got {out['risk_tier']!r}.")
        if not _is_number(out["threshold"]) or not math.isclose(float(out["threshold"]), thr, abs_tol=1e-6):
            errors.append(f"threshold: expected {thr}, got {out['threshold']!r}.")
        if out["model_version"] != ver:
            errors.append(f"model_version: expected {ver!r}, got {out['model_version']!r}.")

    if valid_payload != snap:
        errors.append("Calling `predict_one(payload)` mutated the caller's payload dictionary!")

    # Test real fitted sklearn pipeline + default arguments
    try:
        real_pipe = _make_ref_fitted_pipeline()
        svc_default = cls(real_pipe)
        real_out = svc_default.predict_one(valid_payload)
        expected_p = round(
            float(real_pipe.predict_proba(pd.DataFrame([_ref_validate_payload(valid_payload)]))[0, 1]),
            4,
        )
        if not math.isclose(float(real_out["probability"]), expected_p, abs_tol=1e-4):
            errors.append(f"Real pipeline probability mismatch: expected {expected_p}, got {real_out['probability']!r}.")
        if not math.isclose(float(real_out["threshold"]), 0.35, abs_tol=1e-6):
            errors.append(f"Default threshold should be 0.35, got {real_out['threshold']!r}.")
        if real_out["model_version"] != "v1.0":
            errors.append(f"Default model_version should be 'v1.0', got {real_out['model_version']!r}.")
    except Exception as exc:
        errors.append(f"Failed on fitted scikit-learn Pipeline with defaults: {type(exc).__name__}: {exc}")

    # Invalid payload MUST raise ValueError
    try:
        svc_chk = cls(_FixedProbPipeline(0.2))
        bad_res = svc_chk.predict_one({**valid_payload, "maternal_hb_g_dl": 99.0})
        errors.append(f"Invalid hemoglobin (99.0) was NOT rejected by predict_one (returned {bad_res!r}).")
    except ValueError:
        pass
    except Exception as exc:
        errors.append(f"Expected ValueError on invalid payload, got {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch35_todo_02",
            False,
            "failed",
            "To-Do 35.2 — Check `ClinicalRiskService` Contract & Risk Tiers",
            "Validate via `validate_clinical_payload`, score a 1-row DataFrame with `self.pipeline.predict_proba`, assign High / Borderline / Low tiers, and return the 5-key contract dict.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch35_todo_02",
        True,
        "passed",
        "To-Do 35.2 Passed — `ClinicalRiskService` Inference Wrapper Verified!",
        "Verified input validation, 1-row DataFrame scoring on a real MCH pipeline, all 3 risk tiers (High / Borderline / Low), default arguments, and ValueError rejection.",
        tip="Encapsulating validation + threshold + model_version in one service class makes FastAPI, CLI, and batch scoring share the exact same clinical contract.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_todo_02",
        chapter="ch35",
        title="To-Do 35.2: Clinical Inference Service (ClinicalRiskService)",
        book_section="Section 35.3.2 (Serving Wrapper & FastAPI Pattern)",
        points=10,
        hints={
            1: "In `__init__(self, pipeline, threshold=0.35, model_version='v1.0')`, store `self.pipeline`, `self.threshold = float(threshold)`, and `self.model_version = str(model_version)`.",
            2: "In `predict_one(self, payload)`, call `clean = validate_clinical_payload(payload)`, build `row = pd.DataFrame([clean])`, extract `prob = float(self.pipeline.predict_proba(row)[0, 1])`, and map `prob >= self.threshold` -> `'High'`, `prob >= self.threshold - 0.15` -> `'Borderline'`, else `'Low'`.",
            3: "class ClinicalRiskService:\n    def __init__(self, pipeline, threshold=0.35, model_version='v1.0'):\n        self.pipeline = pipeline\n        self.threshold = float(threshold)\n        self.model_version = str(model_version)\n    def predict_one(self, payload):\n        clean = validate_clinical_payload(payload)\n        prob = float(self.pipeline.predict_proba(pd.DataFrame([clean]))[0, 1])\n        label = int(prob >= self.threshold)\n        tier = 'High' if prob >= self.threshold else ('Borderline' if prob >= self.threshold - 0.15 else 'Low')\n        return {'probability': round(prob, 4), 'predicted_label': label, 'risk_tier': tier, 'threshold': self.threshold, 'model_version': self.model_version}",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 35.3: generate_model_card_markdown(meta)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_todo_03", "To-Do 35.3", fn, "generate_model_card_markdown", 10)
    if bad is not None:
        return bad

    meta = {
        "name": "MCH Child Stunting Risk Model",
        "version": "v1.2.0",
        "intended_use": "Community health worker screening triage in rural Bangladesh.",
        "dataset": "mch_survey_sample (n=500 households, 50 DHS clusters)",
        "roc_auc": 0.7812,
        "brier_score": 0.1014,
        "limitations": "Cross-sectional survey; requires prospective clinic validation before autonomous deployment.",
    }
    meta_snap = copy.deepcopy(meta)

    try:
        md = fn(meta)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch35_todo_03",
            False,
            "failed",
            "To-Do 35.3 — Runtime Error in `generate_model_card_markdown`",
            f"generate_model_card_markdown(meta) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if md is None:
        return _none_return_trap("ch35_todo_03", "To-Do 35.3", "generate_model_card_markdown(meta)", 10)

    if not isinstance(md, str) or not md.strip():
        return CheckResult(
            "ch35_todo_03",
            False,
            "failed",
            "To-Do 35.3 — Must Return a Non-Empty Markdown String",
            f"Expected a non-empty str, got {type(md).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []
    if meta != meta_snap:
        errors.append("Input `meta` dict was mutated in place.")

    for heading in _MODEL_CARD_HEADINGS:
        if heading not in md:
            errors.append(f"Missing required Markdown heading: `{heading}`")

    for field in ("name", "version", "intended_use", "dataset", "limitations"):
        if str(meta[field]) not in md:
            errors.append(f"Rendered Model Card is missing `meta[{field!r}]` value: {meta[field]!r}")

    if "0.7812" not in md and "0.781" not in md:
        errors.append("Rendered Model Card is missing formatted `roc_auc` (0.7812).")
    if "0.1014" not in md and "0.101" not in md:
        errors.append("Rendered Model Card is missing formatted `brier_score` (0.1014).")

    if errors:
        return CheckResult(
            "ch35_todo_03",
            False,
            "failed",
            "To-Do 35.3 — Check Model Card Headings & Populated Metadata",
            "Include all 5 exact Markdown headings (`## Model Details`, `## Intended Use`, `## Training Data`, `## Evaluation Metrics`, `## Fairness & Limitations`) populated with every field from `meta`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch35_todo_03",
        True,
        "passed",
        "To-Do 35.3 Passed — Structured Clinical Model Card Generator Verified!",
        "Rendered all 5 Mitchell et al. Model Card sections populated cleanly with model metadata, discrimination (ROC-AUC), calibration (Brier), and clinical limitations.",
        tip="Generating MODEL_CARD.md programmatically directly from training metrics eliminates copy-paste drift between your code and your manuscript!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_todo_03",
        chapter="ch35",
        title="To-Do 35.3: Programmatic Model Card (generate_model_card_markdown)",
        book_section="Section 35.3.3 (Model Cards for Transparent Clinical Reporting)",
        points=10,
        hints={
            1: "Return a multi-line f-string containing the 5 exact level-2 headings: `## Model Details`, `## Intended Use`, `## Training Data`, `## Evaluation Metrics`, and `## Fairness & Limitations`.",
            2: "Populate each section with `meta['name']`, `meta['version']`, `meta['intended_use']`, `meta['dataset']`, `f\"{meta['roc_auc']:.4f}\"`, `f\"{meta['brier_score']:.4f}\"`, and `meta['limitations']`.",
            3: "def generate_model_card_markdown(meta):\n    return (\n        f\"# Model Card: {meta['name']}\\n\\n\"\n        f\"## Model Details\\n- Name: {meta['name']}\\n- Version: {meta['version']}\\n\\n\"\n        f\"## Intended Use\\n{meta['intended_use']}\\n\\n\"\n        f\"## Training Data\\n{meta['dataset']}\\n\\n\"\n        f\"## Evaluation Metrics\\n- ROC-AUC: {float(meta['roc_auc']):.4f}\\n- Brier Score: {float(meta['brier_score']):.4f}\\n\\n\"\n        f\"## Fairness & Limitations\\n{meta['limitations']}\\n\"\n    )",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 35.4: build_reproducibility_bundle(artifact_path, config_dict, out_dir)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_todo_04", "To-Do 35.4", fn, "build_reproducibility_bundle", 10)
    if bad is not None:
        return bad

    expected_keys = [
        "artifact_name",
        "artifact_bytes",
        "artifact_sha256",
        "python_version",
        "created_utc",
    ]
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        artifact = tmp / "stunting_model.joblib"
        payload_bytes = b"reproducible-clinical-pipeline-v1-bytes"
        artifact.write_bytes(payload_bytes)
        expected_sha = hashlib.sha256(payload_bytes).hexdigest()

        out_dir = tmp / "nested_release" / "bundle_v1"
        cfg = {"seed": 42, "threshold": 0.35, "model": "LogisticRegression"}

        try:
            manifest = fn(artifact, cfg, out_dir)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch35_todo_04",
                False,
                "failed",
                "To-Do 35.4 — Runtime Error in `build_reproducibility_bundle`",
                f"build_reproducibility_bundle(...) raised {type(exc).__name__}: {exc}",
                details=["Create `out_dir` with `Path(out_dir).mkdir(parents=True, exist_ok=True)` before writing JSON files."],
                points_earned=0,
                points_possible=10,
            )

        if manifest is None:
            return _none_return_trap("ch35_todo_04", "To-Do 35.4", "build_reproducibility_bundle(...)", 10)

        key_err = _check_dict_keys("ch35_todo_04", "To-Do 35.4", manifest, expected_keys, 10)
        if key_err is not None:
            return key_err

        if manifest["artifact_name"] != "stunting_model.joblib":
            errors.append(f"artifact_name: expected 'stunting_model.joblib', got {manifest['artifact_name']!r}.")
        if not isinstance(manifest["artifact_bytes"], int) or isinstance(manifest["artifact_bytes"], bool) or manifest["artifact_bytes"] != len(payload_bytes):
            errors.append(f"artifact_bytes: expected int {len(payload_bytes)}, got {manifest['artifact_bytes']!r}.")
        sha = manifest["artifact_sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower() or sha != expected_sha:
            errors.append(f"artifact_sha256 mismatch: expected lowercase 64-char {expected_sha!r}, got {sha!r}.")
        if not isinstance(manifest["python_version"], str) or not manifest["python_version"].strip():
            errors.append(f"python_version must be a non-empty string, got {manifest['python_version']!r}.")
        if not isinstance(manifest["created_utc"], str) or not manifest["created_utc"].strip():
            errors.append(f"created_utc must be a non-empty UTC timestamp string, got {manifest['created_utc']!r}.")

        cfg_path = out_dir / "config.json"
        man_path = out_dir / "manifest.json"
        if not cfg_path.exists():
            errors.append("`config.json` was not written inside `out_dir`.")
        else:
            try:
                loaded_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                if loaded_cfg != cfg:
                    errors.append(f"Saved config.json ({loaded_cfg}) did not match input config_dict ({cfg}).")
            except Exception as exc:
                errors.append(f"Saved config.json could not be parsed as JSON: {exc}")

        if not man_path.exists():
            errors.append("`manifest.json` was not written inside `out_dir`.")
        else:
            try:
                loaded_man = json.loads(man_path.read_text(encoding="utf-8"))
                if loaded_man != manifest:
                    errors.append("Saved manifest.json did not match the returned manifest dictionary.")
            except Exception as exc:
                errors.append(f"Saved manifest.json could not be parsed as JSON: {exc}")

    if errors:
        return CheckResult(
            "ch35_todo_04",
            False,
            "failed",
            "To-Do 35.4 — Check SHA-256 Digest, JSON Files & Manifest Schema",
            "Create `out_dir`, compute lowercase 64-char SHA-256 of `artifact_path`, write `config.json` and `manifest.json`, and return the manifest dict.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch35_todo_04",
        True,
        "passed",
        "To-Do 35.4 Passed — Cryptographic Reproducibility Bundle Verified!",
        "Created target directory, computed exact 64-char SHA-256 checksum, serialized config.json & manifest.json, and returned matching manifest metadata inside a clean temp directory.",
        tip="SHA-256 checksums in `manifest.json` prove bit-for-bit that the model artifact audited by peer reviewers is identical to the file deployed in clinic.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_todo_04",
        chapter="ch35",
        title="To-Do 35.4: SHA-256 Artifact Bundle (build_reproducibility_bundle)",
        book_section="Section 35.3.4 (Cryptographic Artifact Manifests)",
        points=10,
        hints={
            1: "Convert `artifact_path` and `out_dir` to `Path`, call `out.mkdir(parents=True, exist_ok=True)`, and compute `hashlib.sha256(artifact.read_bytes()).hexdigest()`.",
            2: "Build `manifest = {'artifact_name': artifact.name, 'artifact_bytes': int(artifact.stat().st_size), 'artifact_sha256': sha256_hex, 'python_version': platform.python_version(), 'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`.",
            3: "def build_reproducibility_bundle(artifact_path, config_dict, out_dir):\n    artifact, out = Path(artifact_path), Path(out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    sha = hashlib.sha256(artifact.read_bytes()).hexdigest()\n    manifest = {'artifact_name': artifact.name, 'artifact_bytes': int(artifact.stat().st_size), 'artifact_sha256': sha, 'python_version': platform.python_version(), 'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n    (out / 'config.json').write_text(json.dumps(config_dict, indent=2), encoding='utf-8')\n    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')\n    return manifest",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 35.5: verify_reproducibility_bundle(out_dir, artifact_path)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_todo_05", "To-Do 35.5", fn, "verify_reproducibility_bundle", 10)
    if bad is not None:
        return bad

    expected_keys = ["hash_match", "config_present", "bundle_valid"]
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        artifact = tmp / "model.joblib"
        orig_bytes = b"verified-model-weights-001"
        artifact.write_bytes(orig_bytes)
        sha = hashlib.sha256(orig_bytes).hexdigest()

        manifest = {
            "artifact_name": artifact.name,
            "artifact_bytes": len(orig_bytes),
            "artifact_sha256": sha,
            "python_version": platform.python_version(),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (tmp / "config.json").write_text(json.dumps({"seed": 42}, indent=2), encoding="utf-8")

        # Case 1: Valid bundle
        try:
            res_ok = fn(tmp, artifact)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch35_todo_05",
                False,
                "failed",
                "To-Do 35.5 — Runtime Error on Valid Bundle",
                f"verify_reproducibility_bundle(out_dir, artifact_path) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if res_ok is None:
            return _none_return_trap("ch35_todo_05", "To-Do 35.5", "verify_reproducibility_bundle(out_dir, artifact_path)", 10)

        key_err = _check_dict_keys("ch35_todo_05", "To-Do 35.5", res_ok, expected_keys, 10)
        if key_err is not None:
            return key_err

        if not (res_ok["hash_match"] is True and res_ok["config_present"] is True and res_ok["bundle_valid"] is True):
            errors.append(f"[Valid bundle] Expected all True, got {res_ok!r}.")

        # Case 2: Tampered artifact (1-byte drift)
        artifact.write_bytes(b"verified-model-weights-002")
        try:
            res_tamper = fn(tmp, artifact)
            if not (res_tamper["hash_match"] is False and res_tamper["config_present"] is True and res_tamper["bundle_valid"] is False):
                errors.append(
                    f"[Tampered artifact] Expected hash_match=False, config_present=True, bundle_valid=False; got {res_tamper!r}."
                )
        except Exception as exc:
            errors.append(f"[Tampered artifact] Raised unexpected {type(exc).__name__}: {exc}")

        # Case 3: Restore artifact, delete config.json
        artifact.write_bytes(orig_bytes)
        (tmp / "config.json").unlink()
        try:
            res_nocfg = fn(tmp, artifact)
            if not (res_nocfg["hash_match"] is True and res_nocfg["config_present"] is False and res_nocfg["bundle_valid"] is False):
                errors.append(
                    f"[Missing config.json] Expected hash_match=True, config_present=False, bundle_valid=False; got {res_nocfg!r}."
                )
        except Exception as exc:
            errors.append(f"[Missing config.json] Raised unexpected {type(exc).__name__}: {exc}")

        # Case 4: Missing manifest.json should raise FileNotFoundError
        (tmp / "manifest.json").unlink()
        try:
            res_noman = fn(tmp, artifact)
            errors.append(f"[Missing manifest.json] Expected FileNotFoundError, but returned {res_noman!r}.")
        except FileNotFoundError:
            pass
        except Exception as exc:
            errors.append(f"[Missing manifest.json] Expected FileNotFoundError, got {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch35_todo_05",
            False,
            "failed",
            "To-Do 35.5 — Check Integrity Verification & Tamper Detection",
            "Load `manifest.json`, check `(out_dir / 'config.json').exists()`, recompute SHA-256 of `artifact_path`, and return `{'hash_match': bool, 'config_present': bool, 'bundle_valid': bool}`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch35_todo_05",
        True,
        "passed",
        "To-Do 35.5 Passed — Bundle Integrity & Tamper Detection Verified!",
        "Passed all 4 integrity scenarios: valid bundle, 1-byte tampered artifact detection, missing config.json detection, and FileNotFoundError on absent manifest.json.",
        tip="Running `verify_reproducibility_bundle` on server startup prevents silent deployment of corrupted or overwritten model weights.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_todo_05",
        chapter="ch35",
        title="To-Do 35.5: Bundle Integrity Check (verify_reproducibility_bundle)",
        book_section="Section 35.3.4 (Cryptographic Artifact Manifests)",
        points=10,
        hints={
            1: "Read `manifest = json.loads((Path(out_dir) / 'manifest.json').read_text(encoding='utf-8'))` and check `config_present = (Path(out_dir) / 'config.json').exists()`.",
            2: "Recompute `actual_sha = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()` and compare `hash_match = (actual_sha == manifest['artifact_sha256'])`.",
            3: "def verify_reproducibility_bundle(out_dir, artifact_path):\n    out = Path(out_dir)\n    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))\n    actual_sha = hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()\n    hash_match = actual_sha == manifest['artifact_sha256']\n    config_present = (out / 'config.json').exists()\n    return {'hash_match': bool(hash_match), 'config_present': bool(config_present), 'bundle_valid': bool(hash_match and config_present)}",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 35.6: score_tripod_ai_checklist(checklist_dict)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_todo_06", "To-Do 35.6", fn, "score_tripod_ai_checklist", 10)
    if bad is not None:
        return bad

    expected_keys = ["score", "max_score", "pct", "missing_items"]
    errors: list[str] = []

    cases = [
        ("all 8 complete", {k: True for k in _TRIPOD_ITEMS}, 8, 100.0, []),
        (
            "5 complete + 2 False + 1 omitted + extra key",
            {
                "calibration_reported": True,
                "code_released": False,
                "data_source_described": True,
                "leakage_prevented": True,
                "limitations_stated": True,
                "missing_data_handled": True,
                "seeds_fixed": False,
                "bonus_extra_item": True,
            },
            5,
            62.5,
            ["code_released", "seeds_fixed", "subgroup_fairness_audited"],
        ),
        ("empty dict", {}, 0, 0.0, list(_TRIPOD_ITEMS)),
    ]

    for label, inp, exp_score, exp_pct, exp_missing in cases:
        inp_copy = copy.deepcopy(inp)
        try:
            got = fn(inp_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch35_todo_06",
                False,
                "failed",
                "To-Do 35.6 — Runtime Error in `score_tripod_ai_checklist`",
                f"score_tripod_ai_checklist(...) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch35_todo_06", "To-Do 35.6", "score_tripod_ai_checklist(checklist_dict)", 10)

        key_err = _check_dict_keys("ch35_todo_06", "To-Do 35.6", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if inp_copy != inp:
            errors.append(f"[{label}] Input checklist dict was mutated in place.")
        if not isinstance(got["score"], int) or isinstance(got["score"], bool) or got["score"] != exp_score:
            errors.append(f"[{label}] score: expected int {exp_score}, got {got['score']!r}.")
        if not isinstance(got["max_score"], int) or isinstance(got["max_score"], bool) or got["max_score"] != 8:
            errors.append(f"[{label}] max_score: expected int 8, got {got['max_score']!r}.")
        if not _is_number(got["pct"]) or not math.isclose(float(got["pct"]), exp_pct, abs_tol=1e-2):
            errors.append(f"[{label}] pct: expected {exp_pct}, got {got['pct']!r}.")
        if got["missing_items"] != exp_missing:
            errors.append(f"[{label}] missing_items: expected {exp_missing}, got {got['missing_items']!r}.")

    if errors:
        return CheckResult(
            "ch35_todo_06",
            False,
            "failed",
            "To-Do 35.6 — Check 8 Mandatory TRIPOD+AI Checklist Items & Scoring",
            "Audit the 8 exact mandatory TRIPOD+AI keys, ignore extra keys, compute `pct = round(100.0 * score / 8, 1)`, and return sorted `missing_items`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch35_todo_06",
        True,
        "passed",
        "To-Do 35.6 Passed — TRIPOD+AI Reporting Readiness Auditor Verified!",
        "Audited complete, partial (with extra keys ignored), and empty checklists with exact 8-point scoring and sorted missing-item diagnostics.",
        tip="Running an automated TRIPOD+AI check before manuscript submission catches missing calibration plots, unfixed seeds, or unaudited subgroups before peer review!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_todo_06",
        chapter="ch35",
        title="To-Do 35.6: TRIPOD+AI Compliance Scorer (score_tripod_ai_checklist)",
        book_section="Section 35.3.6 (TRIPOD+AI Manuscript Readiness Audit)",
        points=10,
        hints={
            1: "Define the exact list of 8 mandatory items and check `bool(checklist_dict.get(item, False))` for each one so omitted keys count as missing and extra keys are ignored.",
            2: "Collect `missing = sorted(k for k in mandatory if not bool(checklist_dict.get(k, False)))`, set `score = len(mandatory) - len(missing)`, and compute `pct = round(100.0 * score / 8, 1)`.",
            3: "def score_tripod_ai_checklist(checklist_dict):\n    items = ['calibration_reported', 'code_released', 'data_source_described', 'leakage_prevented', 'limitations_stated', 'missing_data_handled', 'seeds_fixed', 'subgroup_fairness_audited']\n    missing = sorted(k for k in items if not bool(checklist_dict.get(k, False)))\n    score = len(items) - len(missing)\n    return {'score': int(score), 'max_score': 8, 'pct': round(100.0 * score / 8, 1), 'missing_items': missing}",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 35.1 (30 pts): batch_inference_with_quarantine(service, payloads)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_exam_01", "Exam Task 35.1", fn, "batch_inference_with_quarantine", 30)
    if bad is not None:
        return bad

    pipe = _make_ref_fitted_pipeline()
    svc = _RefClinicalService(pipe, threshold=0.35, model_version="v1.0")
    payloads = [
        {"maternal_age_years": 25, "anc_visits": 4, "maternal_hb_g_dl": 11.5, "residence": "Rural", "wealth_quintile": "Middle"},
        {"maternal_age_years": 25, "anc_visits": 4, "maternal_hb_g_dl": 99.0, "residence": "Rural", "wealth_quintile": "Middle"},
        {"maternal_age_years": 31, "anc_visits": 6, "maternal_hb_g_dl": 12.6, "residence": " urban ", "wealth_quintile": " richest "},
        {"maternal_age_years": 22, "anc_visits": 2, "maternal_hb_g_dl": 10.1, "wealth_quintile": "Poorest"},
        {"maternal_age_years": 19, "anc_visits": 1, "maternal_hb_g_dl": 9.0, "residence": "Rural", "wealth_quintile": "Poorest"},
    ]
    snap = copy.deepcopy(payloads)
    expected_keys = ["succeeded", "quarantined", "success_rate"]
    errors: list[str] = []

    try:
        got = fn(svc, payloads)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch35_exam_01",
            False,
            "failed",
            "Exam Task 35.1 — Crashed Instead of Quarantining Invalid Payloads",
            f"batch_inference_with_quarantine(service, payloads) raised {type(exc).__name__}: {exc}",
            details=["Wrap `service.predict_one(p)` inside `try: ... except ValueError as exc:` so bad rows go to `quarantined`."],
            points_earned=0,
            points_possible=30,
        )

    if got is None:
        return _none_return_trap("ch35_exam_01", "Exam Task 35.1", "batch_inference_with_quarantine(service, payloads)", 30)

    key_err = _check_dict_keys("ch35_exam_01", "Exam Task 35.1", got, expected_keys, 30)
    if key_err is not None:
        return key_err

    if payloads != snap:
        errors.append("Input `payloads` list was mutated in place.")

    if not isinstance(got["succeeded"], list) or len(got["succeeded"]) != 3:
        errors.append(f"Expected 3 succeeded predictions, got {len(got['succeeded']) if isinstance(got['succeeded'], list) else got['succeeded']!r}.")
    elif not all(isinstance(r, dict) and "probability" in r and "risk_tier" in r for r in got["succeeded"]):
        errors.append("Each element in `succeeded` must be the prediction dict returned by `service.predict_one(p)`.")

    if not isinstance(got["quarantined"], list) or len(got["quarantined"]) != 2:
        errors.append(f"Expected 2 quarantined entries (indices 1 and 3), got {got['quarantined']!r}.")
    else:
        q_indices = [q.get("index") for q in got["quarantined"] if isinstance(q, dict)]
        if q_indices != [1, 3]:
            errors.append(f"Expected quarantined indices [1, 3], got {q_indices!r}.")
        for q in got["quarantined"]:
            if not isinstance(q, dict) or set(q.keys()) != {"index", "error"}:
                errors.append(f"Each quarantined record must have exact keys {{'index', 'error'}}, got {q!r}.")
            elif not isinstance(q.get("error"), str) or not q["error"].strip():
                errors.append(f"Quarantined entry {q!r} must include a non-empty error string.")

    if not _is_number(got["success_rate"]) or not math.isclose(float(got["success_rate"]), 0.6, abs_tol=1e-4):
        errors.append(f"Expected success_rate 0.6 (3/5), got {got['success_rate']!r}.")

    # Empty batch check
    try:
        empty_res = fn(svc, [])
        if empty_res != {"succeeded": [], "quarantined": [], "success_rate": 0.0}:
            errors.append(f"Empty batch should return {{'succeeded': [], 'quarantined': [], 'success_rate': 0.0}}, got {empty_res!r}.")
    except ZeroDivisionError:
        errors.append("Empty batch `[]` raised ZeroDivisionError! Guard `len(payloads) == 0` -> `success_rate = 0.0`.")
    except Exception as exc:
        errors.append(f"Empty batch raised unexpected {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch35_exam_01",
            False,
            "failed",
            "Exam Task 35.1 — Check Batch Quarantine Loop & Success Rate",
            "Iterate `enumerate(payloads)`, score valid rows into `succeeded`, quarantine `ValueError` rows as `{'index': idx, 'error': str(exc)}`, and handle empty batches cleanly.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch35_exam_01",
        True,
        "passed",
        "Exam Task 35.1 Passed (30/30 pts) — Resilient Batch Inference With Quarantine Verified!",
        "Scored 3 valid clinical payloads, quarantined 2 malformed payloads (indices 1 & 3) with diagnostic messages, computed success_rate=0.6, and guarded empty batches.",
        points_earned=30,
        points_possible=30,
        tip="Dead-letter quarantine queues keep hospital nightly batch jobs running even when 1% of clinic intake rows contain typos or missing fields.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_exam_01",
        chapter="ch35",
        title="Exam 35.1: Resilient Batch Scoring With Quarantine (batch_inference_with_quarantine)",
        book_section="Chapter 35 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Initialize `succeeded, quarantined = [], []` and loop `for idx, payload in enumerate(payloads):` calling `service.predict_one(payload)` inside `try / except ValueError as exc:`.",
            2: "On success append the prediction dict to `succeeded`; on `ValueError` append `{'index': idx, 'error': str(exc)}` to `quarantined`.",
            3: "def batch_inference_with_quarantine(service, payloads):\n    succeeded, quarantined = [], []\n    for idx, p in enumerate(payloads):\n        try:\n            succeeded.append(service.predict_one(p))\n        except ValueError as exc:\n            quarantined.append({'index': idx, 'error': str(exc)})\n    rate = round(len(succeeded) / len(payloads), 4) if payloads else 0.0\n    return {'succeeded': succeeded, 'quarantined': quarantined, 'success_rate': rate}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 35.2 (35 pts): benchmark_inference_latency(service, valid_payload, n_runs=50)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch35_exam_02", "Exam Task 35.2", fn, "benchmark_inference_latency", 35)
    if bad is not None:
        return bad

    pipe = _make_ref_fitted_pipeline()
    svc = _RefClinicalService(pipe, threshold=0.35, model_version="v1.0")
    valid_payload = {
        "maternal_age_years": 27,
        "anc_visits": 4,
        "maternal_hb_g_dl": 11.2,
        "residence": "Rural",
        "wealth_quintile": "Middle",
    }
    expected_keys = ["n_runs", "mean_ms", "p95_ms", "max_ms", "throughput_per_sec"]
    errors: list[str] = []

    try:
        got = fn(svc, valid_payload, n_runs=25)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch35_exam_02",
            False,
            "failed",
            "Exam Task 35.2 — Runtime Error in `benchmark_inference_latency`",
            f"benchmark_inference_latency(service, valid_payload, n_runs=25) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap("ch35_exam_02", "Exam Task 35.2", "benchmark_inference_latency(service, payload)", 35)

    key_err = _check_dict_keys("ch35_exam_02", "Exam Task 35.2", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    if svc.calls != 25:
        errors.append(f"Expected service.predict_one to be called exactly 25 times, got {svc.calls}.")
    if not isinstance(got["n_runs"], int) or isinstance(got["n_runs"], bool) or got["n_runs"] != 25:
        errors.append(f"n_runs: expected int 25, got {got['n_runs']!r}.")

    for k in ("mean_ms", "p95_ms", "max_ms", "throughput_per_sec"):
        if not _is_number(got[k]) or float(got[k]) <= 0:
            errors.append(f"{k} must be a positive finite number, got {got[k]!r}.")

    if not errors:
        mean_ms = float(got["mean_ms"])
        p95_ms = float(got["p95_ms"])
        max_ms = float(got["max_ms"])
        tps = float(got["throughput_per_sec"])

        if max_ms + 1e-4 < p95_ms:
            errors.append(f"max_ms ({max_ms}) cannot be smaller than p95_ms ({p95_ms}).")
        if max_ms + 1e-4 < mean_ms:
            errors.append(f"max_ms ({max_ms}) cannot be smaller than mean_ms ({mean_ms}).")
        expected_tps = 1000.0 / mean_ms
        if not math.isclose(tps, expected_tps, rel_tol=0.05, abs_tol=1.0):
            errors.append(
                f"throughput_per_sec ({tps}) is inconsistent with 1000.0 / mean_ms ({expected_tps:.2f})."
            )

    # Verify default n_runs=50 with a fast counting stub
    class _FastStubService:
        def __init__(self) -> None:
            self.calls = 0

        def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            time.sleep(0.0002)
            return {"probability": 0.2, "predicted_label": 0, "risk_tier": "Borderline"}

    stub_svc = _FastStubService()
    try:
        got_def = fn(stub_svc, valid_payload)
        if stub_svc.calls != 50 or got_def.get("n_runs") != 50:
            errors.append(f"Default `n_runs` should be 50 (observed {stub_svc.calls} calls and n_runs={got_def.get('n_runs')!r}).")
    except Exception as exc:
        errors.append(f"Calling with default n_runs=50 raised {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch35_exam_02",
            False,
            "failed",
            "Exam Task 35.2 — Check Latency Timing, Percentiles & Throughput",
            "Time `n_runs` calls to `service.predict_one(valid_payload)` via `time.perf_counter()`, compute `mean_ms`, `p95_ms`, `max_ms`, and `throughput_per_sec = 1000.0 / mean_ms`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch35_exam_02",
        True,
        "passed",
        "Exam Task 35.2 Passed (35/35 pts) — Inference Latency SLA Benchmark Verified!",
        f"Benchmarked ClinicalRiskService (mean={got['mean_ms']} ms, p95={got['p95_ms']} ms, throughput={got['throughput_per_sec']} req/s) and verified default n_runs=50.",
        points_earned=35,
        points_possible=35,
        tip="Always report p95 latency alongside mean latency in clinical deployment specs — tail spikes, not averages, break bedside alert SLAs.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_exam_02",
        chapter="ch35",
        title="Exam 35.2: Inference Latency & p95 SLA Benchmark (benchmark_inference_latency)",
        book_section="Chapter 35 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Loop `n_runs` times: capture `t0 = time.perf_counter()`, call `service.predict_one(valid_payload)`, and append `(time.perf_counter() - t0) * 1000.0` to `times_ms`.",
            2: "Use `mean_ms = float(np.mean(times_ms))`, `p95_ms = float(np.percentile(times_ms, 95))`, `max_ms = float(np.max(times_ms))`, and `tps = 1000.0 / mean_ms`.",
            3: "def benchmark_inference_latency(service, valid_payload, n_runs=50):\n    times_ms = []\n    for _ in range(n_runs):\n        t0 = time.perf_counter()\n        service.predict_one(valid_payload)\n        times_ms.append((time.perf_counter() - t0) * 1000.0)\n    mean_ms = float(np.mean(times_ms))\n    return {'n_runs': int(n_runs), 'mean_ms': round(mean_ms, 4), 'p95_ms': round(float(np.percentile(times_ms, 95)), 4), 'max_ms': round(float(np.max(times_ms)), 4), 'throughput_per_sec': round(1000.0 / max(mean_ms, 1e-9), 2)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 35.3 (35 pts): package_end_to_end_research_release(mch_df, out_dir, seed=42)
# ---------------------------------------------------------------------------
def _ref_expected_auc(mch_df: Any, seed: int = 42) -> float:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["wealth_quintile"] = (
        work["wealth_quintile"].astype(str).str.strip().str.title()
    )
    num_cols = ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]
    cat_cols = ["residence", "wealth_quintile"]
    work[num_cols] = work[num_cols].replace(999, np.nan)
    X = work[num_cols + cat_cols]
    y = work["child_stunted"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    pipe = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=500))])
    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)[:, 1]
    return round(float(roc_auc_score(y_te, probs)), 4)


def _verify_exam_03(fn: Any) -> CheckResult:
    import joblib
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable(
        "ch35_exam_03", "Exam Task 35.3", fn, "package_end_to_end_research_release", 35
    )
    if bad is not None:
        return bad

    mch_raw = load_data("mch_survey_sample", as_pandas=True)
    mch_copy = mch_raw.copy(deep=True)
    expected_keys = ["bundle_valid", "test_auc", "files_created"]
    expected_files = ["MODEL_CARD.md", "config.json", "manifest.json", "model.joblib"]
    exp_auc = _ref_expected_auc(mch_raw, seed=42)
    errors: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "paper_release_v1"
        try:
            got = fn(mch_copy, out_dir, seed=42)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch35_exam_03",
                False,
                "failed",
                "Exam Task 35.3 — Runtime Error in `package_end_to_end_research_release`",
                f"package_end_to_end_research_release(mch_df, out_dir, seed=42) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch35_exam_03",
                "Exam Task 35.3",
                "package_end_to_end_research_release(mch_df, out_dir, seed=42)",
                35,
            )

        key_err = _check_dict_keys("ch35_exam_03", "Exam Task 35.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not mch_copy.equals(mch_raw):
            errors.append("Input `mch_df` was mutated in place! Always start with `work = mch_df.copy()`.")

        if not _is_bool_like(got["bundle_valid"]) or not bool(got["bundle_valid"]):
            errors.append(f"bundle_valid must be True, got {got['bundle_valid']!r}.")

        if not _is_number(got["test_auc"]) or not math.isclose(float(got["test_auc"]), exp_auc, abs_tol=1e-3):
            errors.append(
                f"test_auc: expected ~{exp_auc} (stratified 75/25 split, seed=42, standard Pipeline + LogisticRegression), got {got['test_auc']!r}."
            )

        if got["files_created"] != expected_files:
            errors.append(f"files_created: expected exact sorted list {expected_files}, got {got['files_created']!r}.")

        for fname in expected_files:
            if not (out_dir / fname).exists():
                errors.append(f"Expected release artifact `{fname}` missing in `out_dir`.")

        # Verify model.joblib reloads cleanly & predicts
        model_path = out_dir / "model.joblib"
        if model_path.exists():
            try:
                reloaded = joblib.load(model_path)
                sample = pd.DataFrame(
                    [
                        {
                            "maternal_age_years": 25.0,
                            "anc_visits": 4.0,
                            "maternal_hb_g_dl": 11.2,
                            "residence": "Rural",
                            "wealth_quintile": "Middle",
                        }
                    ]
                )
                prob = float(reloaded.predict_proba(sample)[0, 1])
                if not (0.0 <= prob <= 1.0):
                    errors.append(f"Reloaded model.joblib returned invalid probability {prob!r}.")
            except Exception as exc:
                errors.append(f"Reloading/scoring `model.joblib` failed: {type(exc).__name__}: {exc}")

        # Verify MODEL_CARD.md has all 5 headings
        card_path = out_dir / "MODEL_CARD.md"
        if card_path.exists():
            card_txt = card_path.read_text(encoding="utf-8")
            for h in _MODEL_CARD_HEADINGS:
                if h not in card_txt:
                    errors.append(f"MODEL_CARD.md is missing required heading `{h}`.")

        # Independently verify manifest.json SHA-256 matches model.joblib
        man_path = out_dir / "manifest.json"
        if man_path.exists() and model_path.exists():
            try:
                man = json.loads(man_path.read_text(encoding="utf-8"))
                actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if man.get("artifact_sha256") != actual_sha:
                    errors.append("manifest.json `artifact_sha256` does not match on-disk `model.joblib`!")
            except Exception as exc:
                errors.append(f"Could not parse manifest.json: {exc}")

    if errors:
        return CheckResult(
            "ch35_exam_03",
            False,
            "failed",
            "Exam Task 35.3 — Check End-to-End Research Release Packaging",
            "Clean `mch_df`, fit a standard `Pipeline` on a stratified 75/25 split, write `model.joblib`, `MODEL_CARD.md`, `config.json`, and `manifest.json` inside `out_dir`, verify integrity, and return the summary dict.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch35_exam_03",
        True,
        "passed",
        "Exam Task 35.3 Passed (35/35 pts) — End-to-End Reproducible Research Release Verified!",
        f"Packaged verified release bundle (test_auc={exp_auc}, 4 artifacts: MODEL_CARD.md, config.json, manifest.json, model.joblib) with clean reload and SHA-256 integrity inside a temp directory.",
        points_earned=35,
        points_possible=35,
        tip="You just built a peer-reviewer-proof ML release package: self-contained preprocessing, serialized weights, programmatic Model Card, config snapshot, and SHA-256 checksum!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch35_exam_03",
        chapter="ch35",
        title="Exam 35.3: End-to-End Research Release Packager (package_end_to_end_research_release)",
        book_section="Chapter 35 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Copy `mch_df`, strip/title-case `residence` & `wealth_quintile`, replace `999` with `np.nan` in numeric columns, and split with `train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)`.",
            2: "Fit a standard `Pipeline([('prep', ColumnTransformer(...)), ('model', LogisticRegression(max_iter=500))])`, compute `test_auc` and `brier`, dump `model.joblib`, write `MODEL_CARD.md`, call `build_reproducibility_bundle` and `verify_reproducibility_bundle`.",
            3: "def package_end_to_end_research_release(mch_df, out_dir, seed=42):\n    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)\n    # clean copy, stratified 75/25 split, fit standard Pipeline, compute test_auc & brier\n    model_path = out / 'model.joblib'; joblib.dump(pipe, model_path)\n    (out / 'MODEL_CARD.md').write_text(generate_model_card_markdown(meta), encoding='utf-8')\n    build_reproducibility_bundle(model_path, cfg, out)\n    ver = verify_reproducibility_bundle(out, model_path)\n    return {'bundle_valid': bool(ver['bundle_valid']), 'test_auc': test_auc, 'files_created': sorted(p.name for p in out.iterdir() if p.is_file())}",
        },
        verifier=_verify_exam_03,
    )
)
