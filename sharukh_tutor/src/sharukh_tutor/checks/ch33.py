"""Chapter 33 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch33_todo_01..06) and Coding Exam tasks (ch33_exam_01..03)
for Designing Rigorous Experiments — Config-Driven Runs, Multi-Seed Benchmarks,
and Ablation Studies (opening Part VII). Every verifier lazy-imports NumPy,
Pandas, SciPy, Scikit-Learn, PyTorch, and PyYAML inside the check function,
runs all YAML/JSONL file operations strictly inside tempfile.TemporaryDirectory(),
validates return schemas and non-mutation, and lets NotImplementedError
propagate cleanly for starter stubs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _is_int_strict(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


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
            f"{title} — Pass the Function Name Without Parentheses",
            (
                f"Expected a callable (like {expected_name}), "
                f"but received {type(fn).__name__} ({fn!r})."
            ),
            details=[
                f"Pass the function itself: tutor.check('{ex_id}', {expected_name})",
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
            f"Ensure {call_str} explicitly returns the required dictionary, DataFrame, or integer.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Experiment utilities must return structured outputs so automated ledgers and tests can consume them.",
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


# ---------------------------------------------------------------------------
# GUIDED TO-DO 33.1: set_all_seeds(seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np
    import torch

    bad = _check_callable("ch33_todo_01", "To-Do 33.1", fn, "set_all_seeds", 10)
    if bad is not None:
        return bad

    expected_keys = ["seed", "py_draw", "np_draw", "torch_draw"]
    try:
        r_default = fn()
        r1 = fn(42)
        r2 = fn(42)
        r_alt = fn(123)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch33_todo_01",
            False,
            "failed",
            "To-Do 33.1 — Runtime Error in `set_all_seeds`",
            f"Calling set_all_seeds(...) raised {type(exc).__name__}: {exc}",
            details=[
                "Call `random.seed(seed)`, `np.random.seed(seed)`, and `torch.manual_seed(seed)`.",
                "Draw one scalar from each (`random.random()`, `float(np.random.rand())`, `float(torch.rand(1).item())`).",
            ],
            points_earned=0,
            points_possible=10,
        )

    if r1 is None or r_default is None:
        return _none_return_trap("ch33_todo_01", "To-Do 33.1", "set_all_seeds(42)", 10)

    for label, res in [("default", r_default), ("seed=42", r1), ("seed=123", r_alt)]:
        key_err = _check_dict_keys("ch33_todo_01", f"To-Do 33.1 ({label})", res, expected_keys, 10)
        if key_err is not None:
            return key_err

    errors: list[str] = []
    if r1 != r2:
        errors.append(
            f"Calling set_all_seeds(42) twice produced different outputs ({r1} vs {r2})! "
            "Seed Python, NumPy, and PyTorch BEFORE drawing scalars."
        )
    if r_default != r1:
        errors.append(f"Default call set_all_seeds() ({r_default}) should match set_all_seeds(42) ({r1}).")

    for seed_val, got in [(42, r1), (123, r_alt)]:
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        exp = {
            "seed": int(seed_val),
            "py_draw": round(float(random.random()), 6),
            "np_draw": round(float(np.random.rand()), 6),
            "torch_draw": round(float(torch.rand(1).item()), 6),
        }
        if not _is_int_strict(got["seed"]) or got["seed"] != seed_val:
            errors.append(f"[seed={seed_val}] 'seed' should be int {seed_val}, got {got['seed']!r}.")
        for k in ("py_draw", "np_draw", "torch_draw"):
            if not _is_number(got[k]) or not math.isclose(float(got[k]), exp[k], abs_tol=1e-6):
                errors.append(
                    f"[seed={seed_val}] '{k}': expected {exp[k]}, got {got[k]!r}."
                )

    if r1["py_draw"] == r_alt["py_draw"] and r1["np_draw"] == r_alt["np_draw"]:
        errors.append("Outputs for seed=42 and seed=123 were identical — do not hardcode draws!")

    if errors:
        return CheckResult(
            "ch33_todo_01",
            False,
            "failed",
            "To-Do 33.1 — Check Multi-Library Seeding & Rounded Draws",
            "Seed `random`, `np.random`, and `torch`, draw one scalar from each, and return the 6-decimal rounded dictionary.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch33_todo_01",
        True,
        "passed",
        "To-Do 33.1 Passed — Bit-Exact Multi-Library Seed Control Verified!",
        "Confirmed bit-exact reproducibility across Python `random`, NumPy, and PyTorch for both default seed=42 and seed=123.",
        tip="Always lock every active RNG (Python, NumPy, PyTorch) at the very top of an experiment script before data splits or weight initialization.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_todo_01",
        chapter="ch33",
        title="To-Do 33.1: Multi-Library Deterministic Seed Control (set_all_seeds)",
        book_section="Section 33.3.1 (Multi-Library Seed Control)",
        points=10,
        hints={
            1: "Call `random.seed(seed)`, `np.random.seed(seed)`, and `torch.manual_seed(seed)` in sequence.",
            2: "Draw one scalar from each library: `random.random()`, `float(np.random.rand())`, and `float(torch.rand(1).item())`, rounding each to 6 decimals.",
            3: "def set_all_seeds(seed=42):\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(___)\n    return {\n        'seed': int(seed),\n        'py_draw': round(float(random.random()), 6),\n        'np_draw': round(float(np.random.rand()), 6),\n        'torch_draw': round(float(torch.rand(1).item()), 6),\n    }",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 33.2: load_and_validate_config(yaml_path)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import yaml

    bad = _check_callable("ch33_todo_02", "To-Do 33.2", fn, "load_and_validate_config", 10)
    if bad is not None:
        return bad

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        valid_cfg = {
            "experiment_name": "mch_stunting_audit",
            "seeds": [7, 21, 42],
            "model_type": "logistic_regression",
            "cv_splits": 5,
            "notes": "optional metadata allowed",
        }
        valid_path = tmp / "valid.yaml"
        valid_path.write_text(yaml.safe_dump(valid_cfg, sort_keys=False), encoding="utf-8")

        try:
            got_path = fn(valid_path)
            got_str = fn(str(valid_path))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch33_todo_02",
                False,
                "failed",
                "To-Do 33.2 — Runtime Error Loading Valid YAML Config",
                f"load_and_validate_config(valid_path) raised {type(exc).__name__}: {exc}",
                details=["Open `yaml_path` with UTF-8 encoding and parse via `yaml.safe_load`."],
                points_earned=0,
                points_possible=10,
            )

        if got_path is None:
            return _none_return_trap("ch33_todo_02", "To-Do 33.2", "load_and_validate_config(yaml_path)", 10)

        if not isinstance(got_path, dict) or got_path != valid_cfg:
            errors.append(f"Valid YAML did not return expected dict: got {got_path!r}.")
        if got_str != valid_cfg:
            errors.append("Should accept both `Path` and `str` file paths.")

        bad_cases = [
            ("missing_key.yaml", "experiment_name: x\nseeds: [42]\nmodel_type: lr\n", "missing required key `cv_splits`"),
            ("not_dict.yaml", "- item1\n- item2\n", "non-dict top-level YAML"),
            ("empty_seeds.yaml", "experiment_name: x\nseeds: []\nmodel_type: lr\ncv_splits: 5\n", "empty `seeds` list"),
            ("str_seeds.yaml", "experiment_name: x\nseeds: ['42', 7]\nmodel_type: lr\ncv_splits: 5\n", "non-integer item in `seeds`"),
            ("bool_seeds.yaml", "experiment_name: x\nseeds: [true, 7]\nmodel_type: lr\ncv_splits: 5\n", "boolean in `seeds` (not an int!)"),
            ("low_cv.yaml", "experiment_name: x\nseeds: [42]\nmodel_type: lr\ncv_splits: 1\n", "`cv_splits < 2`"),
        ]
        for fname, content, desc in bad_cases:
            p = tmp / fname
            p.write_text(content, encoding="utf-8")
            try:
                fn(p)
                errors.append(f"Expected `ValueError` for {desc}, but function returned normally.")
            except ValueError:
                pass
            except Exception as exc:
                errors.append(f"Expected `ValueError` for {desc}, but got {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch33_todo_02",
            False,
            "failed",
            "To-Do 33.2 — Check YAML Schema Validation Rules",
            "Load via `yaml.safe_load`, check `isinstance(cfg, dict)`, verify required keys `{'experiment_name', 'seeds', 'model_type', 'cv_splits'}`, validate `seeds` (non-empty list of ints, excluding bools) and `cv_splits >= 2`, raising `ValueError` otherwise.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch33_todo_02",
        True,
        "passed",
        "To-Do 33.2 Passed — YAML Config Loader & Strict Validator Verified!",
        "Validated clean YAML parsing plus fail-fast ValueError rejection across missing keys, non-dict YAML, invalid seeds, and cv_splits < 2.",
        tip="Always use `yaml.safe_load` (never `yaml.load`) and reject `bool` values when validating integer configs (`isinstance(x, int) and not isinstance(x, bool)`).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_todo_02",
        chapter="ch33",
        title="To-Do 33.2: Config-Driven YAML Validator (load_and_validate_config)",
        book_section="Section 33.3.2 (Config-Driven Runs With YAML)",
        points=10,
        hints={
            1: "Read the file with `yaml.safe_load(f)` and verify `isinstance(cfg, dict)` plus all four required keys `{'experiment_name', 'seeds', 'model_type', 'cv_splits'}`.",
            2: "Check that `seeds` is a non-empty `list` where every item satisfies `isinstance(s, int) and not isinstance(s, bool)`, and that `cv_splits` is an `int` (not `bool`) `>= 2`. Raise `ValueError` on any violation.",
            3: "def load_and_validate_config(yaml_path):\n    with open(yaml_path, 'r', encoding='utf-8') as f:\n        cfg = yaml.safe_load(f)\n    required = {'experiment_name', 'seeds', 'model_type', 'cv_splits'}\n    if not isinstance(cfg, dict) or not required.issubset(cfg.keys()):\n        raise ValueError('Invalid or incomplete config')\n    seeds = cfg['seeds']\n    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):\n        raise ValueError('seeds must be a non-empty list of ints')\n    cv = cfg['cv_splits']\n    if not isinstance(cv, int) or isinstance(cv, bool) or cv < ___:\n        raise ValueError('cv_splits must be >= 2')\n    return cfg",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 33.3: bootstrap_metric_ci(y_true, y_score, metric_fn, n_boot=300, alpha=0.05, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    bad = _check_callable("ch33_todo_03", "To-Do 33.3", fn, "bootstrap_metric_ci", 10)
    if bad is not None:
        return bad

    rng_data = np.random.default_rng(13)
    n = 180
    y_true = (rng_data.random(n) < 0.25).astype(int)
    y_score = np.clip(0.25 + 0.35 * y_true + rng_data.normal(0, 0.25, size=n), 0.01, 0.99)

    expected_keys = ["point", "ci_low", "ci_high", "se"]
    try:
        got_95 = fn(y_true, y_score, roc_auc_score, n_boot=300, alpha=0.05, seed=42)
        got_95_rep = fn(y_true, y_score, roc_auc_score, n_boot=300, alpha=0.05, seed=42)
        got_90 = fn(y_true, y_score, roc_auc_score, n_boot=300, alpha=0.10, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch33_todo_03",
            False,
            "failed",
            "To-Do 33.3 — Runtime Error in `bootstrap_metric_ci`",
            f"bootstrap_metric_ci(...) raised {type(exc).__name__}: {exc}",
            details=[
                "Convert `y_true` and `y_score` with `np.asarray(...)` before indexing by bootstrap indices.",
                "Skip degenerate single-class resamples via `if len(np.unique(y_b)) < 2: continue`.",
            ],
            points_earned=0,
            points_possible=10,
        )

    if got_95 is None:
        return _none_return_trap("ch33_todo_03", "To-Do 33.3", "bootstrap_metric_ci(...)", 10)

    key_err = _check_dict_keys("ch33_todo_03", "To-Do 33.3", got_95, expected_keys, 10)
    if key_err is not None:
        return key_err

    # Reference computation
    ref_point = round(float(roc_auc_score(y_true, y_score)), 4)
    rng_ref = np.random.default_rng(42)
    boots = []
    for _ in range(300):
        idx = rng_ref.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    ref_lo, ref_hi = np.percentile(boots, [2.5, 97.5])
    ref_se = float(np.std(boots, ddof=1))

    errors: list[str] = []
    if got_95 != got_95_rep:
        errors.append("Repeated calls with the same `seed=42` did not return identical results — use `np.random.default_rng(seed)`.")
    if not _is_number(got_95["point"]) or not math.isclose(float(got_95["point"]), ref_point, abs_tol=1e-4):
        errors.append(f"point: expected {ref_point}, got {got_95['point']!r}.")
    if not _is_number(got_95["ci_low"]) or not math.isclose(float(got_95["ci_low"]), float(ref_lo), abs_tol=0.015):
        errors.append(f"ci_low: expected ~{round(float(ref_lo), 4)}, got {got_95['ci_low']!r}.")
    if not _is_number(got_95["ci_high"]) or not math.isclose(float(got_95["ci_high"]), float(ref_hi), abs_tol=0.015):
        errors.append(f"ci_high: expected ~{round(float(ref_hi), 4)}, got {got_95['ci_high']!r}.")
    if not _is_number(got_95["se"]) or not math.isclose(float(got_95["se"]), ref_se, abs_tol=0.008):
        errors.append(f"se: expected ~{round(ref_se, 4)}, got {got_95['se']!r}.")

    if all(_is_number(got_95[k]) for k in expected_keys):
        if not (got_95["ci_low"] < got_95["point"] < got_95["ci_high"]):
            errors.append(f"Expected ci_low < point < ci_high, got {got_95}.")
        w95 = float(got_95["ci_high"]) - float(got_95["ci_low"])
        w90 = float(got_90["ci_high"]) - float(got_90["ci_low"])
        if w90 >= w95:
            errors.append(f"90% CI (alpha=0.10, width={w90:.4f}) should be narrower than 95% CI (alpha=0.05, width={w95:.4f}).")

    # Degenerate single-class resample safety test
    y_rare = np.array([1, 1] + [0] * 18)
    s_rare = np.linspace(0.9, 0.1, 20)
    try:
        got_rare = fn(y_rare, s_rare, roc_auc_score, n_boot=120, alpha=0.05, seed=7)
        if not isinstance(got_rare, dict) or not _is_number(got_rare.get("point")):
            errors.append("Rare-event test did not return a valid summary dict.")
    except Exception as exc:
        errors.append(
            f"Crashed on rare-event cohort where some bootstrap draws have only 1 class ({type(exc).__name__}: {exc}). "
            "Add `if len(np.unique(y_b)) < 2: continue`!"
        )

    if errors:
        return CheckResult(
            "ch33_todo_03",
            False,
            "failed",
            "To-Do 33.3 — Check Bootstrap Resampling, Percentile CI & Degenerate Guard",
            "Use `np.random.default_rng(seed)` to draw `n_boot` resamples, skip single-class resamples, and return rounded `point`, `ci_low`, `ci_high`, and `se`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch33_todo_03",
        True,
        "passed",
        "To-Do 33.3 Passed — Non-Parametric Bootstrap Confidence Interval Verified!",
        "Verified deterministic `default_rng` resampling, percentile CI bounds across alpha=0.05/0.10, standard error, and single-class degenerate guard.",
        tip="Never publish a bare ROC-AUC point estimate like 0.695 without a 95% CI — small cohorts often span ±0.06 AUC!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_todo_03",
        chapter="ch33",
        title="To-Do 33.3: Bootstrap Metric Confidence Interval (bootstrap_metric_ci)",
        book_section="Section 33.3.4 (Bootstrap Confidence Intervals)",
        points=10,
        hints={
            1: "Convert inputs via `np.asarray`, compute `point = float(metric_fn(yt, ys))`, and create `rng = np.random.default_rng(seed)`.",
            2: "Loop `n_boot` times: draw `idx = rng.integers(0, len(yt), size=len(yt))`, skip if `len(np.unique(yt[idx])) < 2`, and append `float(metric_fn(yt[idx], ys[idx]))`.",
            3: "def bootstrap_metric_ci(y_true, y_score, metric_fn, n_boot=300, alpha=0.05, seed=42):\n    yt, ys = np.asarray(y_true), np.asarray(y_score)\n    point = float(metric_fn(yt, ys))\n    rng = np.random.default_rng(seed)\n    boots = []\n    for _ in range(n_boot):\n        idx = rng.integers(0, len(yt), size=len(yt))\n        if len(np.unique(yt[idx])) < 2:\n            continue\n        boots.append(float(metric_fn(yt[idx], ys[idx])))\n    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])\n    return {'point': round(point, 4), 'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4), 'se': round(float(np.std(boots, ddof=1)), 4)}",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 33.4: paired_bootstrap_test(y_true, score_a, score_b, metric_fn, n_boot=400, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    bad = _check_callable("ch33_todo_04", "To-Do 33.4", fn, "paired_bootstrap_test", 10)
    if bad is not None:
        return bad

    rng_data = np.random.default_rng(29)
    n = 200
    y_true = (rng_data.random(n) < 0.30).astype(int)
    score_a = np.clip(0.30 + 0.10 * y_true + rng_data.normal(0, 0.30, size=n), 0.01, 0.99)
    score_b = np.clip(0.20 + 0.50 * y_true + rng_data.normal(0, 0.22, size=n), 0.01, 0.99)

    expected_keys = ["delta_point", "ci_low", "ci_high", "p_value_two_sided"]
    try:
        got_sig = fn(y_true, score_a, score_b, roc_auc_score, n_boot=400, seed=42)
        got_sig_rep = fn(y_true, score_a, score_b, roc_auc_score, n_boot=400, seed=42)
        got_tie = fn(y_true, score_a, score_a, roc_auc_score, n_boot=200, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch33_todo_04",
            False,
            "failed",
            "To-Do 33.4 — Runtime Error in `paired_bootstrap_test`",
            f"paired_bootstrap_test(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got_sig is None:
        return _none_return_trap("ch33_todo_04", "To-Do 33.4", "paired_bootstrap_test(...)", 10)

    key_err = _check_dict_keys("ch33_todo_04", "To-Do 33.4", got_sig, expected_keys, 10)
    if key_err is not None:
        return key_err

    ref_delta = float(roc_auc_score(y_true, score_b) - roc_auc_score(y_true, score_a))
    rng_ref = np.random.default_rng(42)
    deltas = []
    for _ in range(400):
        idx = rng_ref.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(
            float(roc_auc_score(y_true[idx], score_b[idx]) - roc_auc_score(y_true[idx], score_a[idx]))
        )
    arr = np.asarray(deltas)
    ref_lo, ref_hi = np.percentile(arr, [2.5, 97.5])
    ref_p = min(1.0, max(0.0, 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))))

    errors: list[str] = []
    if got_sig != got_sig_rep:
        errors.append("Repeated calls with identical `seed=42` differed — use `np.random.default_rng(seed)`.")
    if not _is_number(got_sig["delta_point"]) or not math.isclose(float(got_sig["delta_point"]), ref_delta, abs_tol=1e-4):
        errors.append(
            f"delta_point: expected {round(ref_delta, 4)} (score_b - score_a), got {got_sig['delta_point']!r}."
        )
    if not _is_number(got_sig["ci_low"]) or not math.isclose(float(got_sig["ci_low"]), float(ref_lo), abs_tol=0.015):
        errors.append(f"ci_low: expected ~{round(float(ref_lo), 4)}, got {got_sig['ci_low']!r}.")
    if not _is_number(got_sig["ci_high"]) or not math.isclose(float(got_sig["ci_high"]), float(ref_hi), abs_tol=0.015):
        errors.append(f"ci_high: expected ~{round(float(ref_hi), 4)}, got {got_sig['ci_high']!r}.")
    if not _is_number(got_sig["p_value_two_sided"]) or not math.isclose(float(got_sig["p_value_two_sided"]), ref_p, abs_tol=0.02):
        errors.append(f"p_value_two_sided: expected ~{round(ref_p, 4)}, got {got_sig['p_value_two_sided']!r}.")

    # Tie case tests [0.0, 1.0] clipping (without clipping, 2 * min(1.0, 1.0) == 2.0!)
    if not _is_number(got_tie["p_value_two_sided"]) or not (0.0 <= float(got_tie["p_value_two_sided"]) <= 1.0):
        errors.append(
            f"Identical-model tie returned p_value_two_sided={got_tie['p_value_two_sided']!r} outside [0.0, 1.0]! "
            "Clip `2 * min(mean(deltas <= 0), mean(deltas >= 0))` to [0.0, 1.0]."
        )
    elif not math.isclose(float(got_tie["delta_point"]), 0.0, abs_tol=1e-4) or not math.isclose(float(got_tie["p_value_two_sided"]), 1.0, abs_tol=1e-4):
        errors.append(f"Identical scores should yield delta_point=0.0 and p_value_two_sided=1.0, got {got_tie}.")

    if errors:
        return CheckResult(
            "ch33_todo_04",
            False,
            "failed",
            "To-Do 33.4 — Check Paired Bootstrap Difference & Two-Sided p-Value",
            "Evaluate both models on the EXACT same bootstrap indices `idx`, compute `delta_i = metric(y_b, sb_b) - metric(y_b, sa_b)`, 95% CI, and clipped two-sided p-value.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch33_todo_04",
        True,
        "passed",
        "To-Do 33.4 Passed — Paired Bootstrap Hypothesis Test Verified!",
        "Confirmed paired-index resampling (B - A), 95% difference CI, and properly clipped two-sided p-value on both significant and identical-model cases.",
        tip="Evaluating Model A and Model B on identical bootstrap indices cancels shared patient sampling variance—yielding dramatically tighter CIs than unpaired comparisons!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_todo_04",
        chapter="ch33",
        title="To-Do 33.4: Paired Bootstrap Significance Test (paired_bootstrap_test)",
        book_section="Section 33.3.4 (Paired Bootstrap vs. Exact McNemar Test)",
        points=10,
        hints={
            1: "Compute `delta_point = float(metric_fn(yt, sb) - metric_fn(yt, sa))` and draw one shared index array `idx = rng.integers(0, n, size=n)` per resample.",
            2: "For each valid resample (`len(np.unique(yt[idx])) >= 2`), append `metric_fn(yt[idx], sb[idx]) - metric_fn(yt[idx], sa[idx])`. Compute 95% CI via `np.percentile(arr, [2.5, 97.5])`.",
            3: "p_val = float(np.clip(2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)), 0.0, 1.0))\nreturn {'delta_point': round(delta_point, 4), 'ci_low': round(float(lo), 4), 'ci_high': round(float(hi), 4), 'p_value_two_sided': round(p_val, 4)}",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 33.5: run_feature_ablation(X_df, y, feature_groups, model_factory, cv_splits=5, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch33_todo_05", "To-Do 33.5", fn, "run_feature_ablation", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(17)
    n = 180
    sig1 = rng.normal(0, 1, size=n)
    sig2 = rng.normal(0, 1, size=n)
    noise1 = rng.normal(0, 1, size=n)
    noise2 = rng.normal(0, 1, size=n)
    logits = 1.6 * sig1 - 1.2 * sig2 + rng.normal(0, 0.8, size=n)
    y_syn = (logits > 0).astype(int)
    X_orig = pd.DataFrame({
        "sig1": sig1,
        "sig2": sig2,
        "noise1": noise1,
        "noise2": noise2,
    })
    X_copy = X_orig.copy(deep=True)
    groups = {
        "signal_block": ["sig1", "sig2"],
        "noise_block": ["noise1", "noise2"],
    }
    factory = lambda: LogisticRegression(max_iter=500)

    try:
        got = fn(X_copy, y_syn, groups, factory, cv_splits=5, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch33_todo_05",
            False,
            "failed",
            "To-Do 33.5 — Runtime Error in `run_feature_ablation`",
            f"run_feature_ablation(...) raised {type(exc).__name__}: {exc}",
            details=[
                "Evaluate `X_df` for `'full_model'`, then `X_df.drop(columns=cols)` for each group.",
                "Use `Pipeline([('scaler', StandardScaler()), ('model', model_factory())])` + `StratifiedKFold(cv_splits, shuffle=True, random_state=seed)`.",
            ],
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap("ch33_todo_05", "To-Do 33.5", "run_feature_ablation(...)", 10)

    if not isinstance(got, pd.DataFrame):
        return CheckResult(
            "ch33_todo_05",
            False,
            "failed",
            "To-Do 33.5 — Expected a Pandas DataFrame",
            f"Expected pd.DataFrame, got {type(got).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []
    if not X_copy.equals(X_orig):
        errors.append("Input `X_df` was mutated in place! Use `X_df.drop(columns=cols)` (without `inplace=True`).")

    expected_cols = ["ablation", "mean_auc", "std_auc", "delta_vs_full"]
    if list(got.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {list(got.columns)}.")
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pipe_full = Pipeline([("scaler", StandardScaler()), ("model", factory())])
        sc_full = cross_val_score(pipe_full, X_orig, y_syn, cv=skf, scoring="roc_auc")
        full_auc = float(sc_full.mean())
        ref_rows = {
            "full_model": (round(full_auc, 4), round(float(sc_full.std()), 4), 0.0),
        }
        for gname, cols in groups.items():
            pipe_g = Pipeline([("scaler", StandardScaler()), ("model", factory())])
            sc_g = cross_val_score(pipe_g, X_orig.drop(columns=cols), y_syn, cv=skf, scoring="roc_auc")
            m = float(sc_g.mean())
            ref_rows[f"drop_{gname}"] = (round(m, 4), round(float(sc_g.std()), 4), round(m - full_auc, 4))

        expected_names = ["full_model", "drop_signal_block", "drop_noise_block"]
        got_names = got["ablation"].tolist()
        if got_names != expected_names:
            errors.append(f"Expected ablation rows {expected_names}, got {got_names}.")
        else:
            for idx, name in enumerate(expected_names):
                exp_m, exp_s, exp_d = ref_rows[name]
                g_m = float(got.iloc[idx]["mean_auc"])
                g_s = float(got.iloc[idx]["std_auc"])
                g_d = float(got.iloc[idx]["delta_vs_full"])
                if not math.isclose(g_m, exp_m, abs_tol=1e-3):
                    errors.append(f"[{name}] mean_auc: expected {exp_m}, got {g_m}.")
                if not math.isclose(g_s, exp_s, abs_tol=1e-3):
                    errors.append(f"[{name}] std_auc: expected {exp_s}, got {g_s}.")
                if not math.isclose(g_d, exp_d, abs_tol=1e-3):
                    errors.append(
                        f"[{name}] delta_vs_full: expected {exp_d} (`mean_auc - full_auc`, so negative means dropping hurt!), got {g_d}."
                    )

    if errors:
        return CheckResult(
            "ch33_todo_05",
            False,
            "failed",
            "To-Do 33.5 — Check Feature-Group Ablation Table",
            "Evaluate `'full_model'` and each `'drop_<group>'` with a leak-free `Pipeline(StandardScaler, model_factory())` and `StratifiedKFold(cv_splits, shuffle=True, random_state=seed)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch33_todo_05",
        True,
        "passed",
        "To-Do 33.5 Passed — Systematic Feature-Group Ablation Engine Verified!",
        "Confirmed leak-free CV evaluation across full and leave-one-group-out subsets, negative-when-harmful deltas, and zero caller mutation.",
        tip="Always sign `delta_vs_full = ablated_auc - full_auc`: a strongly negative delta immediately tells reviewers which feature group carried the signal.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_todo_05",
        chapter="ch33",
        title="To-Do 33.5: Systematic Feature-Group Ablation (run_feature_ablation)",
        book_section="Section 33.3.5 (Systematic Feature-Group Ablation Studies)",
        points=10,
        hints={
            1: "Create `skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)` and evaluate `Pipeline([('scaler', StandardScaler()), ('model', model_factory())])` on `X_df` first to obtain `full_auc`.",
            2: "Loop over `for group_name, cols in feature_groups.items():`, build `X_sub = X_df.drop(columns=cols)`, score with `cross_val_score(..., cv=skf, scoring='roc_auc')`, and compute `delta_vs_full = round(mean_auc - full_auc, 4)`.",
            3: "rows = [{'ablation': 'full_model', 'mean_auc': round(full_auc, 4), 'std_auc': round(float(full_sc.std()), 4), 'delta_vs_full': 0.0}]\nfor gname, cols in feature_groups.items():\n    pipe = Pipeline([('scaler', StandardScaler()), ('model', model_factory())])\n    sc = cross_val_score(pipe, X_df.drop(columns=cols), y, cv=skf, scoring='roc_auc')\n    rows.append({'ablation': f'drop_{gname}', 'mean_auc': round(float(sc.mean()), 4), 'std_auc': round(float(sc.std()), 4), 'delta_vs_full': round(float(sc.mean()) - full_auc, 4)})\nreturn pd.DataFrame(rows)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 33.6: append_experiment_ledger(ledger_path, run_record)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch33_todo_06", "To-Do 33.6", fn, "append_experiment_ledger", 10)
    if bad is not None:
        return bad

    errors: list[str] = []
    rec1 = {"model": "lr", "auc": 0.7127, "seed": 42}
    rec2 = {"seed": 101, "model": "rf", "auc": 0.6998}

    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "runs.jsonl"
        try:
            c1 = fn(ledger_path, rec1)
            c2 = fn(ledger_path, rec2)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch33_todo_06",
                False,
                "failed",
                "To-Do 33.6 — Runtime Error in `append_experiment_ledger`",
                f"append_experiment_ledger(ledger_path, run_record) raised {type(exc).__name__}: {exc}",
                details=["Open `ledger_path` in `'a'` mode with `encoding='utf-8'` and write `json.dumps(run_record, sort_keys=True) + '\\n'`."],
                points_earned=0,
                points_possible=10,
            )

        if c1 is None or c2 is None:
            return _none_return_trap("ch33_todo_06", "To-Do 33.6", "append_experiment_ledger(path, rec)", 10)

        if not _is_int_strict(c1) or c1 != 1:
            errors.append(f"After 1st append, expected int 1, got {c1!r}.")
        if not _is_int_strict(c2) or c2 != 2:
            errors.append(f"After 2nd append, expected int 2, got {c2!r} (did you open with `'w'` instead of `'a'`?).")

        if not ledger_path.exists():
            errors.append("Ledger file was not created at `ledger_path`.")
        else:
            raw_lines = [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(raw_lines) != 2:
                errors.append(f"Expected 2 non-empty JSONL lines on disk, found {len(raw_lines)}.")
            else:
                for i, (ln, exp_rec) in enumerate(zip(raw_lines, [rec1, rec2]), start=1):
                    try:
                        parsed = json.loads(ln)
                        if parsed != exp_rec:
                            errors.append(f"Line {i} decoded to {parsed!r}, expected {exp_rec!r}.")
                        elif ln != json.dumps(exp_rec, sort_keys=True):
                            errors.append(f"Line {i} was not serialized with `json.dumps(run_record, sort_keys=True)`.")
                    except json.JSONDecodeError as exc:
                        errors.append(f"Line {i} is not valid JSON ({exc}).")

    if errors:
        return CheckResult(
            "ch33_todo_06",
            False,
            "failed",
            "To-Do 33.6 — Check JSONL Append Mode & Line Counting",
            "Append `json.dumps(run_record, sort_keys=True) + '\\n'` in UTF-8 `'a'` mode and return the total count of non-empty lines as an `int`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch33_todo_06",
        True,
        "passed",
        "To-Do 33.6 Passed — Append-Only JSONL Experiment Ledger Verified!",
        "Confirmed append-only UTF-8 JSONL persistence with deterministic `sort_keys=True` formatting and accurate non-empty line counting inside a clean temp directory.",
        tip="Append-only JSONL ledgers never corrupt previous runs when a job crashes mid-sweep and load directly into Pandas via `pd.read_json(path, lines=True)`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_todo_06",
        chapter="ch33",
        title="To-Do 33.6: Append-Only JSONL Experiment Ledger (append_experiment_ledger)",
        book_section="Section 33.3.2 (Append-Only JSONL Experiment Tracking)",
        points=10,
        hints={
            1: "Open `ledger_path` with `open(ledger_path, 'a', encoding='utf-8')` and write `json.dumps(run_record, sort_keys=True) + '\\n'`.",
            2: "Re-open `ledger_path` in `'r'` mode (UTF-8) and return `sum(1 for line in f if line.strip())`.",
            3: "def append_experiment_ledger(ledger_path, run_record):\n    with open(ledger_path, 'a', encoding='utf-8') as f:\n        f.write(json.dumps(run_record, sort_keys=True) + '\\n')\n    with open(ledger_path, 'r', encoding='utf-8') as f:\n        return int(sum(1 for line in f if line.strip()))",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 33.1 (30 pts): multi_seed_cv_benchmark(X, y, model_factory, seeds, cv_splits=4)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch33_exam_01", "Exam Task 33.1", fn, "multi_seed_cv_benchmark", 30)
    if bad is not None:
        return bad

    X_syn, y_syn = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        weights=[0.75, 0.25],
        random_state=19,
    )
    seeds = [7, 21, 42, 99]
    call_count = 0

    def tracked_factory():
        nonlocal call_count
        call_count += 1
        return LogisticRegression(max_iter=500)

    expected_keys = ["n_seeds", "mean_auc", "std_auc", "min_auc", "max_auc", "seed_scores"]
    try:
        got = fn(X_syn, y_syn, tracked_factory, seeds, cv_splits=4)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch33_exam_01",
            False,
            "failed",
            "Exam Task 33.1 — Runtime Error in `multi_seed_cv_benchmark`",
            f"multi_seed_cv_benchmark(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if got is None:
        return _none_return_trap("ch33_exam_01", "Exam Task 33.1", "multi_seed_cv_benchmark(...)", 30)

    key_err = _check_dict_keys("ch33_exam_01", "Exam Task 33.1", got, expected_keys, 30)
    if key_err is not None:
        return key_err

    ref_raw = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=s)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))])
        sc = cross_val_score(pipe, X_syn, y_syn, cv=skf, scoring="roc_auc")
        ref_raw.append(float(sc.mean()))

    exp_scores = [round(v, 4) for v in ref_raw]
    exp_mean = round(float(np.mean(ref_raw)), 4)
    exp_std_0 = round(float(np.std(ref_raw, ddof=0)), 4)
    exp_std_1 = round(float(np.std(ref_raw, ddof=1)), 4)
    exp_min = round(float(np.min(ref_raw)), 4)
    exp_max = round(float(np.max(ref_raw)), 4)

    errors: list[str] = []
    if call_count < len(seeds):
        errors.append(
            f"Expected `model_factory()` to be called fresh for each seed (>= {len(seeds)} times), got {call_count} call(s)."
        )
    if not _is_int_strict(got["n_seeds"]) or got["n_seeds"] != len(seeds):
        errors.append(f"n_seeds: expected int {len(seeds)}, got {got['n_seeds']!r}.")
    if not isinstance(got["seed_scores"], list) or len(got["seed_scores"]) != len(seeds):
        errors.append(f"seed_scores: expected list of length {len(seeds)}, got {got['seed_scores']!r}.")
    elif not all(_is_number(a) and math.isclose(float(a), b, abs_tol=1e-3) for a, b in zip(got["seed_scores"], exp_scores)):
        errors.append(f"seed_scores mismatch: expected {exp_scores}, got {got['seed_scores']!r}.")
    if not _is_number(got["mean_auc"]) or not math.isclose(float(got["mean_auc"]), exp_mean, abs_tol=1e-3):
        errors.append(f"mean_auc: expected {exp_mean}, got {got['mean_auc']!r}.")
    if not _is_number(got["std_auc"]) or not (
        math.isclose(float(got["std_auc"]), exp_std_0, abs_tol=1e-3)
        or math.isclose(float(got["std_auc"]), exp_std_1, abs_tol=1e-3)
    ):
        errors.append(f"std_auc: expected {exp_std_0}, got {got['std_auc']!r}.")
    if not _is_number(got["min_auc"]) or not math.isclose(float(got["min_auc"]), exp_min, abs_tol=1e-3):
        errors.append(f"min_auc: expected {exp_min}, got {got['min_auc']!r}.")
    if not _is_number(got["max_auc"]) or not math.isclose(float(got["max_auc"]), exp_max, abs_tol=1e-3):
        errors.append(f"max_auc: expected {exp_max}, got {got['max_auc']!r}.")

    if errors:
        return CheckResult(
            "ch33_exam_01",
            False,
            "failed",
            "Exam Task 33.1 — Check Multi-Seed CV Loop & Summary Statistics",
            "Run `StratifiedKFold(cv_splits, shuffle=True, random_state=s)` + leak-free `Pipeline(StandardScaler, model_factory())` for every `s` in `seeds`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch33_exam_01",
        True,
        "passed",
        "Exam Task 33.1 Passed (30/30 pts) — Multi-Seed CV Benchmark Verified!",
        f"Benchmarked 4 random seeds with fresh leak-free pipelines (mean={exp_mean}, std={exp_std_0}, range=[{exp_min}, {exp_max}]).",
        points_earned=30,
        points_possible=30,
        tip="Reporting mean ± std alongside min/max across multiple seeds protects manuscripts against lucky-split variance.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_exam_01",
        chapter="ch33",
        title="Exam 33.1: Multi-Seed Cross-Validation Benchmark (multi_seed_cv_benchmark)",
        book_section="Chapter 33 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Loop over `for s in seeds:`, create `StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=s)` and a fresh `Pipeline([('scaler', StandardScaler()), ('model', model_factory())])`.",
            2: "Compute `float(cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc').mean())` for each seed and collect the values in a list `raw`.",
            3: "return {\n    'n_seeds': int(len(seeds)),\n    'mean_auc': round(float(np.mean(raw)), 4),\n    'std_auc': round(float(np.std(raw)), 4),\n    'min_auc': round(float(np.min(raw)), 4),\n    'max_auc': round(float(np.max(raw)), 4),\n    'seed_scores': [round(float(v), 4) for v in raw],\n}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 33.2 (35 pts): mcnemar_exact_comparison(y_true, pred_a, pred_b)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    from scipy.stats import binomtest

    bad = _check_callable("ch33_exam_02", "Exam Task 33.2", fn, "mcnemar_exact_comparison", 35)
    if bad is not None:
        return bad

    expected_keys = ["b_only_a_correct", "c_only_b_correct", "p_value", "significant_05"]

    # Case 1: b=2, c=18 (strongly significant)
    y1 = np.ones(40, dtype=int)
    pa1 = np.array([1] * 20 + [1] * 2 + [0] * 18)
    pb1 = np.array([1] * 20 + [0] * 2 + [1] * 18)

    # Case 2: b=8, c=10 (not significant)
    y2 = np.ones(30, dtype=int)
    pa2 = np.array([1] * 12 + [1] * 8 + [0] * 10)
    pb2 = np.array([1] * 12 + [0] * 8 + [1] * 10)

    # Case 3: b=0, c=0 (zero discordance edge case)
    y3 = np.array([0, 1, 0, 1, 1, 0])
    pa3 = np.array([0, 1, 1, 1, 0, 0])
    pb3 = np.array([0, 1, 1, 1, 0, 0])

    cases = [
        ("significant (b=2, c=18)", y1, pa1, pb1, 2, 18),
        ("non-significant (b=8, c=10)", y2, pa2, pb2, 8, 10),
        ("zero discordance (b=0, c=0)", y3, pa3, pb3, 0, 0),
    ]

    errors: list[str] = []
    for label, yt, pa, pb, exp_b, exp_c in cases:
        try:
            got = fn(yt, pa, pb)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch33_exam_02",
                False,
                "failed",
                f"Exam Task 33.2 — Runtime Error on {label}",
                f"mcnemar_exact_comparison(...) raised {type(exc).__name__}: {exc}",
                details=["Handle the `b + c == 0` edge case explicitly by returning `p_value = 1.0`."],
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap("ch33_exam_02", "Exam Task 33.2", "mcnemar_exact_comparison(y, pred_a, pred_b)", 35)

        key_err = _check_dict_keys("ch33_exam_02", f"Exam Task 33.2 ({label})", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        exp_p = 1.0 if (exp_b + exp_c) == 0 else float(binomtest(min(exp_b, exp_c), exp_b + exp_c, 0.5).pvalue)
        exp_p_r = round(exp_p, 4)
        exp_sig = bool(exp_p < 0.05)

        if not _is_int_strict(got["b_only_a_correct"]) or got["b_only_a_correct"] != exp_b:
            errors.append(f"[{label}] b_only_a_correct: expected {exp_b}, got {got['b_only_a_correct']!r}.")
        if not _is_int_strict(got["c_only_b_correct"]) or got["c_only_b_correct"] != exp_c:
            errors.append(f"[{label}] c_only_b_correct: expected {exp_c}, got {got['c_only_b_correct']!r}.")
        if not _is_number(got["p_value"]) or not math.isclose(float(got["p_value"]), exp_p_r, abs_tol=1e-4):
            errors.append(f"[{label}] p_value: expected {exp_p_r}, got {got['p_value']!r}.")
        if not _is_bool_like(got["significant_05"]) or bool(got["significant_05"]) != exp_sig:
            errors.append(f"[{label}] significant_05: expected {exp_sig}, got {got['significant_05']!r}.")

    if errors:
        return CheckResult(
            "ch33_exam_02",
            False,
            "failed",
            "Exam Task 33.2 — Check Exact McNemar Discordant Counts & Binomial Test",
            "Count `b` (A right, B wrong) and `c` (A wrong, B right), return `p_value=1.0` when `b + c == 0`, else `scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch33_exam_02",
        True,
        "passed",
        "Exam Task 33.2 Passed (35/35 pts) — Exact McNemar Paired Test Verified!",
        "Passed significant (b=2, c=18, p=0.0004), non-significant (b=8, c=10, p=0.8145), and zero-discordance (b=0, c=0, p=1.0) test cases.",
        points_earned=35,
        points_possible=35,
        tip="Exact binomial McNemar stays valid even when discordant counts `b + c < 25`—where the classic chi-square approximation breaks down.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_exam_02",
        chapter="ch33",
        title="Exam 33.2: Exact McNemar Paired Classifier Test (mcnemar_exact_comparison)",
        book_section="Chapter 33 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Convert inputs with `np.asarray`, build boolean masks `a_ok = (pa == yt)` and `b_ok = (pb == yt)`, then count `b = int(np.sum(a_ok & ~b_ok))` and `c = int(np.sum(~a_ok & b_ok))`.",
            2: "If `b + c == 0`, set `p_val = 1.0`; otherwise call `scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue`.",
            3: "def mcnemar_exact_comparison(y_true, pred_a, pred_b):\n    yt, pa, pb = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)\n    a_ok, b_ok = (pa == yt), (pb == yt)\n    b = int(np.sum(a_ok & ~b_ok))\n    c = int(np.sum(~a_ok & b_ok))\n    p_val = 1.0 if (b + c) == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)\n    return {'b_only_a_correct': b, 'c_only_b_correct': c, 'p_value': round(p_val, 4), 'significant_05': bool(p_val < 0.05)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 33.3 (35 pts): mch_ablation_study_report(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_mch_ablation(mch_df: Any, seed: int = 42) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["is_urban"] = (work["residence"] == "Urban").astype(int)
    wealth_map = {
        "Poorest": 1,
        "Poorer": 2,
        "Middle": 3,
        "Richer": 4,
        "Richest": 5,
    }
    work["wealth_rank"] = (
        work["wealth_quintile"].astype(str).str.strip().str.title().map(wealth_map)
    )
    feature_groups = {
        "demographics": ["maternal_age_years", "is_urban"],
        "care_access": ["anc_visits"],
        "socioeconomic": ["wealth_rank"],
        "nutrition": ["maternal_hb_g_dl"],
    }
    all_cols = [c for cols in feature_groups.values() for c in cols]
    clean = work[all_cols + ["child_stunted", "cluster_id"]].dropna().reset_index(drop=True)
    X = clean[all_cols]
    y = clean["child_stunted"].astype(int)
    clusters = clean["cluster_id"]

    gkf = GroupKFold(n_splits=5)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500)),
    ])
    full_scores = cross_val_score(pipe, X, y, groups=clusters, cv=gkf, scoring="roc_auc")
    full_auc = float(full_scores.mean())
    rows = [
        {
            "ablation": "full_model",
            "mean_auc": round(full_auc, 4),
            "std_auc": round(float(full_scores.std()), 4),
            "delta_vs_full": 0.0,
        }
    ]
    for gname, cols in feature_groups.items():
        sc = cross_val_score(
            pipe, X.drop(columns=cols), y, groups=clusters, cv=gkf, scoring="roc_auc"
        )
        m = float(sc.mean())
        rows.append(
            {
                "ablation": f"drop_{gname}",
                "mean_auc": round(m, 4),
                "std_auc": round(float(sc.std()), 4),
                "delta_vs_full": round(m - full_auc, 4),
            }
        )
    tbl = pd.DataFrame(rows)
    drops = tbl[tbl["ablation"] != "full_model"]
    critical = str(
        drops.loc[drops["delta_vs_full"].idxmin(), "ablation"]
    ).removeprefix("drop_")
    return {
        "full_auc": round(full_auc, 4),
        "ablation_table": tbl,
        "most_critical_group": critical,
    }


def _verify_exam_03(fn: Any) -> CheckResult:
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch33_exam_03", "Exam Task 33.3", fn, "mch_ablation_study_report", 35)
    if bad is not None:
        return bad

    mch_raw = load_data("mch_survey_sample", as_pandas=True)
    mch_copy = mch_raw.copy(deep=True)
    expected_keys = ["full_auc", "ablation_table", "most_critical_group"]

    try:
        got = fn(mch_copy, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch33_exam_03",
            False,
            "failed",
            "Exam Task 33.3 — Runtime Error in `mch_ablation_study_report`",
            f"mch_ablation_study_report(mch_df, seed=42) raised {type(exc).__name__}: {exc}",
            details=[
                "Clean `residence`, replace `anc_visits == 999` with `np.nan`, derive `is_urban` and `wealth_rank` (1..5), and `.dropna()` across the 5 features + `child_stunted` + `cluster_id`.",
                "Pass `groups=clusters` to `cross_val_score(..., cv=GroupKFold(n_splits=5), scoring='roc_auc')`.",
            ],
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap("ch33_exam_03", "Exam Task 33.3", "mch_ablation_study_report(mch_df)", 35)

    key_err = _check_dict_keys("ch33_exam_03", "Exam Task 33.3", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    errors: list[str] = []
    if not mch_copy.equals(mch_raw):
        errors.append("Input `mch_df` was mutated in place! Start with `work = mch_df.copy()`.")

    tbl = got["ablation_table"]
    if not isinstance(tbl, pd.DataFrame):
        return CheckResult(
            "ch33_exam_03",
            False,
            "failed",
            "Exam Task 33.3 — `'ablation_table'` Must Be a Pandas DataFrame",
            f"Expected pd.DataFrame, got {type(tbl).__name__}.",
            points_earned=0,
            points_possible=35,
        )

    exp = _ref_mch_ablation(mch_raw, seed=42)
    exp_tbl = exp["ablation_table"]
    exp_by_name = exp_tbl.set_index("ablation")

    req_cols = ["ablation", "mean_auc", "std_auc", "delta_vs_full"]
    if list(tbl.columns) != req_cols:
        errors.append(f"Expected ablation_table columns {req_cols}, got {list(tbl.columns)}.")
    elif len(tbl) != 5:
        errors.append(f"Expected 5 rows (full_model + 4 drop_<group> rows), got {len(tbl)}.")
    else:
        got_names = set(tbl["ablation"].astype(str).tolist())
        exp_names = set(exp_tbl["ablation"].tolist())
        if got_names != exp_names:
            errors.append(f"Expected ablation row names {sorted(exp_names)}, got {sorted(got_names)}.")
        else:
            for _, row in tbl.iterrows():
                name = str(row["ablation"])
                erow = exp_by_name.loc[name]
                if not math.isclose(float(row["mean_auc"]), float(erow["mean_auc"]), abs_tol=1e-3):
                    errors.append(f"[{name}] mean_auc: expected {erow['mean_auc']}, got {row['mean_auc']}.")
                if not math.isclose(float(row["std_auc"]), float(erow["std_auc"]), abs_tol=1e-3):
                    errors.append(f"[{name}] std_auc: expected {erow['std_auc']}, got {row['std_auc']}.")
                if not math.isclose(float(row["delta_vs_full"]), float(erow["delta_vs_full"]), abs_tol=1e-3):
                    errors.append(f"[{name}] delta_vs_full: expected {erow['delta_vs_full']}, got {row['delta_vs_full']}.")

    if not _is_number(got["full_auc"]) or not math.isclose(float(got["full_auc"]), exp["full_auc"], abs_tol=1e-3):
        errors.append(f"full_auc: expected {exp['full_auc']}, got {got['full_auc']!r}.")

    got_crit = got["most_critical_group"]
    if got_crit == f"drop_{exp['most_critical_group']}":
        errors.append(
            f"most_critical_group should be the bare group name {exp['most_critical_group']!r}, not {got_crit!r} (strip the `'drop_'` prefix!)."
        )
    elif not isinstance(got_crit, str) or got_crit != exp["most_critical_group"]:
        errors.append(f"most_critical_group: expected {exp['most_critical_group']!r}, got {got_crit!r}.")

    if errors:
        return CheckResult(
            "ch33_exam_03",
            False,
            "failed",
            "Exam Task 33.3 — Check MCH Cluster-CV Ablation Study Report",
            "Clean `mch_df`, run 5-fold `GroupKFold` ablation over `demographics`, `care_access`, `socioeconomic`, and `nutrition`, and report the group with the most negative `delta_vs_full`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch33_exam_03",
        True,
        "passed",
        "Exam Task 33.3 Passed (35/35 pts) — MCH Cluster-CV Ablation Study Verified!",
        f"Verified full cluster-CV ROC-AUC ({exp['full_auc']}), all 4 feature-group ablations, and identified '{exp['most_critical_group']}' (delta = -0.0597) as the most critical domain.",
        points_earned=35,
        points_possible=35,
        tip="Socioeconomic wealth quintile explains nearly 6 full points of cluster-held-out ROC-AUC in child stunting—an insight only systematic ablation proves quantitatively!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch33_exam_03",
        chapter="ch33",
        title="Exam 33.3: Cluster-CV MCH Feature-Group Ablation Report (mch_ablation_study_report)",
        book_section="Chapter 33 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Work on `mch_df.copy()`: standardize `residence`, replace `anc_visits == 999` with `np.nan`, derive `is_urban = (work['residence'] == 'Urban').astype(int)`, map `wealth_quintile` (`Poorest: 1 .. Richest: 5`) to `wealth_rank`, and `.dropna()` across the 5 features + `child_stunted` + `cluster_id`.",
            2: "Evaluate `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=500))])` with `GroupKFold(n_splits=5)` grouped by `cluster_id` on `'full_model'` and each `'drop_<group>'` subset.",
            3: "drops = tbl[tbl['ablation'] != 'full_model']\ncritical = str(drops.loc[drops['delta_vs_full'].idxmin(), 'ablation']).removeprefix('drop_')\nreturn {'full_auc': round(full_auc, 4), 'ablation_table': tbl, 'most_critical_group': critical}",
        },
        verifier=_verify_exam_03,
    )
)
