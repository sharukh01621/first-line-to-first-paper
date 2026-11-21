"""Chapter 26 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch26_todo_01..06) and the Part V Capstone Coding Exam
(ch26_exam_01..03) for Production ML Workflows — Scikit-Learn Pipelines,
ColumnTransformer, Nested Cross-Validation, and Hyperparameter Tuning.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn/Joblib inside the check
function, validates return structures, leakage-free pipeline behavior, and
non-mutation, while letting NotImplementedError propagate cleanly for starter
stubs. Any joblib check runs strictly inside tempfile.TemporaryDirectory().
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


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
        f"Calling {call_str} returned None instead of the expected ML result.",
        details=[
            "Remember: print() only displays text; functions must end with `return ...`.",
            f"Ensure {call_str} explicitly returns the required object or dictionary.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Production ML helpers return fitted/unfitted estimators, dicts, or DataFrames for downstream pipelines.",
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


def _make_ref_sentinel_cls() -> Any:
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin

    class _RefSentinelToNaN(BaseEstimator, TransformerMixin):
        def __init__(self, sentinel=999):
            self.sentinel = sentinel

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                return X.copy().replace(self.sentinel, np.nan)
            arr = np.asarray(X, dtype=float).copy()
            arr[arr == self.sentinel] = np.nan
            return arr

    return _RefSentinelToNaN


def _make_ref_dhs_pipeline(num_cols: list[str], cat_cols: list[str], model: Any) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    sentinel_cls = _make_ref_sentinel_cls()
    num_pipe = Pipeline([
        ("sentinel", sentinel_cls(999)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


# ---------------------------------------------------------------------------
# GUIDED TO-DO 26.1: SentinelToNaNTransformer
# ---------------------------------------------------------------------------
def _verify_todo_01(cls: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin, clone
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if not isinstance(cls, type):
        return CheckResult(
            "ch26_todo_01",
            False,
            "failed",
            "To-Do 26.1 — Pass the Class Name `SentinelToNaNTransformer`",
            f"Expected a class (type), got {type(cls).__name__} ({cls!r}).",
            details=[
                "Pass the class itself without parentheses: tutor.check('ch26_todo_01', SentinelToNaNTransformer)",
            ],
            points_earned=0,
            points_possible=10,
        )

    if not (issubclass(cls, BaseEstimator) and issubclass(cls, TransformerMixin)):
        return CheckResult(
            "ch26_todo_01",
            False,
            "failed",
            "To-Do 26.1 — Inherit From `BaseEstimator` and `TransformerMixin`",
            "Define `class SentinelToNaNTransformer(BaseEstimator, TransformerMixin):` so scikit-learn Pipelines, clone(), and GridSearchCV recognize it.",
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []
    try:
        tr_default = cls()
        tr_custom = cls(sentinel=-99)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch26_todo_01",
            False,
            "failed",
            "To-Do 26.1 — Error Instantiating `SentinelToNaNTransformer`",
            f"Calling SentinelToNaNTransformer() or SentinelToNaNTransformer(sentinel=-99) raised {type(exc).__name__}: {exc}",
            details=["Implement `def __init__(self, sentinel=999): self.sentinel = sentinel`."],
            points_earned=0,
            points_possible=10,
        )

    if getattr(tr_default, "sentinel", None) != 999:
        errors.append(f"Default `sentinel` attribute should be 999, got {getattr(tr_default, 'sentinel', None)!r}.")
    if getattr(tr_custom, "sentinel", None) != -99:
        errors.append(f"Custom `sentinel` attribute should be -99, got {getattr(tr_custom, 'sentinel', None)!r}.")

    try:
        cloned = clone(tr_custom)
        if getattr(cloned, "sentinel", None) != -99:
            errors.append("sklearn.base.clone() did not preserve `sentinel=-99` — store `self.sentinel = sentinel` unchanged in `__init__`.")
    except Exception as exc:
        errors.append(f"sklearn.base.clone() failed: {type(exc).__name__}: {exc}")

    df_orig = pd.DataFrame({"anc_visits": [3, 999, 5, 999], "hb": [11.2, 10.8, 999.0, 12.1]})
    df_copy = df_orig.copy(deep=True)
    arr_orig = np.array([[3, 11], [999, 10], [4, 999]])
    arr_copy = arr_orig.copy()
    arr_custom = np.array([[1.0, -99.0], [-99.0, 2.5]])

    try:
        fitted_self = tr_default.fit(df_copy)
        df_out = tr_default.transform(df_copy)
        arr_out = tr_default.fit_transform(arr_copy)
        arr_custom_out = tr_custom.fit_transform(arr_custom.copy())
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch26_todo_01",
            False,
            "failed",
            "To-Do 26.1 — Runtime Error in `fit` / `transform`",
            f"Transformer failed on DataFrame/ndarray input with {type(exc).__name__}: {exc}",
            details=[
                "Handle DataFrame via `X.copy().replace(self.sentinel, np.nan)`.",
                "Handle integer/float ndarray via `arr = np.asarray(X, dtype=float).copy(); arr[arr == self.sentinel] = np.nan`.",
            ],
            points_earned=0,
            points_possible=10,
        )

    if fitted_self is not tr_default:
        errors.append("`.fit(X, y=None)` must return `self` so `fit_transform()` and `Pipeline` work.")

    if not df_copy.equals(df_orig):
        errors.append("DataFrame input `X` was mutated in place! Always transform a `.copy()`.")
    if not np.array_equal(arr_copy, arr_orig):
        errors.append("NumPy ndarray input `X` was mutated in place! Always transform a `.copy()`.")

    if df_out is None:
        return _none_return_trap("ch26_todo_01", "To-Do 26.1", "transformer.transform(df)", 10)
    if arr_out is None:
        return _none_return_trap("ch26_todo_01", "To-Do 26.1", "transformer.transform(ndarray)", 10)

    df_vals = np.asarray(df_out, dtype=float)
    exp_df_nan = np.array([[False, False], [True, False], [False, True], [True, False]])
    if df_vals.shape != (4, 2) or not np.array_equal(np.isnan(df_vals), exp_df_nan):
        errors.append(f"DataFrame sentinel replacement mismatch: got NaN mask {np.isnan(df_vals).tolist()}.")
    elif not math.isclose(float(df_vals[0, 0]), 3.0) or not math.isclose(float(df_vals[3, 1]), 12.1):
        errors.append("Non-sentinel DataFrame values were altered unexpectedly.")

    arr_vals = np.asarray(arr_out, dtype=float)
    exp_arr_nan = np.array([[False, False], [True, False], [False, True]])
    if arr_vals.shape != (3, 2) or not np.array_equal(np.isnan(arr_vals), exp_arr_nan):
        errors.append(f"ndarray sentinel replacement mismatch: got NaN mask {np.isnan(arr_vals).tolist()}.")

    custom_vals = np.asarray(arr_custom_out, dtype=float)
    exp_custom_nan = np.array([[False, True], [True, False]])
    if not np.array_equal(np.isnan(custom_vals), exp_custom_nan):
        errors.append("Custom `sentinel=-99` did not replace -99 with np.nan (did you hardcode 999 instead of `self.sentinel`?).")

    try:
        pipe = Pipeline([("sentinel", cls(999)), ("imp", SimpleImputer(strategy="median"))])
        imp_out = pipe.fit_transform(df_orig)
        if np.isnan(imp_out).any():
            errors.append("Pipeline chaining SentinelToNaNTransformer -> SimpleImputer still contained NaNs.")
    except Exception as exc:
        errors.append(f"Failed when chained inside sklearn Pipeline: {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch26_todo_01",
            False,
            "failed",
            "To-Do 26.1 — Check `SentinelToNaNTransformer` Implementation",
            "Inherit from `(BaseEstimator, TransformerMixin)`, store `self.sentinel = sentinel`, return `self` in `fit`, and return a non-mutated copy with `self.sentinel` replaced by `np.nan`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch26_todo_01",
        True,
        "passed",
        "To-Do 26.1 Passed — Custom `SentinelToNaNTransformer` Verified!",
        "Passed BaseEstimator/TransformerMixin inheritance, sklearn clone(), DataFrame & integer ndarray non-mutation, custom sentinel=-99, and Pipeline chaining.",
        tip="Never alter `__init__` parameters (like casting or mutating them) in sklearn estimators — `clone()` relies on exact parameter names.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_todo_01",
        chapter="ch26",
        title="To-Do 26.1: Custom Survey Sentinel Transformer (SentinelToNaNTransformer)",
        book_section="Section 26.3.1 (Custom Scikit-Learn Transformers)",
        points=10,
        hints={
            1: "Subclass `(BaseEstimator, TransformerMixin)`. In `__init__(self, sentinel=999)`, store `self.sentinel = sentinel`. In `fit(self, X, y=None)`, simply `return self`.",
            2: "In `transform(self, X)`, branch on `isinstance(X, pd.DataFrame)`: return `X.copy().replace(self.sentinel, np.nan)` for DataFrames, or convert to float copy `arr = np.asarray(X, dtype=float).copy(); arr[arr == self.sentinel] = np.nan; return arr` for ndarrays.",
            3: "class SentinelToNaNTransformer(BaseEstimator, TransformerMixin):\n    def __init__(self, sentinel=999):\n        self.sentinel = ___\n    def fit(self, X, y=None):\n        return ___\n    def transform(self, X):\n        if isinstance(X, pd.DataFrame):\n            return X.copy().replace(self.sentinel, np.nan)\n        arr = np.asarray(X, dtype=float).copy()\n        arr[arr == self.sentinel] = np.nan\n        return arr",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 26.2: build_mixed_preprocessor(num_cols, cat_cols)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer

    bad = _check_callable("ch26_todo_02", "To-Do 26.2", fn, "build_mixed_preprocessor", 10)
    if bad is not None:
        return bad

    num_cols = ["age", "hb"]
    cat_cols = ["residence", "wealth"]
    try:
        ct = fn(num_cols, cat_cols)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch26_todo_02",
            False,
            "failed",
            "To-Do 26.2 — Runtime Error in `build_mixed_preprocessor`",
            f"Calling build_mixed_preprocessor(num_cols, cat_cols) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if ct is None:
        return _none_return_trap("ch26_todo_02", "To-Do 26.2", "build_mixed_preprocessor(num_cols, cat_cols)", 10)

    if not isinstance(ct, ColumnTransformer):
        return CheckResult(
            "ch26_todo_02",
            False,
            "failed",
            "To-Do 26.2 — Return a `ColumnTransformer`",
            f"Expected a sklearn ColumnTransformer, got {type(ct).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    X_train = pd.DataFrame(
        {
            "age": [22.0, 28.0, np.nan, 35.0, 24.0, 31.0],
            "hb": [10.5, np.nan, 11.5, 12.5, 10.0, 12.0],
            "residence": ["Rural", "Urban", None, "Rural", "Rural", "Urban"],
            "wealth": ["Poorest", "Richest", "Middle", "Poorest", None, "Middle"],
        }
    )
    X_test = pd.DataFrame(
        {
            "age": [np.nan, 26.0],
            "hb": [11.5, np.nan],
            "residence": ["SemiUrban", None],
            "wealth": ["UltraRich", "Middle"],
        }
    )

    errors: list[str] = []
    try:
        Z_train = ct.fit_transform(X_train)
    except Exception as exc:
        return CheckResult(
            "ch26_todo_02",
            False,
            "failed",
            "To-Do 26.2 — Failed During `fit_transform` on Mixed Missing Data",
            f"fit_transform raised {type(exc).__name__}: {exc}",
            details=["Chain SimpleImputer before StandardScaler (numeric) and OneHotEncoder (categorical)."],
            points_earned=0,
            points_possible=10,
        )

    try:
        Z_test = ct.transform(X_test)
    except ValueError as exc:
        return CheckResult(
            "ch26_todo_02",
            False,
            "failed",
            "To-Do 26.2 — Crashed on Unseen Test Category!",
            f"transform(X_test) raised ValueError: {exc}",
            details=["Pass `handle_unknown='ignore'` to `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`."],
            points_earned=0,
            points_possible=10,
        )
    except Exception as exc:
        return CheckResult(
            "ch26_todo_02",
            False,
            "failed",
            "To-Do 26.2 — Runtime Error During `transform(X_test)`",
            f"transform(X_test) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if not isinstance(Z_train, np.ndarray):
        errors.append(
            f"Expected a dense NumPy ndarray (got {type(Z_train).__name__}). Pass `sparse_output=False` to `OneHotEncoder`!"
        )
    else:
        if Z_train.shape != (6, 7):
            errors.append(f"Expected transformed shape (6, 7) (2 numeric + 2 residence + 3 wealth), got {Z_train.shape}.")
        if np.isnan(Z_train).any() or np.isnan(Z_test).any():
            errors.append("Transformed output still contains NaNs — verify both numeric and categorical imputers.")

        # Reference comparison on numeric imputation (median) + standardization
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        ref_ct = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ])
        ref_train = ref_ct.fit_transform(X_train)
        ref_test = ref_ct.transform(X_test)
        if Z_train.shape == ref_train.shape and not np.allclose(Z_train, ref_train, atol=1e-4):
            errors.append(
                "Transformed training matrix does not match reference (median impute + StandardScaler on num_cols, most_frequent impute + OneHotEncoder on cat_cols)."
            )
        if Z_test.shape == ref_test.shape and not np.allclose(Z_test, ref_test, atol=1e-4):
            errors.append(
                "Transformed test matrix with unseen categories does not match reference output (unseen categories should encode to all zeros)."
            )

    if errors:
        return CheckResult(
            "ch26_todo_02",
            False,
            "failed",
            "To-Do 26.2 — Check `ColumnTransformer` Pipelines & Options",
            "Combine a numeric Pipeline (`SimpleImputer(strategy='median')` + `StandardScaler()`) and categorical Pipeline (`SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch26_todo_02",
        True,
        "passed",
        "To-Do 26.2 Passed — Mixed-Type `ColumnTransformer` Verified!",
        "Imputed missing numeric (median) and categorical (most_frequent) values, scaled numeric features, and safely zero-encoded unseen test categories into a dense matrix.",
        tip="Always set `handle_unknown='ignore'` in production OneHotEncoders so new clinic districts or rare categories never crash deployed inference.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_todo_02",
        chapter="ch26",
        title="To-Do 26.2: Mixed-Type Preprocessor (build_mixed_preprocessor)",
        book_section="Section 26.3.2 (ColumnTransformer for Mixed Clinical Data)",
        points=10,
        hints={
            1: "Create two `Pipeline` objects: `num_pipe` with `SimpleImputer(strategy='median')` and `StandardScaler()`, and `cat_pipe` with `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.",
            2: "Wrap both pipelines in `ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])` and return it.",
            3: "def build_mixed_preprocessor(num_cols, cat_cols):\n    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\n    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='___', sparse_output=False))])\n    return ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 26.3: cross_validate_pipeline(pipe, X, y, cv=5, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch26_todo_03", "To-Do 26.3", fn, "cross_validate_pipeline", 10)
    if bad is not None:
        return bad

    X_syn, y_syn = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        weights=[0.75, 0.25],
        random_state=7,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))])

    expected_keys = ["mean_roc_auc", "std_roc_auc", "fold_scores"]
    errors: list[str] = []

    for cv_k, seed in [(5, 42), (4, 99)]:
        try:
            got = fn(pipe, X_syn, y_syn, cv=cv_k, seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch26_todo_03",
                False,
                "failed",
                "To-Do 26.3 — Runtime Error in `cross_validate_pipeline`",
                f"cross_validate_pipeline(pipe, X, y, cv={cv_k}, seed={seed}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch26_todo_03", "To-Do 26.3", "cross_validate_pipeline(pipe, X, y)", 10)

        key_err = _check_dict_keys("ch26_todo_03", "To-Do 26.3", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
        ref_scores = cross_val_score(pipe, X_syn, y_syn, cv=skf, scoring="roc_auc")
        exp_mean = round(float(ref_scores.mean()), 4)
        exp_std = round(float(ref_scores.std()), 4)
        exp_folds = [round(float(s), 4) for s in ref_scores]

        if not _is_number(got["mean_roc_auc"]) or not math.isclose(float(got["mean_roc_auc"]), exp_mean, abs_tol=1e-3):
            errors.append(f"[cv={cv_k}, seed={seed}] mean_roc_auc: expected {exp_mean}, got {got['mean_roc_auc']!r}.")
        if not _is_number(got["std_roc_auc"]) or not math.isclose(float(got["std_roc_auc"]), exp_std, abs_tol=1e-3):
            errors.append(f"[cv={cv_k}, seed={seed}] std_roc_auc: expected {exp_std}, got {got['std_roc_auc']!r}.")
        if not isinstance(got["fold_scores"], list) or len(got["fold_scores"]) != cv_k:
            errors.append(f"[cv={cv_k}, seed={seed}] fold_scores should be a Python list of length {cv_k}, got {got['fold_scores']!r}.")
        elif not all(_is_number(a) and math.isclose(float(a), b, abs_tol=1e-4) for a, b in zip(got["fold_scores"], exp_folds)):
            errors.append(
                f"[cv={cv_k}, seed={seed}] fold_scores: expected {exp_folds}, got {got['fold_scores']!r} "
                "(ensure `StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)` and `scoring='roc_auc'`)."
            )

    if errors:
        return CheckResult(
            "ch26_todo_03",
            False,
            "failed",
            "To-Do 26.3 — Check StratifiedKFold Parameters & ROC-AUC Scoring",
            "Use `StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)` with `cross_val_score(..., scoring='roc_auc')`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch26_todo_03",
        True,
        "passed",
        "To-Do 26.3 Passed — Stratified K-Fold Pipeline Cross-Validation Verified!",
        "Verified deterministic StratifiedKFold shuffling, ROC-AUC scoring, fold-level list, mean, and standard deviation across multiple seeds.",
        tip="Passing a full Pipeline into `cross_val_score` guarantees scalers and imputers refit fresh on every training fold—zero test leakage!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_todo_03",
        chapter="ch26",
        title="To-Do 26.3: Stratified K-Fold Pipeline CV (cross_validate_pipeline)",
        book_section="Section 26.3.3 (Stratified vs. Grouped vs. Temporal CV)",
        points=10,
        hints={
            1: "Create `skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)` and run `scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')`.",
            2: "Return `{'mean_roc_auc': round(float(scores.mean()), 4), 'std_roc_auc': round(float(scores.std()), 4), 'fold_scores': [round(float(s), 4) for s in scores]}`.",
            3: "def cross_validate_pipeline(pipe, X, y, cv=5, seed=42):\n    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=___)\n    scores = cross_val_score(pipe, X, y, cv=skf, scoring='___')\n    return {\n        'mean_roc_auc': round(float(scores.mean()), 4),\n        'std_roc_auc': round(float(scores.std()), 4),\n        'fold_scores': [round(float(s), 4) for s in scores],\n    }",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 26.4: group_cv_evaluation(pipe, X, y, groups, n_splits=5)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch26_todo_04", "To-Do 26.4", fn, "group_cv_evaluation", 10)
    if bad is not None:
        return bad

    X_syn, y_syn = make_classification(
        n_samples=150,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=11,
    )
    groups = np.repeat(np.arange(15), 10)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))])

    expected_keys = ["mean_roc_auc", "std_roc_auc", "n_splits"]
    errors: list[str] = []

    for k in (5, 3):
        try:
            got = fn(pipe, X_syn, y_syn, groups, n_splits=k)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch26_todo_04",
                False,
                "failed",
                "To-Do 26.4 — Runtime Error in `group_cv_evaluation`",
                f"group_cv_evaluation(..., n_splits={k}) raised {type(exc).__name__}: {exc}",
                details=["Remember to pass `groups=groups` into `cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring='roc_auc')`!"],
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch26_todo_04", "To-Do 26.4", "group_cv_evaluation(pipe, X, y, groups)", 10)

        key_err = _check_dict_keys("ch26_todo_04", "To-Do 26.4", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        gkf = GroupKFold(n_splits=k)
        ref_scores = cross_val_score(pipe, X_syn, y_syn, groups=groups, cv=gkf, scoring="roc_auc")
        kf_scores = cross_val_score(pipe, X_syn, y_syn, cv=KFold(n_splits=k), scoring="roc_auc")
        exp_mean = round(float(ref_scores.mean()), 4)
        exp_std = round(float(ref_scores.std()), 4)

        if not _is_number(got["mean_roc_auc"]) or not math.isclose(float(got["mean_roc_auc"]), exp_mean, abs_tol=1e-3):
            extra = ""
            if _is_number(got["mean_roc_auc"]) and math.isclose(float(got["mean_roc_auc"]), round(float(kf_scores.mean()), 4), abs_tol=1e-3):
                extra = " (Looks like plain KFold instead of GroupKFold!)"
            errors.append(f"[n_splits={k}] mean_roc_auc: expected {exp_mean}, got {got['mean_roc_auc']!r}{extra}")
        if not _is_number(got["std_roc_auc"]) or not math.isclose(float(got["std_roc_auc"]), exp_std, abs_tol=1e-3):
            errors.append(f"[n_splits={k}] std_roc_auc: expected {exp_std}, got {got['std_roc_auc']!r}.")
        if not isinstance(got["n_splits"], int) or isinstance(got["n_splits"], bool) or got["n_splits"] != k:
            errors.append(f"[n_splits={k}] n_splits: expected int {k}, got {got['n_splits']!r}.")

    if errors:
        return CheckResult(
            "ch26_todo_04",
            False,
            "failed",
            "To-Do 26.4 — Check `GroupKFold` & `groups=` Forwarding",
            "Use `GroupKFold(n_splits=n_splits)` and pass `groups=groups` to `cross_val_score(..., scoring='roc_auc')`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch26_todo_04",
        True,
        "passed",
        "To-Do 26.4 Passed — Cluster-Aware `GroupKFold` Evaluation Verified!",
        "Verified non-overlapping group splits across n_splits=5 and n_splits=3 with exact mean/std ROC-AUC reporting.",
        tip="In DHS surveys and multi-center clinical trials, GroupKFold prevents households from the same village or patients from the same hospital from leaking across folds.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_todo_04",
        chapter="ch26",
        title="To-Do 26.4: Cluster-Aware GroupKFold CV (group_cv_evaluation)",
        book_section="Section 26.3.3 (Stratified vs. Grouped vs. Temporal CV)",
        points=10,
        hints={
            1: "Instantiate `gkf = GroupKFold(n_splits=n_splits)` and pass BOTH `groups=groups` and `cv=gkf` to `cross_val_score(..., scoring='roc_auc')`.",
            2: "Return `{'mean_roc_auc': round(float(scores.mean()), 4), 'std_roc_auc': round(float(scores.std()), 4), 'n_splits': int(n_splits)}`.",
            3: "def group_cv_evaluation(pipe, X, y, groups, n_splits=5):\n    gkf = GroupKFold(n_splits=n_splits)\n    scores = cross_val_score(pipe, X, y, groups=___, cv=gkf, scoring='roc_auc')\n    return {\n        'mean_roc_auc': round(float(scores.mean()), 4),\n        'std_roc_auc': round(float(scores.std()), 4),\n        'n_splits': int(n_splits),\n    }",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 26.5: temporal_cv_forecast_rmse(pipe, X, y, n_splits=4)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable("ch26_todo_05", "To-Do 26.5", fn, "temporal_cv_forecast_rmse", 10)
    if bad is not None:
        return bad

    epi = load_data("epi_surveillance", as_pandas=True)
    dhaka = epi[epi["district"] == "Dhaka"].sort_values(["year", "epi_week"]).reset_index(drop=True)
    X_ts = dhaka[["rainfall_mm", "mean_temp_c", "epi_week"]]
    y_ts = dhaka["dengue_cases"]
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])

    expected_keys = ["mean_rmse", "fold_rmses"]
    errors: list[str] = []

    for k in (4, 3):
        try:
            got = fn(pipe, X_ts, y_ts, n_splits=k)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch26_todo_05",
                False,
                "failed",
                "To-Do 26.5 — Runtime Error in `temporal_cv_forecast_rmse`",
                f"temporal_cv_forecast_rmse(pipe, X, y, n_splits={k}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch26_todo_05", "To-Do 26.5", "temporal_cv_forecast_rmse(pipe, X, y)", 10)

        key_err = _check_dict_keys("ch26_todo_05", "To-Do 26.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        tscv = TimeSeriesSplit(n_splits=k)
        rmses = -cross_val_score(pipe, X_ts, y_ts, cv=tscv, scoring="neg_root_mean_squared_error")
        exp_mean = round(float(rmses.mean()), 4)
        exp_folds = [round(float(r), 4) for r in rmses]

        if _is_number(got["mean_rmse"]) and float(got["mean_rmse"]) < 0:
            errors.append(
                f"[n_splits={k}] mean_rmse is negative ({got['mean_rmse']})! "
                "Negate `neg_root_mean_squared_error` scores so RMSE is positive."
            )
            continue

        if not _is_number(got["mean_rmse"]) or not math.isclose(float(got["mean_rmse"]), exp_mean, abs_tol=1e-3):
            errors.append(f"[n_splits={k}] mean_rmse: expected {exp_mean}, got {got['mean_rmse']!r}.")
        if not isinstance(got["fold_rmses"], list) or len(got["fold_rmses"]) != k:
            errors.append(f"[n_splits={k}] fold_rmses should be a list of length {k}, got {got['fold_rmses']!r}.")
        elif not all(_is_number(a) and math.isclose(float(a), b, abs_tol=1e-3) for a, b in zip(got["fold_rmses"], exp_folds)):
            errors.append(f"[n_splits={k}] fold_rmses: expected {exp_folds}, got {got['fold_rmses']!r}.")

    if errors:
        return CheckResult(
            "ch26_todo_05",
            False,
            "failed",
            "To-Do 26.5 — Check `TimeSeriesSplit` & Positive RMSE Conversion",
            "Run `TimeSeriesSplit(n_splits=n_splits)` with `scoring='neg_root_mean_squared_error'` and negate the returned scores so RMSE values are positive.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch26_todo_05",
        True,
        "passed",
        "To-Do 26.5 Passed — Expanding-Window `TimeSeriesSplit` Forecast RMSE Verified!",
        "Evaluated Dhaka weekly dengue forecasting chronologically without future-to-past look-ahead leakage.",
        tip="Never shuffle longitudinal surveillance series! TimeSeriesSplit always trains on past epi-weeks and evaluates strictly on future weeks.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_todo_05",
        chapter="ch26",
        title="To-Do 26.5: Temporal Expanding-Window CV (temporal_cv_forecast_rmse)",
        book_section="Section 26.3.3 (Stratified vs. Grouped vs. Temporal CV)",
        points=10,
        hints={
            1: "Create `tscv = TimeSeriesSplit(n_splits=n_splits)` and run `neg_scores = cross_val_score(pipe, X, y, cv=tscv, scoring='neg_root_mean_squared_error')`.",
            2: "Flip the sign (`rmses = -neg_scores`) so all RMSEs are positive, then return `{'mean_rmse': round(float(rmses.mean()), 4), 'fold_rmses': [round(float(r), 4) for r in rmses]}`.",
            3: "def temporal_cv_forecast_rmse(pipe, X, y, n_splits=4):\n    tscv = TimeSeriesSplit(n_splits=n_splits)\n    rmses = -cross_val_score(pipe, X, y, cv=tscv, scoring='___')\n    return {\n        'mean_rmse': round(float(rmses.mean()), 4),\n        'fold_rmses': [round(float(r), 4) for r in rmses],\n    }",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 26.6: save_and_reload_pipeline(pipe, model_path, X_sample)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch26_todo_06", "To-Do 26.6", fn, "save_and_reload_pipeline", 10)
    if bad is not None:
        return bad

    X_syn, y_syn = make_classification(n_samples=80, n_features=4, random_state=19)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=300))])
    pipe.fit(X_syn, y_syn)
    X_sample = X_syn[:8]
    expected_preds = pipe.predict(X_sample)
    expected_keys = ["file_bytes", "predictions_match", "reloaded_preds"]

    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "clinical_pipeline.joblib"
        try:
            got = fn(pipe, model_path, X_sample)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch26_todo_06",
                False,
                "failed",
                "To-Do 26.6 — Runtime Error in `save_and_reload_pipeline`",
                f"save_and_reload_pipeline(pipe, model_path, X_sample) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch26_todo_06", "To-Do 26.6", "save_and_reload_pipeline(pipe, path, X_sample)", 10)

        key_err = _check_dict_keys("ch26_todo_06", "To-Do 26.6", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if not model_path.exists():
            errors.append("No artifact file was created at `model_path` — call `joblib.dump(pipe, model_path)`.")
        else:
            actual_bytes = model_path.stat().st_size
            if not isinstance(got["file_bytes"], int) or isinstance(got["file_bytes"], bool) or got["file_bytes"] <= 0:
                errors.append(f"file_bytes must be a positive int, got {got['file_bytes']!r}.")
            elif got["file_bytes"] != actual_bytes:
                errors.append(f"file_bytes ({got['file_bytes']}) does not match on-disk size ({actual_bytes}).")

        if not _is_bool_like(got["predictions_match"]) or not bool(got["predictions_match"]):
            errors.append(f"predictions_match should be True after reloading the identical fitted pipeline, got {got['predictions_match']!r}.")

        try:
            reloaded_arr = np.asarray(got["reloaded_preds"])
            if reloaded_arr.shape != expected_preds.shape or not np.allclose(reloaded_arr, expected_preds):
                errors.append(f"reloaded_preds did not match `pipe.predict(X_sample)`: expected {expected_preds.tolist()}, got {reloaded_arr.tolist()}.")
        except Exception as exc:
            errors.append(f"reloaded_preds could not be compared to predictions: {exc}")

    if errors:
        return CheckResult(
            "ch26_todo_06",
            False,
            "failed",
            "To-Do 26.6 — Check `joblib.dump`, `joblib.load`, and Prediction Parity",
            "Save `pipe` to `model_path` via `joblib.dump`, reload with `joblib.load`, check `Path(model_path).stat().st_size`, and compare predictions on `X_sample`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch26_todo_06",
        True,
        "passed",
        "To-Do 26.6 Passed — Reproducible `joblib` Pipeline Serialization Verified!",
        "Serialized fitted Pipeline to disk (inside a clean temp directory), verified byte size, reloaded cleanly, and confirmed bit-exact prediction parity.",
        tip="Always serialize the ENTIRE fitted `Pipeline` (preprocessing + estimator together) — never save a bare classifier separately from its scaler!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_todo_06",
        chapter="ch26",
        title="To-Do 26.6: Pipeline Serialization Round-Trip (save_and_reload_pipeline)",
        book_section="Section 26.3.6 (Model Serialization with joblib)",
        points=10,
        hints={
            1: "Call `joblib.dump(pipe, model_path)`, then `reloaded = joblib.load(model_path)` and measure `Path(model_path).stat().st_size`.",
            2: "Compute `orig_preds = pipe.predict(X_sample)` and `reloaded_preds = reloaded.predict(X_sample)`, then check `bool(np.allclose(orig_preds, reloaded_preds))`.",
            3: "def save_and_reload_pipeline(pipe, model_path, X_sample):\n    joblib.dump(pipe, model_path)\n    reloaded = joblib.load(___)\n    orig_preds = pipe.predict(X_sample)\n    reloaded_preds = reloaded.predict(X_sample)\n    return {\n        'file_bytes': int(Path(model_path).stat().st_size),\n        'predictions_match': bool(np.allclose(orig_preds, reloaded_preds)),\n        'reloaded_preds': reloaded_preds,\n    }",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 26.1 (30 pts): build_dhs_end_to_end_pipeline(num_cols, cat_cols, model)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    bad = _check_callable("ch26_exam_01", "Exam Task 26.1", fn, "build_dhs_end_to_end_pipeline", 30)
    if bad is not None:
        return bad

    num_cols = ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]
    cat_cols = ["residence", "wealth_quintile"]
    try:
        pipe = fn(num_cols, cat_cols, LogisticRegression(max_iter=500))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch26_exam_01",
            False,
            "failed",
            "Exam Task 26.1 — Runtime Error Building Pipeline",
            f"build_dhs_end_to_end_pipeline(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if pipe is None:
        return _none_return_trap("ch26_exam_01", "Exam Task 26.1", "build_dhs_end_to_end_pipeline(num_cols, cat_cols, model)", 30)

    if not isinstance(pipe, Pipeline):
        return CheckResult(
            "ch26_exam_01",
            False,
            "failed",
            "Exam Task 26.1 — Must Return a `sklearn.pipeline.Pipeline`",
            f"Expected a Pipeline, got {type(pipe).__name__}.",
            points_earned=0,
            points_possible=30,
        )

    errors: list[str] = []
    if "prep" not in pipe.named_steps or not isinstance(pipe.named_steps["prep"], ColumnTransformer):
        errors.append("Pipeline must include a `('prep', ColumnTransformer(...))` step.")
    if "model" not in pipe.named_steps or not isinstance(pipe.named_steps["model"], LogisticRegression):
        errors.append("Pipeline must end with a `('model', model)` step.")

    if errors:
        return CheckResult(
            "ch26_exam_01",
            False,
            "failed",
            "Exam Task 26.1 — Check Pipeline Step Names (`'prep'` and `'model'`)",
            "Return `Pipeline([('prep', preprocessor), ('model', model)])`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    rng = np.random.default_rng(42)
    n_tr = 60
    X_tr = pd.DataFrame(
        {
            "maternal_age_years": rng.integers(18, 40, size=n_tr).astype(float),
            "anc_visits": rng.integers(0, 9, size=n_tr).astype(float),
            "maternal_hb_g_dl": rng.normal(11.3, 1.2, size=n_tr),
            "residence": rng.choice(["Rural", "Urban"], size=n_tr),
            "wealth_quintile": rng.choice(["Poorest", "Middle", "Richest"], size=n_tr),
        }
    )
    X_tr.loc[[2, 17, 33], "anc_visits"] = 999.0
    X_tr.loc[[5, 22], "maternal_hb_g_dl"] = np.nan
    X_tr.loc[[10], "wealth_quintile"] = None
    y_tr = (X_tr["maternal_hb_g_dl"].fillna(11.3) < 11.0).astype(int)

    # Test DataFrame with sentinel 999, NaN, and UNSEEN categories!
    X_te = pd.DataFrame(
        {
            "maternal_age_years": [25.0, 25.0, 30.0],
            "anc_visits": [999.0, np.nan, 4.0],
            "maternal_hb_g_dl": [10.8, 10.8, np.nan],
            "residence": ["Rural", "Rural", "PeriUrbanNew"],
            "wealth_quintile": ["Poorest", "Poorest", "UnseenTier"],
        }
    )

    try:
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
    except Exception as exc:
        return CheckResult(
            "ch26_exam_01",
            False,
            "failed",
            "Exam Task 26.1 — Pipeline Crashed on Sentinel 999 / NaN / Unseen Categories",
            f"fit/predict_proba failed with {type(exc).__name__}: {exc}",
            details=[
                "Chain SentinelToNaNTransformer(999) -> SimpleImputer(strategy='median') -> StandardScaler() on num_cols.",
                "Chain SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False) on cat_cols.",
            ],
            points_earned=0,
            points_possible=30,
        )

    # Row 0 (anc_visits=999) and Row 1 (anc_visits=NaN) are otherwise identical:
    # If SentinelToNaNTransformer(999) ran BEFORE SimpleImputer, their transformed rows MUST be identical!
    if not math.isclose(float(probs[0]), float(probs[1]), abs_tol=1e-6):
        errors.append(
            f"Row with anc_visits=999 predicted {probs[0]:.4f} while identical row with anc_visits=NaN predicted {probs[1]:.4f}! "
            "Place `SentinelToNaNTransformer(999)` BEFORE `SimpleImputer(strategy='median')` inside the numeric pipeline."
        )

    ref_pipe = _make_ref_dhs_pipeline(num_cols, cat_cols, LogisticRegression(max_iter=500))
    ref_pipe.fit(X_tr, y_tr)
    ref_probs = ref_pipe.predict_proba(X_te)[:, 1]
    if not np.allclose(probs, ref_probs, atol=1e-4):
        errors.append(
            f"Predicted probabilities {np.round(probs, 4).tolist()} differed from reference {np.round(ref_probs, 4).tolist()}."
        )

    if errors:
        return CheckResult(
            "ch26_exam_01",
            False,
            "failed",
            "Exam Task 26.1 — Check Sentinel-to-NaN Ordering & Preprocessing Parity",
            "Verify exact step order: SentinelToNaNTransformer(999) -> SimpleImputer('median') -> StandardScaler() on numeric columns, and SimpleImputer('most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False) on categorical columns.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch26_exam_01",
        True,
        "passed",
        "Exam Task 26.1 Passed (30/30 pts) — Leak-Free DHS End-to-End Pipeline Verified!",
        "Confirmed `('prep', ColumnTransformer)` + `('model', model)` structure, exact 999->NaN->median imputation parity, and graceful unseen-category handling.",
        points_earned=30,
        points_possible=30,
        tip="Notice how row-0 (anc_visits=999) and row-1 (anc_visits=NaN) produced identical probabilities — proof that sentinel cleaning lives cleanly inside the Pipeline!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_exam_01",
        chapter="ch26",
        title="Exam 26.1: Leak-Free DHS End-to-End Pipeline (build_dhs_end_to_end_pipeline)",
        book_section="Chapter 26 Part V Capstone Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Build a numeric Pipeline chaining `SentinelToNaNTransformer(999)` -> `SimpleImputer(strategy='median')` -> `StandardScaler()` and a categorical Pipeline chaining `SimpleImputer(strategy='most_frequent')` -> `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.",
            2: "Combine both inside `prep = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])`, then return `Pipeline([('prep', prep), ('model', model)])`.",
            3: "def build_dhs_end_to_end_pipeline(num_cols, cat_cols, model):\n    num_pipe = Pipeline([('sentinel', SentinelToNaNTransformer(999)), ('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\n    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])\n    prep = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])\n    return Pipeline([('___', prep), ('___', model)])",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 26.2 (35 pts): run_nested_cv_audit(...)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from ..datasets import load_data

    bad = _check_callable("ch26_exam_02", "Exam Task 26.2", fn, "run_nested_cv_audit", 35)
    if bad is not None:
        return bad

    mch = load_data("mch_survey_sample", as_pandas=True).copy()
    mch["residence"] = mch["residence"].astype(str).str.strip().str.title()
    num_cols = ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]
    cat_cols = ["residence", "wealth_quintile"]
    X = mch[num_cols + cat_cols]
    y = mch["child_stunted"]
    pipe = _make_ref_dhs_pipeline(num_cols, cat_cols, LogisticRegression(max_iter=500))
    param_grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}

    expected_keys = ["outer_mean_auc", "outer_std_auc", "outer_scores", "fold_best_params"]
    errors: list[str] = []

    try:
        got = fn(pipe, param_grid, X, y, outer_splits=4, inner_splits=3, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch26_exam_02",
            False,
            "failed",
            "Exam Task 26.2 — Runtime Error in `run_nested_cv_audit`",
            f"run_nested_cv_audit(...) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `.iloc[idx]` when slicing DataFrames/Series inside the outer CV loop.",
                "Inside each outer fold, fit `GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc')` on `(X_train, y_train)` only.",
            ],
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap("ch26_exam_02", "Exam Task 26.2", "run_nested_cv_audit(pipe, param_grid, X, y)", 35)

    key_err = _check_dict_keys("ch26_exam_02", "Exam Task 26.2", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    ref_scores = []
    ref_params = []
    for tr_idx, te_idx in outer_cv.split(X, y):
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="roc_auc")
        gs.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        probs = gs.best_estimator_.predict_proba(X.iloc[te_idx])[:, 1]
        ref_scores.append(round(float(roc_auc_score(y.iloc[te_idx], probs)), 4))
        ref_params.append(gs.best_params_)

    exp_mean = round(float(np.mean(ref_scores)), 4)
    exp_std = round(float(np.std(ref_scores)), 4)

    if not isinstance(got["outer_scores"], list) or len(got["outer_scores"]) != 4:
        errors.append(f"outer_scores must be a list of length 4, got {got['outer_scores']!r}.")
    elif not all(_is_number(a) and math.isclose(float(a), b, abs_tol=1e-3) for a, b in zip(got["outer_scores"], ref_scores)):
        errors.append(
            f"outer_scores mismatch: expected {ref_scores}, got {got['outer_scores']!r}. "
            "Check outer/inner StratifiedKFold(shuffle=True, random_state=seed) and evaluate `best_estimator_` on the untouched outer test fold."
        )

    if not _is_number(got["outer_mean_auc"]) or not math.isclose(float(got["outer_mean_auc"]), exp_mean, abs_tol=1e-3):
        errors.append(f"outer_mean_auc: expected {exp_mean}, got {got['outer_mean_auc']!r}.")
    if not _is_number(got["outer_std_auc"]) or not math.isclose(float(got["outer_std_auc"]), exp_std, abs_tol=1e-3):
        errors.append(f"outer_std_auc: expected {exp_std}, got {got['outer_std_auc']!r}.")

    if not isinstance(got["fold_best_params"], list) or len(got["fold_best_params"]) != 4:
        errors.append(f"fold_best_params must be a list of 4 dicts (one per outer fold), got {got['fold_best_params']!r}.")
    elif not all(isinstance(p, dict) and "model__C" in p for p in got["fold_best_params"]):
        errors.append(f"Each element of fold_best_params must be a dict like `gs.best_params_`, got {got['fold_best_params']!r}.")
    elif got["fold_best_params"] != ref_params:
        errors.append(f"fold_best_params mismatch: expected {ref_params}, got {got['fold_best_params']!r}.")

    if errors:
        return CheckResult(
            "ch26_exam_02",
            False,
            "failed",
            "Exam Task 26.2 — Check Nested Stratified CV Loop & Outer Test Scoring",
            "Run an outer `StratifiedKFold(outer_splits, shuffle=True, random_state=seed)` loop, tune `GridSearchCV` inside an inner `StratifiedKFold(inner_splits, shuffle=True, random_state=seed)`, and score `best_estimator_` on the held-out outer fold.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch26_exam_02",
        True,
        "passed",
        "Exam Task 26.2 Passed (35/35 pts) — True Nested Stratified CV Audit Verified!",
        f"Verified 4x3 Nested CV on MCH stunting (outer mean AUC={exp_mean}, std={exp_std}) with per-fold best hyperparameters logged cleanly.",
        points_earned=35,
        points_possible=35,
        tip="Notice how different outer folds picked different `model__C` values — Nested CV measures the true unbiased generalization of your entire tuning procedure!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_exam_02",
        chapter="ch26",
        title="Exam 26.2: True Nested Stratified CV Audit (run_nested_cv_audit)",
        book_section="Chapter 26 Part V Capstone Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Create `outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)`. Loop `for tr_idx, te_idx in outer_cv.split(X, y):` and slice train/test folds.",
            2: "Inside each outer fold, build `inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)`, fit `gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc')` on `(X_tr, y_tr)`, score `gs.best_estimator_` on `(X_te, y_te)` via `roc_auc_score`, and append `gs.best_params_`.",
            3: "def run_nested_cv_audit(pipe, param_grid, X, y, outer_splits=4, inner_splits=3, seed=42):\n    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)\n    outer_scores, fold_best_params = [], []\n    for tr_idx, te_idx in outer_cv.split(X, y):\n        X_tr = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]\n        X_te = X.iloc[te_idx] if hasattr(X, 'iloc') else X[te_idx]\n        y_tr = y.iloc[tr_idx] if hasattr(y, 'iloc') else y[tr_idx]\n        y_te = y.iloc[te_idx] if hasattr(y, 'iloc') else y[te_idx]\n        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)\n        gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='roc_auc')\n        gs.fit(X_tr, y_tr)\n        probs = gs.best_estimator_.predict_proba(X_te)[:, 1]\n        outer_scores.append(round(float(roc_auc_score(y_te, probs)), 4))\n        fold_best_params.append(gs.best_params_)\n    return {'outer_mean_auc': round(float(np.mean(outer_scores)), 4), 'outer_std_auc': round(float(np.std(outer_scores)), 4), 'outer_scores': outer_scores, 'fold_best_params': fold_best_params}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 26.3 (35 pts): benchmark_part5_capstone(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_benchmark_part5(mch_df: Any, seed: int = 42) -> dict[str, Any]:
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    num_cols = ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]
    cat_cols = ["residence", "wealth_quintile"]
    X = work[num_cols + cat_cols]
    y = work["child_stunted"]
    groups = work["cluster_id"]
    gkf = GroupKFold(n_splits=5)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=80, max_depth=6, random_state=seed
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_depth=4, random_state=seed
        ),
    }
    rows = []
    for name, mdl in models.items():
        pipe = _make_ref_dhs_pipeline(num_cols, cat_cols, mdl)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring="roc_auc")
        rows.append(
            {
                "model": name,
                "mean_auc": round(float(scores.mean()), 4),
                "std_auc": round(float(scores.std()), 4),
            }
        )
    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("mean_auc", ascending=False)
        .reset_index(drop=True)
    )
    return {"leaderboard": leaderboard, "winner": str(leaderboard.loc[0, "model"])}


def _verify_exam_03(fn: Any) -> CheckResult:
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch26_exam_03", "Exam Task 26.3", fn, "benchmark_part5_capstone", 35)
    if bad is not None:
        return bad

    mch_raw = load_data("mch_survey_sample", as_pandas=True)
    mch_copy = mch_raw.copy(deep=True)
    expected_keys = ["leaderboard", "winner"]

    try:
        got = fn(mch_copy, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch26_exam_03",
            False,
            "failed",
            "Exam Task 26.3 — Runtime Error in `benchmark_part5_capstone`",
            f"benchmark_part5_capstone(mch_df, seed=42) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap("ch26_exam_03", "Exam Task 26.3", "benchmark_part5_capstone(mch_df, seed=42)", 35)

    key_err = _check_dict_keys("ch26_exam_03", "Exam Task 26.3", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    errors: list[str] = []
    if not mch_copy.equals(mch_raw):
        errors.append("Input DataFrame `mch_df` was mutated in place! Start with `work = mch_df.copy()`.")

    lb = got["leaderboard"]
    if not isinstance(lb, pd.DataFrame):
        return CheckResult(
            "ch26_exam_03",
            False,
            "failed",
            "Exam Task 26.3 — `'leaderboard'` Must Be a Pandas DataFrame",
            f"Expected pd.DataFrame for 'leaderboard', got {type(lb).__name__}.",
            points_earned=0,
            points_possible=35,
        )

    exp = _ref_benchmark_part5(mch_raw, seed=42)
    exp_lb = exp["leaderboard"]

    required_cols = ["model", "mean_auc", "std_auc"]
    if list(lb.columns) != required_cols:
        errors.append(f"Expected leaderboard columns {required_cols}, got {list(lb.columns)}.")
    elif len(lb) != 3:
        errors.append(f"Expected 3 benchmarked models in leaderboard, got {len(lb)} rows.")
    else:
        aucs = [float(v) for v in lb["mean_auc"].tolist()]
        if aucs != sorted(aucs, reverse=True):
            errors.append(f"Leaderboard must be sorted descending by `mean_auc`, got {aucs}.")

        for idx in range(3):
            g_name = str(lb.iloc[idx]["model"])
            e_name = str(exp_lb.iloc[idx]["model"])
            g_mean = float(lb.iloc[idx]["mean_auc"])
            e_mean = float(exp_lb.iloc[idx]["mean_auc"])
            g_std = float(lb.iloc[idx]["std_auc"])
            e_std = float(exp_lb.iloc[idx]["std_auc"])
            if g_name != e_name:
                errors.append(f"Row {idx} model: expected {e_name!r}, got {g_name!r}.")
            if not math.isclose(g_mean, e_mean, abs_tol=1e-3):
                errors.append(
                    f"Row {idx} ({g_name}) mean_auc: expected {e_mean}, got {g_mean} "
                    "(verify `residence` casing cleanup, `GroupKFold(n_splits=5)` on `cluster_id`, and exact hyperparameters)."
                )
            if not math.isclose(g_std, e_std, abs_tol=1e-3):
                errors.append(f"Row {idx} ({g_name}) std_auc: expected {e_std}, got {g_std}.")

    if not isinstance(got["winner"], str) or got["winner"] != exp["winner"]:
        errors.append(f"winner: expected {exp['winner']!r}, got {got['winner']!r}.")

    if errors:
        return CheckResult(
            "ch26_exam_03",
            False,
            "failed",
            "Exam Task 26.3 — Check Part V Capstone Benchmark Leaderboard",
            "Clean `residence`, run 5-fold `GroupKFold` grouped by `cluster_id` across the 3 leak-free pipelines predicting `child_stunted`, and return `{'leaderboard': df_sorted, 'winner': top_model_name}`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch26_exam_03",
        True,
        "passed",
        "Exam Task 26.3 Passed (35/35 pts) — Part V Capstone Benchmark Verified!",
        "Benchmarked LogisticRegression (0.7168), RandomForestClassifier (0.6870), and HistGradientBoostingClassifier (0.6665) under strict cluster-level GroupKFold — LogisticRegression wins!",
        points_earned=35,
        points_possible=35,
        tip="Congratulations on completing the Part V Capstone! On modest-N DHS surveys, a clean regularized linear pipeline frequently beats heavier tree ensembles under honest cluster-level validation.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch26_exam_03",
        chapter="ch26",
        title="Exam 26.3: Part V Capstone Cluster-CV Model Benchmark (benchmark_part5_capstone)",
        book_section="Chapter 26 Part V Capstone Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Work on `mch_df.copy()`, standardize `residence` via `.astype(str).str.strip().str.title()`, and define `X = work[num_cols + cat_cols]`, `y = work['child_stunted']`, `groups = work['cluster_id']`.",
            2: "Loop over `{'LogisticRegression': LogisticRegression(max_iter=500), 'RandomForestClassifier': RandomForestClassifier(n_estimators=80, max_depth=6, random_state=seed), 'HistGradientBoostingClassifier': HistGradientBoostingClassifier(max_depth=4, random_state=seed)}`, wrap each in `build_dhs_end_to_end_pipeline`, and score via `cross_val_score(..., groups=groups, cv=GroupKFold(n_splits=5), scoring='roc_auc')`.",
            3: "def benchmark_part5_capstone(mch_df, seed=42):\n    work = mch_df.copy()\n    work['residence'] = work['residence'].astype(str).str.strip().str.title()\n    num_cols = ['maternal_age_years', 'anc_visits', 'maternal_hb_g_dl']\n    cat_cols = ['residence', 'wealth_quintile']\n    X, y, groups = work[num_cols + cat_cols], work['child_stunted'], work['cluster_id']\n    gkf = GroupKFold(n_splits=5)\n    models = {...}\n    rows = []\n    for name, mdl in models.items():\n        pipe = build_dhs_end_to_end_pipeline(num_cols, cat_cols, mdl)\n        scores = cross_val_score(pipe, X, y, groups=groups, cv=gkf, scoring='roc_auc')\n        rows.append({'model': name, 'mean_auc': round(float(scores.mean()), 4), 'std_auc': round(float(scores.std()), 4)})\n    lb = pd.DataFrame(rows).sort_values('mean_auc', ascending=False).reset_index(drop=True)\n    return {'leaderboard': lb, 'winner': str(lb.loc[0, 'model'])}",
        },
        verifier=_verify_exam_03,
    )
)
