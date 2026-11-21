"""Chapter 25 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch25_todo_01..06) and the Chapter 25 Coding Exam
(ch25_exam_01..03) for Unsupervised Learning — Clustering (K-Means,
Hierarchical Ward, DBSCAN) and Dimensionality Reduction (PCA).
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
function, validates return structures, rounding, mathematical formulas,
label-permutation invariance where appropriate, and non-mutation, while
letting NotImplementedError propagate cleanly for starter stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, (bool, np.bool_)
    )


def _is_int_like(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_))


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
                f"Expected a callable function (like {expected_name}), "
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
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable ML utilities always return structured arrays, dicts, or DataFrames.",
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


def _make_separated_blobs(seed: int = 123) -> Any:
    """Create deterministic, cleanly separated 3-cluster 2D synthetic data."""
    import numpy as np

    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[-5.0, -4.0], scale=0.45, size=(20, 2))
    c2 = rng.normal(loc=[0.0, 5.0], scale=0.45, size=(20, 2))
    c3 = rng.normal(loc=[5.5, -2.5], scale=0.45, size=(20, 2))
    return np.vstack([c1, c2, c3])


def _same_partition(labels_a: Any, labels_b: Any) -> bool:
    """Return True if two label vectors define the exact same clustering partition."""
    import numpy as np

    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if a.shape != b.shape:
        return False
    # Two partitions match iff pairwise co-membership matrices match
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    return bool(np.array_equal(same_a, same_b))


# ---------------------------------------------------------------------------
# GUIDED TO-DO 25.1: kmeans_from_scratch(X, k, max_iter=50, seed=42)
# ---------------------------------------------------------------------------
def _ref_kmeans_from_scratch(
    X: Any, k: int, max_iter: int = 50, seed: int = 42
) -> tuple[Any, Any, float]:
    import numpy as np

    X_arr = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(X_arr), size=k, replace=False)
    centroids = X_arr[init_idx].copy()
    labels = np.zeros(len(X_arr), dtype=int)

    for _ in range(max_iter):
        dists_sq = ((X_arr[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists_sq, axis=1).astype(int)
        new_centroids = centroids.copy()
        for cid in range(k):
            mask = new_labels == cid
            if np.any(mask):
                new_centroids[cid] = X_arr[mask].mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centroids, centroids):
            centroids = new_centroids
            labels = new_labels
            break
        centroids = new_centroids
        labels = new_labels

    dists_sq = ((X_arr[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dists_sq, axis=1).astype(int)
    inertia = float(np.sum((X_arr - centroids[labels]) ** 2))
    return labels, centroids, inertia


def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch25_todo_01", "To-Do 25.1", fn, "kmeans_from_scratch", 10)
    if bad is not None:
        return bad

    X_blobs = _make_separated_blobs(seed=101)
    rng = np.random.default_rng(202)
    X_two = np.vstack(
        [
            rng.normal(loc=[-4.0, 0.0, 1.0], scale=0.4, size=(15, 3)),
            rng.normal(loc=[4.0, 2.0, -1.0], scale=0.4, size=(15, 3)),
        ]
    )

    cases = [
        (X_blobs, 3, {}, "3 separated 2D blobs (defaults)"),
        (X_two, 2, {"max_iter": 30, "seed": 7}, "2 separated 3D blobs (seed=7)"),
    ]
    errors = []

    for X_in, k, kwargs, desc in cases:
        X_copy = X_in.copy()
        try:
            got = fn(X_copy, k, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch25_todo_01",
                False,
                "failed",
                "To-Do 25.1 — Check Function Signature `kmeans_from_scratch(X, k, max_iter=50, seed=42)`",
                f"Call failed on {desc} with TypeError: {exc}",
                details=[
                    "Define `def kmeans_from_scratch(X, k, max_iter=50, seed=42):`."
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch25_todo_01",
                False,
                "failed",
                "To-Do 25.1 — Runtime Error in `kmeans_from_scratch`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch25_todo_01", "To-Do 25.1", f"kmeans_from_scratch(X, {k})", 10
            )

        if not isinstance(got, tuple) or len(got) != 3:
            return CheckResult(
                "ch25_todo_01",
                False,
                "failed",
                "To-Do 25.1 — Return a 3-Tuple `(labels, centroids, inertia)`",
                f"Expected a 3-tuple `(labels, centroids, inertia)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_in):
            errors.append(f"[{desc}] Mutated input array X! Never overwrite caller data.")

        got_labels, got_centroids, got_inertia = got
        got_labels_arr = np.asarray(got_labels)
        got_centroids_arr = np.asarray(got_centroids, dtype=float)

        if got_labels_arr.ndim != 1 or got_labels_arr.shape != (len(X_in),):
            errors.append(
                f"[{desc}] labels shape must be ({len(X_in)},), got {got_labels_arr.shape}."
            )
            continue
        if not np.issubdtype(got_labels_arr.dtype, np.integer):
            errors.append(
                f"[{desc}] labels must have integer dtype, got {got_labels_arr.dtype}."
            )
            continue
        if got_centroids_arr.shape != (k, X_in.shape[1]):
            errors.append(
                f"[{desc}] centroids shape must be {(k, X_in.shape[1])}, got {got_centroids_arr.shape}."
            )
            continue
        if not _is_number(got_inertia):
            errors.append(
                f"[{desc}] inertia must be a float, got {type(got_inertia).__name__}."
            )
            continue

        if set(int(v) for v in np.unique(got_labels_arr)) - set(range(k)):
            errors.append(
                f"[{desc}] labels must lie in 0..{k - 1}, got unique {np.unique(got_labels_arr).tolist()}."
            )
            continue

        # Check internal consistency of inertia against returned centroids & labels
        calc_inertia = float(np.sum((X_in - got_centroids_arr[got_labels_arr]) ** 2))
        if not math.isclose(float(got_inertia), calc_inertia, rel_tol=1e-4, abs_tol=1e-4):
            errors.append(
                f"[{desc}] inertia ({float(got_inertia):.4f}) does not match sum of squared "
                f"distances to returned centroids ({calc_inertia:.4f})."
            )

        # Check optimality against reference K-Means on well-separated blobs
        ref_labels, _, ref_inertia = _ref_kmeans_from_scratch(X_in, k, **kwargs)
        if not math.isclose(float(got_inertia), ref_inertia, rel_tol=1e-3, abs_tol=1e-2):
            errors.append(
                f"[{desc}] inertia={float(got_inertia):.4f} is far from converged optimal "
                f"{ref_inertia:.4f}. Verify Euclidean assignment and centroid mean update!"
            )

        # Check determinism with same seed
        got2 = fn(X_in.copy(), k, **kwargs)
        if not (
            np.array_equal(np.asarray(got2[0]), got_labels_arr)
            and np.allclose(np.asarray(got2[1], dtype=float), got_centroids_arr)
        ):
            errors.append(
                f"[{desc}] Non-deterministic output for fixed seed! Use `np.random.default_rng(seed)`."
            )

    if errors:
        return CheckResult(
            "ch25_todo_01",
            False,
            "failed",
            "To-Do 25.1 — Check Pure-NumPy K-Means Loop, Centroids & Inertia",
            "Initialize `k` distinct points with `np.random.default_rng(seed)`, iterate assign -> update until convergence, and return `(labels, centroids, float(inertia))`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch25_todo_01",
        True,
        "passed",
        "To-Do 25.1 Passed — Pure-NumPy K-Means Verified!",
        "Verified deterministic seeding, 1D int labels, (k, d) centroids, exact SSE inertia, and convergence on 2D and 3D separated clusters.",
        tip="Notice how K-Means is simply alternating nearest-centroid assignment and mean-vector recomputation!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_todo_01",
        chapter="ch25",
        title="To-Do 25.1: Pure-NumPy K-Means From Scratch (kmeans_from_scratch)",
        book_section="Section 25.3.2 (K-Means From Scratch in NumPy)",
        points=10,
        hints={
            1: "Convert `X` via `X_arr = np.asarray(X, dtype=float)`. Draw `k` unique indices with `rng = np.random.default_rng(seed); centroids = X_arr[rng.choice(len(X_arr), size=k, replace=False)].copy()`.",
            2: "In each iteration up to `max_iter`, compute squared distances `((X_arr[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)`, assign `labels = np.argmin(dists_sq, axis=1).astype(int)`, recompute each centroid mean (guarding empty clusters), and break when converged.",
            3: "def kmeans_from_scratch(X, k, max_iter=50, seed=42):\n    X_arr = np.asarray(X, dtype=float)\n    rng = np.random.default_rng(seed)\n    centroids = X_arr[rng.choice(len(X_arr), size=k, replace=False)].copy()\n    for _ in range(max_iter):\n        dists_sq = ((X_arr[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)\n        labels = np.argmin(dists_sq, axis=1).astype(int)\n        new_c = np.array([X_arr[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c] for c in range(k)])\n        if np.allclose(new_c, centroids):\n            centroids = new_c\n            break\n        centroids = new_c\n    inertia = float(np.sum((X_arr - centroids[labels]) ** 2))\n    return labels, centroids, ___",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 25.2: evaluate_k_range(X_scaled, k_values, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable("ch25_todo_02", "To-Do 25.2", fn, "evaluate_k_range", 10)
    if bad is not None:
        return bad

    X_blobs = StandardScaler().fit_transform(_make_separated_blobs(seed=55))
    clinic = load_data("clinic_triage", as_pandas=True)
    feat_cols = [
        "age_years",
        "weight_kg",
        "height_cm",
        "temp_c",
        "systolic_bp",
        "hemoglobin_g_dl",
        "fever_days",
    ]
    X_clinic = StandardScaler().fit_transform(clinic[feat_cols].to_numpy(dtype=float))

    cases = [
        (X_blobs, [2, 3, 4], 42, "standardized 3-blob synthetic data"),
        (X_clinic, [2, 3, 4, 5], 42, "standardized clinic_triage features"),
    ]
    expected_cols = ["k", "inertia", "silhouette"]
    errors = []

    for X_in, k_vals, seed, desc in cases:
        X_copy = X_in.copy()
        try:
            got = fn(X_copy, list(k_vals), seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_todo_02",
                False,
                "failed",
                "To-Do 25.2 — Runtime Error in `evaluate_k_range`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch25_todo_02", "To-Do 25.2", "evaluate_k_range(X_scaled, k_values)", 10
            )

        if not isinstance(got, pd.DataFrame):
            return CheckResult(
                "ch25_todo_02",
                False,
                "failed",
                "To-Do 25.2 — Return a Pandas DataFrame",
                f"Expected a pd.DataFrame with columns {expected_cols}, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_in):
            errors.append(f"[{desc}] Mutated X_scaled in place!")

        if list(got.columns) != expected_cols:
            errors.append(
                f"[{desc}] Column mismatch: expected {expected_cols}, got {list(got.columns)}."
            )
            continue

        if len(got) != len(k_vals):
            errors.append(f"[{desc}] Row count mismatch: expected {len(k_vals)}, got {len(got)}.")
            continue

        for idx, k in enumerate(k_vals):
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            lbl = km.fit_predict(X_in)
            exp_in = round(float(km.inertia_), 4)
            exp_sil = round(float(silhouette_score(X_in, lbl)), 4)

            row = got.iloc[idx]
            if int(row["k"]) != k:
                errors.append(f"[{desc}] row {idx} k: expected {k}, got {row['k']!r}")
            if not math.isclose(float(row["inertia"]), exp_in, abs_tol=1e-3):
                errors.append(
                    f"[{desc}] k={k} inertia: expected {exp_in}, got {row['inertia']!r} "
                    "(Check `n_init=10` and `random_state=seed`!)"
                )
            if not math.isclose(float(row["silhouette"]), exp_sil, abs_tol=1e-3):
                errors.append(
                    f"[{desc}] k={k} silhouette: expected {exp_sil}, got {row['silhouette']!r}"
                )

    if errors:
        return CheckResult(
            "ch25_todo_02",
            False,
            "failed",
            "To-Do 25.2 — Check KMeans Sweep (`n_init=10`), Silhouette & Rounding",
            "Fit `KMeans(n_clusters=k, n_init=10, random_state=seed)` for each `k` in `k_values` and return a DataFrame with columns `['k', 'inertia', 'silhouette']` rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch25_todo_02",
        True,
        "passed",
        "To-Do 25.2 Passed — Elbow Inertia & Silhouette Sweep Verified!",
        "Matched exact inertia and silhouette scores across synthetic clusters and standardized `clinic_triage` features.",
        tip="Inertia always drops as k increases; Silhouette peaks where clusters are both tight internally and well separated externally.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_todo_02",
        chapter="ch25",
        title="To-Do 25.2: Elbow Inertia & Silhouette K-Sweep (evaluate_k_range)",
        book_section="Section 25.3.3 (Choosing k: Elbow Method & Silhouette Score)",
        points=10,
        hints={
            1: "Loop over `k` in `k_values`, fit `km = KMeans(n_clusters=k, n_init=10, random_state=seed)`, and get `labels = km.fit_predict(X_scaled)`.",
            2: "Record `{'k': int(k), 'inertia': round(float(km.inertia_), 4), 'silhouette': round(float(silhouette_score(X_scaled, labels)), 4)}` for each `k`.",
            3: "def evaluate_k_range(X_scaled, k_values, seed=42):\n    rows = []\n    for k in k_values:\n        km = KMeans(n_clusters=k, n_init=10, random_state=seed)\n        labels = km.fit_predict(X_scaled)\n        rows.append({\n            'k': int(k),\n            'inertia': round(float(km.inertia_), 4),\n            'silhouette': round(float(silhouette_score(X_scaled, ___)), 4),\n        })\n    return pd.DataFrame(rows)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 25.3: fit_agglomerative_ward(X_scaled, n_clusters=3)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable("ch25_todo_03", "To-Do 25.3", fn, "fit_agglomerative_ward", 10)
    if bad is not None:
        return bad

    clinic = load_data("clinic_triage", as_pandas=True)
    feat_cols = [
        "age_years",
        "weight_kg",
        "height_cm",
        "temp_c",
        "systolic_bp",
        "hemoglobin_g_dl",
        "fever_days",
    ]
    X_clinic = StandardScaler().fit_transform(clinic[feat_cols].to_numpy(dtype=float))
    X_blobs = StandardScaler().fit_transform(_make_separated_blobs(seed=88))

    cases = [
        (X_clinic, 3, "clinic_triage Ward k=3"),
        (X_blobs, 3, "synthetic blobs Ward k=3"),
        (X_clinic, 4, "clinic_triage Ward k=4"),
    ]
    expected_keys = ["labels", "silhouette", "cluster_sizes"]
    errors = []

    for X_in, n_cl, desc in cases:
        try:
            got = fn(X_in.copy(), n_clusters=n_cl)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_todo_03",
                False,
                "failed",
                "To-Do 25.3 — Runtime Error in `fit_agglomerative_ward`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch25_todo_03", "To-Do 25.3", f"fit_agglomerative_ward(X_scaled, n_clusters={n_cl})", 10
            )

        key_err = _check_dict_keys("ch25_todo_03", "To-Do 25.3", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        ref_model = AgglomerativeClustering(n_clusters=n_cl, linkage="ward")
        ref_labels = ref_model.fit_predict(X_in)
        exp_sil = round(float(silhouette_score(X_in, ref_labels)), 4)
        exp_sizes = {
            int(k): int(v) for k, v in zip(*np.unique(ref_labels, return_counts=True))
        }

        got_labels = np.asarray(got["labels"])
        if got_labels.shape != ref_labels.shape or not np.array_equal(got_labels, ref_labels):
            errors.append(
                f"[{desc}] labels do not match `AgglomerativeClustering(n_clusters={n_cl}, linkage='ward')`."
            )
        if not _is_number(got["silhouette"]) or not math.isclose(
            float(got["silhouette"]), exp_sil, abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] silhouette: expected {exp_sil}, got {got['silhouette']!r}"
            )
        got_sizes = got["cluster_sizes"]
        if not isinstance(got_sizes, dict) or got_sizes != exp_sizes:
            errors.append(
                f"[{desc}] cluster_sizes: expected {exp_sizes}, got {got_sizes!r}"
            )
        elif not all(
            isinstance(k, int) and not isinstance(k, bool) and isinstance(v, int) and not isinstance(v, bool)
            for k, v in got_sizes.items()
        ):
            errors.append(
                f"[{desc}] cluster_sizes keys and values must be plain Python `int` (cast via `int(k): int(v)`)."
            )

    if errors:
        return CheckResult(
            "ch25_todo_03",
            False,
            "failed",
            "To-Do 25.3 — Check Ward Agglomerative Clustering & Size Dict",
            "Fit `AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')` and return `{'labels': labels, 'silhouette': round4, 'cluster_sizes': {int: int}}`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch25_todo_03",
        True,
        "passed",
        "To-Do 25.3 Passed — Hierarchical Ward Clustering Verified!",
        "Verified deterministic Ward linkage labels, silhouette score (0.2847 on clinic_triage k=3), and plain Python int cluster_sizes.",
        tip="Unlike K-Means, Ward agglomerative clustering is deterministic (no random seed needed) and exposes a full merge hierarchy.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_todo_03",
        chapter="ch25",
        title="To-Do 25.3: Hierarchical Ward Clustering (fit_agglomerative_ward)",
        book_section="Section 25.3.4 (Hierarchical Agglomerative Clustering with Ward Linkage)",
        points=10,
        hints={
            1: "Instantiate `AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')` and call `.fit_predict(X_scaled)`.",
            2: "Compute `sil = round(float(silhouette_score(X_scaled, labels)), 4)` and build `cluster_sizes = {int(c): int(cnt) for c, cnt in zip(*np.unique(labels, return_counts=True))}`.",
            3: "def fit_agglomerative_ward(X_scaled, n_clusters=3):\n    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='___')\n    labels = model.fit_predict(X_scaled)\n    unique, counts = np.unique(labels, return_counts=True)\n    return {\n        'labels': labels,\n        'silhouette': round(float(silhouette_score(X_scaled, labels)), 4),\n        'cluster_sizes': {int(c): int(cnt) for c, cnt in zip(unique, counts)},\n    }",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 25.4: detect_dbscan_anomalies(X_scaled, eps=0.8, min_samples=5)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.cluster import DBSCAN

    bad = _check_callable("ch25_todo_04", "To-Do 25.4", fn, "detect_dbscan_anomalies", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(77)
    dense_a = rng.normal(loc=[-3.0, -3.0], scale=0.25, size=(20, 2))
    dense_b = rng.normal(loc=[3.0, 3.0], scale=0.25, size=(20, 2))
    outliers = np.array([[0.0, 0.0], [8.5, -7.0], [-9.0, 6.5]])
    X_with_outliers = np.vstack([dense_a, dense_b, outliers])

    cases = [
        (X_with_outliers, 0.8, 5, "2 dense clusters + 3 clear outliers"),
        (X_with_outliers, 0.05, 5, "tiny eps where all points become noise"),
    ]
    expected_keys = ["n_clusters", "n_noise", "labels"]
    errors = []

    for X_in, eps, min_s, desc in cases:
        try:
            got = fn(X_in.copy(), eps=eps, min_samples=min_s)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_todo_04",
                False,
                "failed",
                "To-Do 25.4 — Runtime Error in `detect_dbscan_anomalies`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch25_todo_04",
                "To-Do 25.4",
                f"detect_dbscan_anomalies(X_scaled, eps={eps}, min_samples={min_s})",
                10,
            )

        key_err = _check_dict_keys("ch25_todo_04", "To-Do 25.4", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        ref_labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X_in)
        exp_clusters = int(len(set(ref_labels.tolist()) - {-1}))
        exp_noise = int(np.sum(ref_labels == -1))

        if not _is_int_like(got["n_clusters"]) or int(got["n_clusters"]) != exp_clusters:
            extra = ""
            if _is_int_like(got["n_clusters"]) and int(got["n_clusters"]) == exp_clusters + 1:
                extra = " (Did you forget to exclude noise label `-1` from `n_clusters`?)"
            errors.append(
                f"[{desc}] n_clusters: expected {exp_clusters}, got {got['n_clusters']!r}{extra}"
            )
        if not _is_int_like(got["n_noise"]) or int(got["n_noise"]) != exp_noise:
            errors.append(f"[{desc}] n_noise: expected {exp_noise}, got {got['n_noise']!r}")
        if not np.array_equal(np.asarray(got["labels"]), ref_labels):
            errors.append(
                f"[{desc}] labels do not match `DBSCAN(eps={eps}, min_samples={min_s}).fit_predict(X_scaled)`."
            )

    if errors:
        return CheckResult(
            "ch25_todo_04",
            False,
            "failed",
            "To-Do 25.4 — Check DBSCAN Parameters & Noise Label `-1` Handling",
            "Fit `DBSCAN(eps=eps, min_samples=min_samples)`, exclude `-1` from `n_clusters`, count `labels == -1` for `n_noise`, and return `{'n_clusters', 'n_noise', 'labels'}`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch25_todo_04",
        True,
        "passed",
        "To-Do 25.4 Passed — DBSCAN Density & Anomaly Detection Verified!",
        "Correctly separated dense clusters from `-1` noise anomalies, including the all-noise tiny-eps boundary case.",
        tip="Always subtract `{-1}` before counting DBSCAN clusters—otherwise noise points masquerade as an extra cluster!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_todo_04",
        chapter="ch25",
        title="To-Do 25.4: Density Clustering & Outlier Flagging (detect_dbscan_anomalies)",
        book_section="Section 25.3.5 (DBSCAN: Density Clustering & Anomaly Flagging)",
        points=10,
        hints={
            1: "Call `labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)`. DBSCAN marks outliers with label `-1`.",
            2: "Count clusters excluding `-1` via `n_clusters = len(set(labels) - {-1})` and noise points via `n_noise = int(np.sum(labels == -1))`.",
            3: "def detect_dbscan_anomalies(X_scaled, eps=0.8, min_samples=5):\n    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)\n    n_clusters = int(len(set(labels.tolist()) - {___}))\n    n_noise = int(np.sum(labels == -1))\n    return {'n_clusters': n_clusters, 'n_noise': n_noise, 'labels': labels}",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 25.5: pca_from_scratch_svd(X, n_components=2)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch25_todo_05", "To-Do 25.5", fn, "pca_from_scratch_svd", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(314)
    z1 = rng.normal(0.0, 3.0, size=60)
    z2 = rng.normal(0.0, 1.2, size=60)
    # Deliberately add large column offsets to catch uncentered SVD implementations
    X_syn = np.column_stack(
        [
            50.0 + z1 + 0.2 * rng.normal(size=60),
            120.0 + 0.8 * z1 + z2,
            10.0 - 0.5 * z1 + 0.9 * z2,
            37.0 + 0.1 * rng.normal(size=60),
        ]
    )

    cases = [
        (X_syn, 2, "4D correlated data with non-zero means (k=2)"),
        (X_syn, 3, "4D correlated data with non-zero means (k=3)"),
    ]
    errors = []

    for X_in, n_comp, desc in cases:
        X_copy = X_in.copy()
        try:
            got = fn(X_copy, n_components=n_comp)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_todo_05",
                False,
                "failed",
                "To-Do 25.5 — Runtime Error in `pca_from_scratch_svd`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch25_todo_05", "To-Do 25.5", f"pca_from_scratch_svd(X, n_components={n_comp})", 10
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch25_todo_05",
                False,
                "failed",
                "To-Do 25.5 — Return a 2-Tuple `(X_projected, explained_variance_ratio)`",
                f"Expected a 2-tuple `(X_projected, explained_variance_ratio)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_in):
            errors.append(f"[{desc}] Mutated input matrix X in place!")

        got_proj = np.asarray(got[0], dtype=float)
        got_evr = np.asarray(got[1], dtype=float)

        if got_proj.shape != (len(X_in), n_comp):
            errors.append(
                f"[{desc}] X_projected shape must be {(len(X_in), n_comp)}, got {got_proj.shape}."
            )
            continue
        if got_evr.shape != (n_comp,):
            errors.append(
                f"[{desc}] explained_variance_ratio shape must be {(n_comp,)}, got {got_evr.shape}."
            )
            continue

        X_c = X_in - X_in.mean(axis=0)
        _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        ref_proj = X_c @ Vt[:n_comp].T
        ref_evr = (S ** 2) / np.sum(S ** 2)
        ref_evr = ref_evr[:n_comp]

        # Check explained variance ratio (sign-invariant)
        if not np.allclose(got_evr, ref_evr, atol=1e-4):
            _, S_unc, _ = np.linalg.svd(X_in, full_matrices=False)
            evr_unc = ((S_unc ** 2) / np.sum(S_unc ** 2))[:n_comp]
            extra = ""
            if np.allclose(got_evr, evr_unc, atol=1e-3):
                extra = " (Looks like you forgot to center X by subtracting `X.mean(axis=0)` before SVD!)"
            errors.append(
                f"[{desc}] explained_variance_ratio mismatch: expected {np.round(ref_evr, 4).tolist()}, "
                f"got {np.round(got_evr, 4).tolist()}{extra}"
            )

        # Check projected coordinates up to sign flip per principal axis
        aligned = got_proj.copy()
        for j in range(n_comp):
            if np.dot(aligned[:, j], ref_proj[:, j]) < 0:
                aligned[:, j] *= -1.0
        if not np.allclose(aligned, ref_proj, atol=1e-4):
            errors.append(
                f"[{desc}] Projected coordinates do not match `X_centered @ Vt[:n_components].T` "
                "(even allowing +/- sign flips per axis)."
            )

    if errors:
        return CheckResult(
            "ch25_todo_05",
            False,
            "failed",
            "To-Do 25.5 — Check Mean-Centering & SVD Projection Math",
            "Center `X_c = X - X.mean(axis=0)`, run `U, S, Vt = np.linalg.svd(X_c, full_matrices=False)`, project `X_c @ Vt[:n_components].T`, and compute `(S**2 / np.sum(S**2))[:n_components]`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch25_todo_05",
        True,
        "passed",
        "To-Do 25.5 Passed — PCA From Scratch via NumPy SVD Verified!",
        "Verified mean-centering, (n, k) projected scores (sign-aligned), and exact explained variance ratios from singular values.",
        tip="Mean-centering is mandatory before SVD—otherwise the first singular vector merely points toward the data cloud's mean offset!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_todo_05",
        chapter="ch25",
        title="To-Do 25.5: PCA From Scratch via SVD (pca_from_scratch_svd)",
        book_section="Section 25.3.6 (PCA From Scratch via SVD vs. Scikit-Learn)",
        points=10,
        hints={
            1: "Convert `X` to float array and center each column: `X_c = X_arr - X_arr.mean(axis=0)`. Then call `U, S, Vt = np.linalg.svd(X_c, full_matrices=False)`.",
            2: "Project onto the top `n_components` right singular vectors via `X_proj = X_c @ Vt[:n_components].T`, and compute explained variance ratios as `(S ** 2) / np.sum(S ** 2)` sliced to `[:n_components]`.",
            3: "def pca_from_scratch_svd(X, n_components=2):\n    X_arr = np.asarray(X, dtype=float)\n    X_c = X_arr - X_arr.mean(axis=0)\n    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)\n    X_proj = X_c @ Vt[:n_components].T\n    evr = (S ** 2) / np.sum(S ** 2)\n    return X_proj, evr[:___]",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 25.6: components_for_variance_target(X_scaled, target_var=0.90)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable(
        "ch25_todo_06", "To-Do 25.6", fn, "components_for_variance_target", 10
    )
    if bad is not None:
        return bad

    clinic = load_data("clinic_triage", as_pandas=True)
    feat_cols = [
        "age_years",
        "weight_kg",
        "height_cm",
        "temp_c",
        "systolic_bp",
        "hemoglobin_g_dl",
        "fever_days",
    ]
    X_clinic = StandardScaler().fit_transform(clinic[feat_cols].to_numpy(dtype=float))

    cases = [
        (X_clinic, 0.90, "clinic_triage target=0.90 (expects 5)"),
        (X_clinic, 0.85, "clinic_triage target=0.85 (expects 4)"),
        (X_clinic, 0.30, "clinic_triage target=0.30 (expects 1)"),
    ]
    errors = []

    for X_in, target, desc in cases:
        try:
            got = fn(X_in.copy(), target_var=target)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_todo_06",
                False,
                "failed",
                "To-Do 25.6 — Runtime Error in `components_for_variance_target`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch25_todo_06",
                "To-Do 25.6",
                f"components_for_variance_target(X_scaled, target_var={target})",
                10,
            )

        pca = PCA().fit(X_in)
        cum = np.cumsum(pca.explained_variance_ratio_)
        exp_k = int(np.argmax(cum >= target) + 1)

        if not _is_int_like(got) or int(got) != exp_k:
            extra = ""
            if _is_int_like(got) and int(got) == exp_k - 1:
                extra = " (Off-by-one! Remember NumPy index `k - 1` corresponds to `k` components.)"
            errors.append(f"[{desc}] Expected int {exp_k}, got {got!r}{extra}")
        elif not isinstance(got, int) or isinstance(got, bool):
            errors.append(
                f"[{desc}] Return a plain Python `int` (`int(...)`), got {type(got).__name__}."
            )

    if errors:
        return CheckResult(
            "ch25_todo_06",
            False,
            "failed",
            "To-Do 25.6 — Check Cumulative Explained Variance Threshold",
            "Fit full `PCA().fit(X_scaled)`, compute `cum = np.cumsum(pca.explained_variance_ratio_)`, and return the smallest 1-indexed `int` k where `cum[k - 1] >= target_var`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch25_todo_06",
        True,
        "passed",
        "To-Do 25.6 Passed — Variance-Target Component Selector Verified!",
        "Correctly selected 1, 4, and 5 components for 30%, 85%, and 90% variance retention on `clinic_triage`.",
        tip="Selecting k by cumulative explained variance makes PCA compression transparent and reproducible in paper methods sections.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_todo_06",
        chapter="ch25",
        title="To-Do 25.6: Minimum PCs for Variance Target (components_for_variance_target)",
        book_section="Section 25.3.6 (Cumulative Variance & Scree Retention)",
        points=10,
        hints={
            1: "Fit a full `pca = PCA().fit(X_scaled)` and compute `cum_var = np.cumsum(pca.explained_variance_ratio_)`.",
            2: "Find the first index where `cum_var >= target_var` using `np.argmax(cum_var >= target_var)` and add `1` (cast to Python `int`).",
            3: "def components_for_variance_target(X_scaled, target_var=0.90):\n    pca = PCA().fit(X_scaled)\n    cum_var = np.cumsum(pca.explained_variance_ratio_)\n    return int(np.argmax(cum_var >= ___) + 1)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 25.1 (30 pts): select_optimal_k_silhouette(X, k_range, seed=42)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable(
        "ch25_exam_01", "Exam Task 25.1", fn, "select_optimal_k_silhouette", 30
    )
    if bad is not None:
        return bad

    # Synthetic 3-cluster data with massive scale mismatch to catch missing StandardScaler!
    rng = np.random.default_rng(404)
    blobs = _make_separated_blobs(seed=404)
    # Column 0 is shrunk by 0.001, column 1 is normal, column 2 is huge-scale noise
    X_unscaled_trap = np.column_stack(
        [
            blobs[:, 0] * 0.01,
            blobs[:, 1] * 0.01,
            rng.uniform(0.0, 5000.0, size=len(blobs)),
        ]
    )
    # Make column 2 cluster-aligned weakly, or better: make blobs separated on col 0 & 1 and moderate on col 2
    X_scaled_test = np.column_stack(
        [
            blobs[:, 0] * 0.01,
            blobs[:, 1] * 100.0,
        ]
    )

    clinic = load_data("clinic_triage", as_pandas=True)
    feat_cols = [
        "age_years",
        "weight_kg",
        "height_cm",
        "temp_c",
        "systolic_bp",
        "hemoglobin_g_dl",
        "fever_days",
    ]
    X_clinic_raw = clinic[feat_cols].to_numpy(dtype=float)

    cases = [
        (X_scaled_test, [2, 3, 4, 5], 42, "unscaled 2D blobs with 10,000x scale mismatch"),
        (X_clinic_raw, [2, 3, 4, 5, 6], 42, "raw clinic_triage 7 clinical vitals"),
    ]
    expected_keys = ["best_k", "best_silhouette", "table"]
    errors = []

    for X_in, k_range, seed, desc in cases:
        X_copy = X_in.copy()
        try:
            got = fn(X_copy, list(k_range), seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_exam_01",
                False,
                "failed",
                "Exam Task 25.1 — Runtime Error in `select_optimal_k_silhouette`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch25_exam_01", "Exam Task 25.1", "select_optimal_k_silhouette(X, k_range)", 30
            )

        key_err = _check_dict_keys("ch25_exam_01", "Exam Task 25.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        if not np.array_equal(X_copy, X_in):
            errors.append(f"[{desc}] Mutated input array X!")

        X_sc = StandardScaler().fit_transform(X_in)
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=seed)
            lbl = km.fit_predict(X_sc)
            rows.append(
                {
                    "k": int(k),
                    "inertia": round(float(km.inertia_), 4),
                    "silhouette": round(float(silhouette_score(X_sc, lbl)), 4),
                }
            )
        ref_tbl = pd.DataFrame(rows)
        best_row = ref_tbl.loc[ref_tbl["silhouette"].idxmax()]
        exp_k = int(best_row["k"])
        exp_sil = float(best_row["silhouette"])

        if not _is_int_like(got["best_k"]) or int(got["best_k"]) != exp_k:
            errors.append(
                f"[{desc}] best_k: expected {exp_k}, got {got['best_k']!r} "
                "(Did you standardize X with `StandardScaler()` before sweeping `k_range`?)"
            )
        if not _is_number(got["best_silhouette"]) or not math.isclose(
            float(got["best_silhouette"]), exp_sil, abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] best_silhouette: expected {exp_sil}, got {got['best_silhouette']!r}"
            )
        tbl = got["table"]
        if not isinstance(tbl, pd.DataFrame) or list(tbl.columns) != ["k", "inertia", "silhouette"]:
            errors.append(
                f"[{desc}] table must be a DataFrame with columns ['k', 'inertia', 'silhouette']."
            )
            continue
        if len(tbl) != len(ref_tbl):
            errors.append(f"[{desc}] table row count mismatch: expected {len(ref_tbl)}, got {len(tbl)}.")
            continue
        for i in range(len(ref_tbl)):
            if not math.isclose(
                float(tbl.iloc[i]["silhouette"]), float(ref_tbl.iloc[i]["silhouette"]), abs_tol=1e-3
            ) or not math.isclose(
                float(tbl.iloc[i]["inertia"]), float(ref_tbl.iloc[i]["inertia"]), abs_tol=1e-2
            ):
                errors.append(
                    f"[{desc}] table row {i} (k={k_range[i]}) mismatch: expected "
                    f"inertia={ref_tbl.iloc[i]['inertia']}, sil={ref_tbl.iloc[i]['silhouette']}; "
                    f"got inertia={tbl.iloc[i]['inertia']}, sil={tbl.iloc[i]['silhouette']}."
                )

    if errors:
        return CheckResult(
            "ch25_exam_01",
            False,
            "failed",
            "Exam Task 25.1 — Check StandardScaler Preprocessing & Silhouette Selection",
            "Standardize `X` via `StandardScaler().fit_transform(X)`, sweep `k_range` with `KMeans(n_clusters=k, n_init=10, random_state=seed)`, and pick the `k` maximizing silhouette.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch25_exam_01",
        True,
        "passed",
        "Exam Task 25.1 Passed (30/30 pts) — Scale-Safe Optimal-K Selector Verified!",
        "Standardized raw multi-scale inputs, swept candidate k values, and selected optimal k by peak silhouette score.",
        points_earned=30,
        points_possible=30,
        tip="Standardizing before distance-based clustering prevents large-unit features from drowning out clinically vital small-unit biomarkers.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_exam_01",
        chapter="ch25",
        title="Exam 25.1: Scale-Safe Optimal-K Selector (select_optimal_k_silhouette)",
        book_section="Chapter 25 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "First scale `X_scaled = StandardScaler().fit_transform(X)`. Then sweep each `k` in `k_range` using `KMeans(n_clusters=k, n_init=10, random_state=seed)`.",
            2: "Build a DataFrame `table` with columns `['k', 'inertia', 'silhouette']` rounded to 4 decimals, then select the best row via `table.loc[table['silhouette'].idxmax()]`.",
            3: "def select_optimal_k_silhouette(X, k_range, seed=42):\n    X_scaled = StandardScaler().fit_transform(X)\n    rows = []\n    for k in k_range:\n        km = KMeans(n_clusters=k, n_init=10, random_state=seed)\n        lbl = km.fit_predict(X_scaled)\n        rows.append({'k': int(k), 'inertia': round(float(km.inertia_), 4), 'silhouette': round(float(silhouette_score(X_scaled, lbl)), 4)})\n    table = pd.DataFrame(rows)\n    best = table.loc[table['silhouette'].idxmax()]\n    return {'best_k': int(best['k']), 'best_silhouette': round(float(best['silhouette']), 4), 'table': ___}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 25.2 (35 pts): district_risk_phenotyping(clinic_df, n_clusters=3, seed=42)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable(
        "ch25_exam_02", "Exam Task 25.2", fn, "district_risk_phenotyping", 35
    )
    if bad is not None:
        return bad

    clinic = load_data("clinic_triage", as_pandas=True)

    # Secondary synthetic 8-district cohort (including both "Moderate" and "Urgent" flags)
    rng = np.random.default_rng(2026)
    syn_rows = []
    profiles = [
        ("Barishal", 36.8, 118, 14.2, ["Routine", "Routine", "Routine", "Moderate"]),
        ("Bogura", 36.9, 120, 14.0, ["Routine", "Routine", "Routine", "Moderate"]),
        ("Cumilla", 37.8, 134, 12.1, ["Moderate", "Urgent", "Routine", "Moderate"]),
        ("Mymensingh", 37.9, 136, 11.9, ["Urgent", "Moderate", "Urgent", "Routine"]),
        ("Jessore", 37.2, 145, 13.5, ["Routine", "Moderate", "Routine", "Moderate"]),
        ("Noakhali", 37.1, 147, 13.4, ["Routine", "Moderate", "Routine", "Moderate"]),
    ]
    for dist, t0, s0, h0, prios in profiles:
        for i in range(12):
            syn_rows.append(
                {
                    "district": dist,
                    "temp_c": round(float(t0 + rng.normal(0, 0.15)), 2),
                    "systolic_bp": int(round(s0 + rng.normal(0, 2.0))),
                    "hemoglobin_g_dl": round(float(h0 + rng.normal(0, 0.2)), 2),
                    "triage_priority": prios[i % len(prios)],
                }
            )
    syn_df = pd.DataFrame(syn_rows)

    cases = [
        (clinic, 3, 42, "real clinic_triage (6 Bangladesh districts)"),
        (syn_df, 3, 42, "synthetic 6-district multi-phenotype cohort"),
    ]
    expected_keys = ["district_clusters", "silhouette"]
    errors = []

    for df_in, n_cl, seed, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy, n_clusters=n_cl, seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_exam_02",
                False,
                "failed",
                "Exam Task 25.2 — Runtime Error in `district_risk_phenotyping`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch25_exam_02",
                "Exam Task 25.2",
                f"district_risk_phenotyping(clinic_df, n_clusters={n_cl})",
                35,
            )

        key_err = _check_dict_keys("ch25_exam_02", "Exam Task 25.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Mutated input DataFrame `clinic_df`! Work on a copy or groupby result.")

        ref = df_in.groupby("district", as_index=False).agg(
            mean_temp=("temp_c", "mean"),
            mean_sbp=("systolic_bp", "mean"),
            mean_hb=("hemoglobin_g_dl", "mean"),
            urgent_rate=("triage_priority", lambda s: (s != "Routine").mean()),
        )
        feat_cols = ["mean_temp", "mean_sbp", "mean_hb", "urgent_rate"]
        X_sc = StandardScaler().fit_transform(ref[feat_cols].to_numpy(dtype=float))
        km = KMeans(n_clusters=n_cl, n_init=10, random_state=seed)
        ref["cluster"] = km.fit_predict(X_sc).astype(int)
        exp_sil = round(float(silhouette_score(X_sc, ref["cluster"])), 4)

        dc = got["district_clusters"]
        if not isinstance(dc, pd.DataFrame):
            errors.append(f"[{desc}] `district_clusters` must be a pd.DataFrame, got {type(dc).__name__}.")
            continue
        if "district" not in dc.columns or "cluster" not in dc.columns:
            errors.append(
                f"[{desc}] `district_clusters` must include at least `'district'` and `'cluster'` columns; got {list(dc.columns)}."
            )
            continue

        dc_sorted = dc.sort_values("district").reset_index(drop=True)
        ref_sorted = ref.sort_values("district").reset_index(drop=True)

        if list(dc_sorted["district"]) != list(ref_sorted["district"]):
            errors.append(
                f"[{desc}] District list mismatch: expected {list(ref_sorted['district'])}, got {list(dc_sorted['district'])}."
            )
            continue

        if not _same_partition(dc_sorted["cluster"].to_numpy(), ref_sorted["cluster"].to_numpy()):
            errors.append(
                f"[{desc}] District cluster partition does not match scaled KMeans(n_clusters={n_cl}, n_init=10, random_state={seed}). "
                "Check all 4 aggregates (`mean_temp`, `mean_sbp`, `mean_hb`, `urgent_rate = (triage_priority != 'Routine').mean()`) and StandardScaler!"
            )

        if not _is_number(got["silhouette"]) or not math.isclose(
            float(got["silhouette"]), exp_sil, abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] silhouette: expected {exp_sil}, got {got['silhouette']!r}"
            )

    if errors:
        return CheckResult(
            "ch25_exam_02",
            False,
            "failed",
            "Exam Task 25.2 — Check District Aggregation, Scaling & K-Means Clustering",
            "Aggregate `mean_temp`, `mean_sbp`, `mean_hb`, and `urgent_rate = (triage_priority != 'Routine').mean()` per district, standardize via `StandardScaler`, cluster with `KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)`, and return `{'district_clusters': df, 'silhouette': round4}`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch25_exam_02",
        True,
        "passed",
        "Exam Task 25.2 Passed (35/35 pts) — District Risk Phenotyping Pipeline Verified!",
        "Aggregated clinical indicators across districts, standardized features, and partitioned districts into interpretable risk phenotypes (silhouette=0.2000 on clinic_triage).",
        points_earned=35,
        points_possible=35,
        tip="Entity-level aggregation followed by scaling + clustering is a workhorse pattern for health-system sub-national targeting.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_exam_02",
        chapter="ch25",
        title="Exam 25.2: District Risk Phenotyping Pipeline (district_risk_phenotyping)",
        book_section="Chapter 25 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Group by `'district'` with `as_index=False` and compute `mean_temp` (`temp_c`), `mean_sbp` (`systolic_bp`), `mean_hb` (`hemoglobin_g_dl`), and `urgent_rate` (`(s != 'Routine').mean()` on `triage_priority`).",
            2: "Standardize `['mean_temp', 'mean_sbp', 'mean_hb', 'urgent_rate']` with `StandardScaler().fit_transform(...)`, fit `KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)`, and store integer labels in `agg['cluster']`.",
            3: "def district_risk_phenotyping(clinic_df, n_clusters=3, seed=42):\n    agg = clinic_df.groupby('district', as_index=False).agg(\n        mean_temp=('temp_c', 'mean'),\n        mean_sbp=('systolic_bp', 'mean'),\n        mean_hb=('hemoglobin_g_dl', 'mean'),\n        urgent_rate=('triage_priority', lambda s: (s != 'Routine').mean()),\n    )\n    feats = agg[['mean_temp', 'mean_sbp', 'mean_hb', 'urgent_rate']].to_numpy(dtype=float)\n    X_sc = StandardScaler().fit_transform(feats)\n    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X_sc)\n    agg['cluster'] = labels.astype(int)\n    return {'district_clusters': agg, 'silhouette': round(float(silhouette_score(X_sc, ___)), 4)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 25.3 (35 pts): pca_clinical_compression_audit(X, variance_threshold=0.85)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable(
        "ch25_exam_03", "Exam Task 25.3", fn, "pca_clinical_compression_audit", 35
    )
    if bad is not None:
        return bad

    clinic = load_data("clinic_triage", as_pandas=True)
    feat_cols = [
        "age_years",
        "weight_kg",
        "height_cm",
        "temp_c",
        "systolic_bp",
        "hemoglobin_g_dl",
        "fever_days",
    ]
    X_clinic = clinic[feat_cols].to_numpy(dtype=float)

    rng = np.random.default_rng(909)
    latent = rng.normal(size=80)
    # Feature index 2 has strongest loading on PC1 after standardization
    X_syn = np.column_stack(
        [
            0.3 * latent + rng.normal(scale=0.9, size=80),
            0.4 * latent + rng.normal(scale=0.8, size=80),
            5.0 * latent + rng.normal(scale=0.1, size=80),
            -4.5 * latent + rng.normal(scale=0.2, size=80),
            rng.normal(scale=1.0, size=80),
        ]
    )

    cases = [
        (X_clinic, 0.85, "clinic_triage 7 features (threshold=0.85)"),
        (X_clinic, 0.90, "clinic_triage 7 features (threshold=0.90)"),
        (X_syn, 0.80, "5D correlated synthetic features (threshold=0.80)"),
    ]
    expected_keys = [
        "n_components",
        "cumulative_variance",
        "top_pc1_feature_idx",
        "reconstruction_mse",
    ]
    errors = []

    for X_in, thresh, desc in cases:
        X_copy = X_in.copy()
        try:
            got = fn(X_copy, variance_threshold=thresh)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch25_exam_03",
                False,
                "failed",
                "Exam Task 25.3 — Runtime Error in `pca_clinical_compression_audit`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch25_exam_03",
                "Exam Task 25.3",
                f"pca_clinical_compression_audit(X, variance_threshold={thresh})",
                35,
            )

        key_err = _check_dict_keys("ch25_exam_03", "Exam Task 25.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not np.array_equal(X_copy, X_in):
            errors.append(f"[{desc}] Mutated input array X!")

        X_sc = StandardScaler().fit_transform(X_in)
        full_pca = PCA().fit(X_sc)
        cum = np.cumsum(full_pca.explained_variance_ratio_)
        exp_k = int(np.argmax(cum >= thresh) + 1)

        pca_k = PCA(n_components=exp_k)
        Z = pca_k.fit_transform(X_sc)
        X_hat = pca_k.inverse_transform(Z)

        exp_cum = round(float(np.sum(pca_k.explained_variance_ratio_)), 4)
        exp_top_idx = int(np.argmax(np.abs(pca_k.components_[0])))
        exp_mse = round(float(np.mean((X_sc - X_hat) ** 2)), 5)

        if not _is_int_like(got["n_components"]) or int(got["n_components"]) != exp_k:
            errors.append(
                f"[{desc}] n_components: expected {exp_k}, got {got['n_components']!r} "
                "(Check StandardScaler preprocessing and cumulative variance threshold!)"
            )
        if not _is_number(got["cumulative_variance"]) or not math.isclose(
            float(got["cumulative_variance"]), exp_cum, abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] cumulative_variance: expected {exp_cum}, got {got['cumulative_variance']!r}"
            )
        if not _is_int_like(got["top_pc1_feature_idx"]) or int(got["top_pc1_feature_idx"]) != exp_top_idx:
            errors.append(
                f"[{desc}] top_pc1_feature_idx: expected {exp_top_idx}, got {got['top_pc1_feature_idx']!r} "
                "(Use `int(np.argmax(np.abs(pca_k.components_[0])))` — don't forget `np.abs(...)`!)"
            )
        if not _is_number(got["reconstruction_mse"]) or not math.isclose(
            float(got["reconstruction_mse"]), exp_mse, abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] reconstruction_mse: expected {exp_mse}, got {got['reconstruction_mse']!r} "
                "(Compute MSE on standardized `X_scaled` vs `pca_k.inverse_transform(Z)`!)"
            )

    if errors:
        return CheckResult(
            "ch25_exam_03",
            False,
            "failed",
            "Exam Task 25.3 — Check Scaling, PC Selection, PC1 Loading & Reconstruction MSE",
            "Standardize `X`, find smallest `k` reaching `variance_threshold`, fit `PCA(n_components=k)`, and compute `cumulative_variance` (4 dec), `top_pc1_feature_idx` via `abs(components_[0])`, and `reconstruction_mse` (5 dec) on `X_scaled`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch25_exam_03",
        True,
        "passed",
        "Exam Task 25.3 Passed (35/35 pts) — PCA Compression & Reconstruction Audit Verified!",
        "Audited clinical dimensionality reduction (4 PCs retain 86.21% variance with MSE=0.13789, led by age_years on PC1).",
        points_earned=35,
        points_possible=35,
        tip="Notice that on standardized features, reconstruction MSE equals `1 - cumulative_variance` (1 - 0.86211 = 0.13789)!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch25_exam_03",
        chapter="ch25",
        title="Exam 25.3: PCA Clinical Compression & Reconstruction Audit (pca_clinical_compression_audit)",
        book_section="Chapter 25 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Standardize via `X_sc = StandardScaler().fit_transform(X)`. Fit `PCA().fit(X_sc)` and compute `k = int(np.argmax(np.cumsum(full_pca.explained_variance_ratio_) >= variance_threshold) + 1)`.",
            2: "Fit `pca_k = PCA(n_components=k)`, project `Z = pca_k.fit_transform(X_sc)`, and reconstruct `X_hat = pca_k.inverse_transform(Z)`. Use `np.abs(pca_k.components_[0])` for the strongest PC1 loading.",
            3: "def pca_clinical_compression_audit(X, variance_threshold=0.85):\n    X_sc = StandardScaler().fit_transform(X)\n    full = PCA().fit(X_sc)\n    k = int(np.argmax(np.cumsum(full.explained_variance_ratio_) >= variance_threshold) + 1)\n    pca_k = PCA(n_components=k)\n    Z = pca_k.fit_transform(X_sc)\n    X_hat = pca_k.inverse_transform(Z)\n    return {\n        'n_components': k,\n        'cumulative_variance': round(float(np.sum(pca_k.explained_variance_ratio_)), 4),\n        'top_pc1_feature_idx': int(np.argmax(np.abs(pca_k.components_[0]))),\n        'reconstruction_mse': round(float(np.mean((X_sc - ___) ** 2)), 5),\n    }",
        },
        verifier=_verify_exam_03,
    )
)
