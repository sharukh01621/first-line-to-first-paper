"""Chapter 19 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch19_todo_01..06) and Coding Exam (ch19_exam_01..03)
for Statistical Visualization & Publication-Ready Figures (Matplotlib & Seaborn).
All plotting libraries (matplotlib, seaborn, pandas, numpy) are lazy-imported
inside verifiers; every Figure created during grading is explicitly closed via
`plt.close(fig)` in a `finally` block, and all file exports run inside
`tempfile.TemporaryDirectory()` so zero stray files or figure leaks remain.
"""

from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple

from ..engine import CheckResult, ExerciseSpec, register_exercise


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
            f"Expected a function (like {expected_name}), but received {type(fn).__name__}.",
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
    expected_desc: str = "(fig, ax)",
    points_possible: int = 10,
) -> CheckResult:
    return CheckResult(
        ex_id,
        False,
        "failed",
        f"{title} — Function Returned None (Did You Forget `return`?)",
        f"Calling {call_str} returned None instead of {expected_desc}.",
        details=[
            f"Plotting functions must end with `return {expected_desc}` so callers and export pipelines can inspect or save the Figure.",
            "Avoid calling `plt.show()` without returning the created objects.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="In Matplotlib's object-oriented API, always create `fig, ax = plt.subplots(...)` and `return fig, ax`.",
    )


def _unpack_fig_ax(
    result: Any,
    ex_id: str,
    title: str,
    call_str: str,
    points_possible: int = 10,
) -> Tuple[Optional[Any], Optional[Any], Optional[CheckResult]]:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if result is None:
        return None, None, _none_return_trap(ex_id, title, call_str, "(fig, ax)", points_possible)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return (
            None,
            None,
            CheckResult(
                ex_id,
                False,
                "failed",
                f"{title} — Expected a `(fig, ax)` 2-Tuple",
                f"{call_str} returned {type(result).__name__} instead of `(fig, ax)`.",
                details=["Create `fig, ax = plt.subplots(...)` and finish with `return fig, ax`."],
                points_earned=0,
                points_possible=points_possible,
            ),
        )
    fig, ax = result[0], result[1]
    if not isinstance(fig, Figure) or not isinstance(ax, Axes):
        return (
            fig if isinstance(fig, Figure) else None,
            None,
            CheckResult(
                ex_id,
                False,
                "failed",
                f"{title} — Return Real Matplotlib `(Figure, Axes)` Objects",
                f"Expected `(Figure, Axes)`, got `({type(fig).__name__}, {type(ax).__name__})`.",
                details=["Use `fig, ax = plt.subplots(...)` and return `(fig, ax)` in that exact order."],
                points_earned=0,
                points_possible=points_possible,
            ),
        )
    return fig, ax, None


def _unpack_fig_axes(
    result: Any,
    expected_len: int,
    ex_id: str,
    title: str,
    call_str: str,
    points_possible: int = 35,
) -> Tuple[Optional[Any], Optional[list], Optional[CheckResult]]:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import numpy as np

    if result is None:
        return None, None, _none_return_trap(ex_id, title, call_str, "(fig, axes)", points_possible)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return (
            None,
            None,
            CheckResult(
                ex_id,
                False,
                "failed",
                f"{title} — Expected a `(fig, axes)` 2-Tuple",
                f"{call_str} returned {type(result).__name__} instead of `(fig, axes)`.",
                details=["Use `fig, axes = plt.subplots(...)` and finish with `return fig, axes`."],
                points_earned=0,
                points_possible=points_possible,
            ),
        )
    fig, axes_raw = result[0], result[1]
    if not isinstance(fig, Figure):
        return (
            None,
            None,
            CheckResult(
                ex_id,
                False,
                "failed",
                f"{title} — First Element Must Be a Matplotlib `Figure`",
                f"Expected `Figure` as first tuple element, got `{type(fig).__name__}`.",
                points_earned=0,
                points_possible=points_possible,
            ),
        )
    if isinstance(axes_raw, np.ndarray):
        axes_list = list(axes_raw.ravel())
    elif isinstance(axes_raw, (list, tuple)):
        axes_list = list(axes_raw)
    else:
        return (
            fig,
            None,
            CheckResult(
                ex_id,
                False,
                "failed",
                f"{title} — Second Element Must Be a Collection of `Axes`",
                f"Expected an array/list of {expected_len} Axes, got `{type(axes_raw).__name__}`.",
                points_earned=0,
                points_possible=points_possible,
            ),
        )
    if len(axes_list) != expected_len or not all(isinstance(a, Axes) for a in axes_list):
        return (
            fig,
            None,
            CheckResult(
                ex_id,
                False,
                "failed",
                f"{title} — Expected Exactly {expected_len} Subplot `Axes`",
                f"Found {len(axes_list)} axes objects; expected {expected_len} Matplotlib `Axes`.",
                points_earned=0,
                points_possible=points_possible,
            ),
        )
    return fig, axes_list, None


def _png_dimensions(path: Path) -> Tuple[int, int]:
    """Read (width, height) in pixels directly from PNG IHDR header bytes."""
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return (0, 0)
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _check_labels(ax: Any, require_title: bool = True, require_x: bool = True, require_y: bool = True) -> list[str]:
    errs = []
    if require_title and not (ax.get_title() or "").strip():
        errs.append("Missing figure/panel title — call `ax.set_title(...)` with a descriptive title.")
    if require_x and not (ax.get_xlabel() or "").strip():
        errs.append("Missing x-axis label — call `ax.set_xlabel(...)` including measurement units.")
    if require_y and not (ax.get_ylabel() or "").strip():
        errs.append("Missing y-axis label — call `ax.set_ylabel(...)` describing what is measured.")
    return errs


# ---------------------------------------------------------------------------
# GUIDED TO-DO 19.1: plot_hb_distribution(df, cutoff=11.0)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_todo_01", "To-Do 19.1", fn, "plot_hb_distribution", 10)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sample_df = pd.DataFrame(
        {
            "patient_id": [f"P-{i:03d}" for i in range(1, 17)],
            "hemoglobin_g_dl": [
                9.4, 10.1, 10.5, 10.8, 11.2, 11.5, 11.9, 12.1,
                12.4, 12.8, 13.0, 13.3, 13.7, 14.1, 14.6, 15.0,
            ],
        }
    )
    snap = sample_df.copy(deep=True)

    fig1 = fig2 = None
    try:
        try:
            res1 = fn(sample_df)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_todo_01",
                False,
                "failed",
                "To-Do 19.1 — Runtime Error in `plot_hb_distribution(df)`",
                f"Calling `plot_hb_distribution(df)` raised {type(exc).__name__}: {exc}",
                details=["Plot `df['hemoglobin_g_dl']` using `ax.hist(...)` and `ax.axvline(cutoff, linestyle='--')`."],
            )

        fig1, ax1, err_res = _unpack_fig_ax(res1, "ch19_todo_01", "To-Do 19.1", "plot_hb_distribution(df)", 10)
        if err_res is not None:
            return err_res

        errors = []
        if not sample_df.equals(snap):
            errors.append("Do NOT mutate the caller's DataFrame inside `plot_hb_distribution`.")

        if len(ax1.patches) < 3:
            errors.append(
                f"Expected histogram bars (`ax.hist(...)`) in `ax.patches`, found only {len(ax1.patches)} patch(es)."
            )

        # Verify dashed vertical line at default cutoff=11.0
        has_default_cutoff = False
        solid_cutoff_found = False
        for line in ax1.lines:
            xdata = np.asarray(line.get_xdata(), dtype=float)
            if xdata.size >= 2 and np.allclose(xdata, 11.0, atol=1e-6):
                ls = line.get_linestyle()
                if ls in ("--", "dashed", "-.", ":"):
                    has_default_cutoff = True
                else:
                    solid_cutoff_found = True

        if not has_default_cutoff:
            if solid_cutoff_found:
                errors.append(
                    "Found a vertical line at `cutoff=11.0`, but it is solid — pass `linestyle='--'` to `ax.axvline(...)`."
                )
            else:
                errors.append(
                    "Missing vertical clinical threshold line at `cutoff=11.0` — call `ax.axvline(cutoff, linestyle='--', ...)`."
                )

        errors.extend(_check_labels(ax1))

        # Test custom cutoff=12.5 to catch hardcoded 11.0
        try:
            res2 = fn(sample_df, cutoff=12.5)
        except NotImplementedError:
            raise
        except TypeError as exc:
            errors.append(
                f"Function must accept an optional `cutoff=11.0` parameter; `plot_hb_distribution(df, cutoff=12.5)` raised TypeError: {exc}"
            )
        except Exception as exc:
            errors.append(f"`plot_hb_distribution(df, cutoff=12.5)` raised {type(exc).__name__}: {exc}")
        else:
            fig2, ax2, err_res2 = _unpack_fig_ax(
                res2, "ch19_todo_01", "To-Do 19.1", "plot_hb_distribution(df, cutoff=12.5)", 10
            )
            if err_res2 is not None:
                return err_res2
            has_custom_cutoff = any(
                np.asarray(ln.get_xdata(), dtype=float).size >= 2
                and np.allclose(np.asarray(ln.get_xdata(), dtype=float), 12.5, atol=1e-6)
                for ln in ax2.lines
            )
            if not has_custom_cutoff:
                errors.append(
                    "When called with `cutoff=12.5`, no vertical line appeared at x=12.5 — use the `cutoff` parameter instead of hardcoding `11.0`!"
                )

        if errors:
            return CheckResult(
                "ch19_todo_01",
                False,
                "failed",
                "To-Do 19.1 — Check Histogram, Dashed Cutoff Line & Axis Labels",
                "Draw a histogram of `hemoglobin_g_dl`, add `ax.axvline(cutoff, linestyle='--')`, label all axes, and return `(fig, ax)`.",
                details=errors,
            )

        return CheckResult(
            "ch19_todo_01",
            True,
            "passed",
            "To-Do 19.1 Passed — Clinical Hemoglobin Distribution Verified!",
            "Built OO `(fig, ax)` histogram with dynamic dashed `cutoff` reference line, clean title, and labeled axes.",
            tip="Pairing a distribution plot with a dashed clinical threshold (`ax.axvline`) lets readers spot the at-risk tail in one glance.",
        )
    finally:
        if fig1 is not None:
            plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_todo_01",
        chapter="ch19",
        title="To-Do 19.1: Hemoglobin Distribution with Clinical Cutoff (`plot_hb_distribution`)",
        book_section="Section 19.3.1 (Matplotlib OO Figure & Axes API)",
        points=10,
        hints={
            1: "Create `fig, ax = plt.subplots(...)`, call `ax.hist(df['hemoglobin_g_dl'].dropna(), ...)`, add `ax.axvline(cutoff, linestyle='--', ...)`, set title/xlabel/ylabel, and `return fig, ax`.",
            2: "Blueprint:\n1. `fig, ax = plt.subplots(figsize=(6.5, 4.0))`\n2. `ax.hist(df['hemoglobin_g_dl'].dropna(), bins=15, color='#0072B2', edgecolor='white')`\n3. `ax.axvline(cutoff, color='#D55E00', linestyle='--', linewidth=2)`\n4. Set `ax.set_title(...)`, `ax.set_xlabel(...)`, `ax.set_ylabel(...)`\n5. `return fig, ax`",
            3: "def plot_hb_distribution(df, cutoff=11.0):\n    fig, ax = plt.subplots(figsize=(6.5, 4.0))\n    ax.hist(df['hemoglobin_g_dl'].dropna(), bins=15, color='#0072B2', edgecolor='white')\n    ax.axvline(___, color='#D55E00', linestyle='--', linewidth=2)\n    ax.set_title('Hemoglobin Distribution')\n    ax.set_xlabel('Hemoglobin (g/dL)')\n    ax.set_ylabel('Patient Count')\n    return ___, ___",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 19.2: plot_district_anemia_bar(summary_df)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_todo_02", "To-Do 19.2", fn, "plot_district_anemia_bar", 10)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    summary_df = pd.DataFrame(
        {
            "district": ["Dhaka", "Sylhet", "Rajshahi", "Chattogram"],
            "anemia_pct": [28.5, 41.2, 22.0, 34.8],
        }
    )
    snap = summary_df.copy(deep=True)

    fig = None
    try:
        try:
            res = fn(summary_df)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_todo_02",
                False,
                "failed",
                "To-Do 19.2 — Runtime Error in `plot_district_anemia_bar`",
                f"Calling `plot_district_anemia_bar(summary_df)` raised {type(exc).__name__}: {exc}",
            )

        fig, ax, err_res = _unpack_fig_ax(res, "ch19_todo_02", "To-Do 19.2", "plot_district_anemia_bar(summary_df)", 10)
        if err_res is not None:
            return err_res

        errors = []
        if not summary_df.equals(snap):
            errors.append("Do NOT sort `summary_df` in place (`inplace=True`); assign a new sorted DataFrame.")

        patches = list(ax.patches)
        if len(patches) != len(summary_df):
            errors.append(
                f"Expected {len(summary_df)} horizontal bars in `ax.patches`, found {len(patches)}."
            )
        else:
            # Verify horizontal orientation (`ax.barh`) vs vertical (`ax.bar`)
            widths = [p.get_width() for p in patches]
            heights = [p.get_height() for p in patches]
            ys = [p.get_y() for p in patches]

            expected_sorted = [22.0, 28.5, 34.8, 41.2]
            if all(h > 5.0 for h in heights) and all(w <= 1.5 for w in widths):
                errors.append(
                    "It looks like you drew vertical bars (`ax.bar`) instead of horizontal bars (`ax.barh`)!"
                )
            else:
                # Order patches by ascending y position (bottom -> top)
                by_y = [w for _, w in sorted(zip(ys, widths), key=lambda pair: pair[0])]
                if np.allclose(by_y, expected_sorted[::-1], atol=1e-4):
                    errors.append(
                        "Bars are sorted descending instead of ascending — sort `ascending=True` so the highest district (Sylhet, 41.2%) sits at the TOP!"
                    )
                elif not np.allclose(by_y, expected_sorted, atol=1e-4):
                    errors.append(
                        f"Bar widths from bottom to top were {[round(v, 2) for v in by_y]}; expected ascending {expected_sorted}."
                    )

        errors.extend(_check_labels(ax))

        if errors:
            return CheckResult(
                "ch19_todo_02",
                False,
                "failed",
                "To-Do 19.2 — Check Sorted Horizontal Bar Chart",
                "Sort `summary_df` ascending by `anemia_pct`, plot with `ax.barh(...)`, label title/x/y, and return `(fig, ax)`.",
                details=errors,
            )

        return CheckResult(
            "ch19_todo_02",
            True,
            "passed",
            "To-Do 19.2 Passed — Sorted Horizontal District Bar Chart Verified!",
            "Horizontal bars sorted ascending (`ax.barh`) place the highest-burden district cleanly at the top with readable category labels.",
            tip="Horizontal bar charts eliminate rotated overlapping district labels and guide the reader's eye directly to the highest-priority region.",
        )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_todo_02",
        chapter="ch19",
        title="To-Do 19.2: Sorted Horizontal District Bar Chart (`plot_district_anemia_bar`)",
        book_section="Section 19.3.2 (Colorblind-Safe Palettes & Sorted Comparisons)",
        points=10,
        hints={
            1: "Sort `summary_df` ascending by `'anemia_pct'` into a new DataFrame, then call `ax.barh(ordered['district'], ordered['anemia_pct'])` so the largest value sits at the top.",
            2: "Blueprint:\n1. `ordered = summary_df.sort_values('anemia_pct', ascending=True)`\n2. `fig, ax = plt.subplots(figsize=(6.5, 4.0))`\n3. `ax.barh(ordered['district'], ordered['anemia_pct'], color='#009E73')`\n4. Set `ax.set_title(...)`, `ax.set_xlabel(...)`, `ax.set_ylabel(...)`\n5. `return fig, ax`",
            3: "def plot_district_anemia_bar(summary_df):\n    ordered = summary_df.sort_values('anemia_pct', ascending=___)\n    fig, ax = plt.subplots(figsize=(6.5, 4.0))\n    ax.barh(ordered['district'], ordered['anemia_pct'], color='#009E73')\n    ax.set_title('Anemia Prevalence by District')\n    ax.set_xlabel('Anemia Prevalence (%)')\n    ax.set_ylabel('District')\n    return fig, ax",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 19.3: plot_bmi_vs_sbp_scatter(df)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_todo_03", "To-Do 19.3", fn, "plot_bmi_vs_sbp_scatter", 10)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "bmi": [18.5, 20.1, 21.8, 23.4, 25.0, 26.7, 28.2, 30.5],
            "systolic_bp": [108, 114, 118, 122, 134, 140, 146, 158],
            "triage_priority": [
                "Routine", "Routine", "Routine", "Routine",
                "Moderate", "Moderate", "Urgent", "Urgent",
            ],
        }
    )
    snap = df.copy(deep=True)
    expected_groups = set(df["triage_priority"].unique())

    fig = None
    try:
        try:
            res = fn(df)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_todo_03",
                False,
                "failed",
                "To-Do 19.3 — Runtime Error in `plot_bmi_vs_sbp_scatter`",
                f"Calling `plot_bmi_vs_sbp_scatter(df)` raised {type(exc).__name__}: {exc}",
            )

        fig, ax, err_res = _unpack_fig_ax(res, "ch19_todo_03", "To-Do 19.3", "plot_bmi_vs_sbp_scatter(df)", 10)
        if err_res is not None:
            return err_res

        errors = []
        if not df.equals(snap):
            errors.append("Do NOT mutate the input DataFrame inside `plot_bmi_vs_sbp_scatter`.")

        # Count plotted scatter points across PathCollections
        all_offsets = []
        for coll in ax.collections:
            if hasattr(coll, "get_offsets"):
                offs = np.asarray(coll.get_offsets())
                if offs.ndim == 2 and offs.shape[1] == 2 and offs.shape[0] > 0:
                    all_offsets.append(offs)

        total_pts = sum(o.shape[0] for o in all_offsets)
        if total_pts != len(df):
            errors.append(
                f"Expected {len(df)} scatter points across `ax.collections`, found {total_pts}."
            )
        else:
            stacked = np.vstack(all_offsets)
            xs_sorted = np.sort(stacked[:, 0])
            ys_sorted = np.sort(stacked[:, 1])
            if not (
                np.allclose(xs_sorted, np.sort(df["bmi"].to_numpy(dtype=float)), atol=1e-4)
                and np.allclose(ys_sorted, np.sort(df["systolic_bp"].to_numpy(dtype=float)), atol=1e-4)
            ):
                errors.append(
                    "Scatter coordinates do not match `x='bmi'` and `y='systolic_bp'` — check axis order!"
                )

        legend = ax.get_legend()
        if legend is None:
            errors.append(
                "Missing legend — pass `label=priority` to each `ax.scatter(...)` and call `ax.legend()`, or use `sns.scatterplot(..., hue='triage_priority', ax=ax)`."
            )
        else:
            legend_labels = {t.get_text() for t in legend.get_texts()}
            if not expected_groups.issubset(legend_labels):
                errors.append(
                    f"Legend labels {sorted(legend_labels)} do not include all `triage_priority` groups {sorted(expected_groups)}."
                )

        errors.extend(_check_labels(ax))

        if errors:
            return CheckResult(
                "ch19_todo_03",
                False,
                "failed",
                "To-Do 19.3 — Check Grouped Scatter Plot, Legend & Labels",
                "Plot `bmi` vs `systolic_bp` grouped/colored by `triage_priority` with a legend and labeled axes.",
                details=errors,
            )

        return CheckResult(
            "ch19_todo_03",
            True,
            "passed",
            "To-Do 19.3 Passed — Grouped Clinical Scatter Plot Verified!",
            "Plotted all BMI vs. systolic BP points grouped by `triage_priority` with a clear legend and labeled axes.",
            tip="Encoding a third categorical clinical variable via color (using a colorblind-safe palette like Okabe-Ito) reveals risk clusters immediately.",
        )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_todo_03",
        chapter="ch19",
        title="To-Do 19.3: Grouped BMI vs. Systolic BP Scatter (`plot_bmi_vs_sbp_scatter`)",
        book_section="Section 19.3.2 (Colorblind-Safe Palettes & Grouped Scatter)",
        points=10,
        hints={
            1: "Create `fig, ax = plt.subplots(...)`, loop over `df.groupby('triage_priority')` calling `ax.scatter(sub['bmi'], sub['systolic_bp'], label=priority)` (or use `sns.scatterplot(..., hue='triage_priority', ax=ax)`), add `ax.legend()`, and set title/xlabel/ylabel.",
            2: "Blueprint:\n1. `fig, ax = plt.subplots(figsize=(6.5, 4.2))`\n2. Define an Okabe-Ito palette list (`['#0072B2', '#E69F00', '#D55E00', '#009E73']`)\n3. For each `(priority, sub)` in `df.groupby('triage_priority')`: `ax.scatter(sub['bmi'], sub['systolic_bp'], label=priority, alpha=0.8)`\n4. Call `ax.legend(title='Triage Priority')`, set title & x/y labels, `return fig, ax`",
            3: "def plot_bmi_vs_sbp_scatter(df):\n    fig, ax = plt.subplots(figsize=(6.5, 4.2))\n    palette = ['#0072B2', '#E69F00', '#D55E00', '#009E73']\n    for idx, (priority, sub) in enumerate(df.groupby('triage_priority')):\n        ax.scatter(sub['bmi'], sub['systolic_bp'], color=palette[idx % len(palette)], label=___, alpha=0.8)\n    ax.legend(title='Triage Priority')\n    ax.set_title('BMI vs. Systolic Blood Pressure by Triage Priority')\n    ax.set_xlabel('BMI (kg/m^2)')\n    ax.set_ylabel('Systolic BP (mmHg)')\n    return fig, ax",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 19.4: plot_dengue_seasonality(epi_df, district)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_todo_04", "To-Do 19.4", fn, "plot_dengue_seasonality", 10)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    epi_df = pd.DataFrame(
        {
            "year": [2024] * 8 + [2024] * 8,
            "epi_week": list(range(1, 9)) + list(range(1, 9)),
            "district": ["Dhaka"] * 8 + ["Sylhet"] * 8,
            "dengue_cases": [
                20, 30, 50, 100, 140, 180, 120, 80,
                5, 8, 12, 15, 22, 18, 10, 6,
            ],
            "rainfall_mm": [
                10.0, 18.0, 45.0, 95.0, 130.0, 110.0, 60.0, 25.0,
                25.0, 40.0, 65.0, 80.0, 90.0, 75.0, 50.0, 30.0,
            ],
        }
    )
    snap = epi_df.copy(deep=True)

    fig1 = fig2 = None
    try:
        try:
            res1 = fn(epi_df, "Dhaka")
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_todo_04",
                False,
                "failed",
                "To-Do 19.4 — Runtime Error in `plot_dengue_seasonality`",
                f"Calling `plot_dengue_seasonality(epi_df, 'Dhaka')` raised {type(exc).__name__}: {exc}",
            )

        fig1, ax1, err_res = _unpack_fig_ax(
            res1, "ch19_todo_04", "To-Do 19.4", "plot_dengue_seasonality(epi_df, 'Dhaka')", 10
        )
        if err_res is not None:
            return err_res

        errors = []
        if not epi_df.equals(snap):
            errors.append("Do NOT mutate `epi_df` in place inside `plot_dengue_seasonality`.")

        dhaka_raw = np.array([20, 30, 50, 100, 140, 180, 120, 80], dtype=float)
        dhaka_roll_tail = pd.Series(dhaka_raw).rolling(4).mean().to_numpy()[3:]  # [50, 80, 117.5, 135, 130]

        if len(ax1.lines) < 2:
            errors.append(
                f"Expected at least 2 plotted lines (weekly dengue cases + 4-week rolling mean), found {len(ax1.lines)}."
            )
        else:
            raw_matched = False
            roll_matched = False
            for ln in ax1.lines:
                ydata = np.asarray(ln.get_ydata(), dtype=float)
                if ydata.size == dhaka_raw.size and np.allclose(ydata, dhaka_raw, atol=1e-4, equal_nan=False):
                    raw_matched = True
                if ydata.size == dhaka_raw.size and np.allclose(ydata[3:], dhaka_roll_tail, atol=1e-4):
                    roll_matched = True
            if not raw_matched:
                errors.append(
                    "Could not find a line matching Dhaka's weekly `dengue_cases` — ensure you filter `epi_df['district'] == district` and sort by `(year, epi_week)`."
                )
            if not roll_matched:
                errors.append(
                    "Could not find a line matching the 4-week rolling mean (`sub['dengue_cases'].rolling(4).mean()` or `min_periods=1`)."
                )

        legend = ax1.get_legend()
        if legend is None or len(legend.get_texts()) < 2:
            errors.append("Add a legend labeling both the weekly series and the 4-week rolling mean (`ax.legend()`).")

        errors.extend(_check_labels(ax1))

        # Test second district to prevent hardcoded 'Dhaka'
        try:
            res2 = fn(epi_df, "Sylhet")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"`plot_dengue_seasonality(epi_df, 'Sylhet')` raised {type(exc).__name__}: {exc}")
        else:
            fig2, ax2, err_res2 = _unpack_fig_ax(
                res2, "ch19_todo_04", "To-Do 19.4", "plot_dengue_seasonality(epi_df, 'Sylhet')", 10
            )
            if err_res2 is not None:
                return err_res2
            sylhet_raw = np.array([5, 8, 12, 15, 22, 18, 10, 6], dtype=float)
            sylhet_ok = any(
                np.asarray(ln.get_ydata(), dtype=float).size == sylhet_raw.size
                and np.allclose(np.asarray(ln.get_ydata(), dtype=float), sylhet_raw, atol=1e-4)
                for ln in ax2.lines
            )
            if not sylhet_ok:
                errors.append(
                    "When called with `district='Sylhet'`, the plotted line did not match Sylhet's dengue cases — make sure you use the `district` argument!"
                )

        if errors:
            return CheckResult(
                "ch19_todo_04",
                False,
                "failed",
                "To-Do 19.4 — Check District Filter, Rolling Mean & Legend",
                "Filter to `district`, plot weekly `dengue_cases` plus a 4-week rolling mean with a legend and labeled axes.",
                details=errors,
            )

        return CheckResult(
            "ch19_todo_04",
            True,
            "passed",
            "To-Do 19.4 Passed — Dengue Seasonality & Rolling Mean Verified!",
            "Filtered dynamically across districts, overlaid weekly counts with a 4-week rolling mean, and added a clean legend.",
            tip="Plotting raw weekly surveillance counts semi-transparently behind a bold 4-week rolling mean exposes seasonality without hiding true volatility.",
        )
    finally:
        if fig1 is not None:
            plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_todo_04",
        chapter="ch19",
        title="To-Do 19.4: Dengue Seasonality + 4-Week Rolling Mean (`plot_dengue_seasonality`)",
        book_section="Section 19.3.3 (Time-Series Trends, Annotations & Legends)",
        points=10,
        hints={
            1: "Filter `sub = epi_df[epi_df['district'] == district].sort_values(['year', 'epi_week'])`, compute `rolling_4w = sub['dengue_cases'].rolling(4, min_periods=1).mean()`, plot both lines on `ax` with labels, and call `ax.legend()`.",
            2: "Blueprint:\n1. `sub = epi_df[epi_df['district'] == district].sort_values(['year', 'epi_week']).reset_index(drop=True)`\n2. `x = range(1, len(sub) + 1)`\n3. `roll = sub['dengue_cases'].rolling(4, min_periods=1).mean()`\n4. Plot weekly `sub['dengue_cases']` (light/alpha) and `roll` (bold), add `ax.legend()`, title, x/y labels, `return fig, ax`",
            3: "def plot_dengue_seasonality(epi_df, district):\n    sub = epi_df[epi_df['district'] == district].sort_values(['year', 'epi_week']).reset_index(drop=True)\n    x = range(1, len(sub) + 1)\n    roll = sub['dengue_cases'].rolling(___, min_periods=1).mean()\n    fig, ax = plt.subplots(figsize=(7.0, 4.0))\n    ax.plot(x, sub['dengue_cases'], color='#56B4E9', alpha=0.65, label='Weekly Cases')\n    ax.plot(x, roll, color='#D55E00', linewidth=2.2, label='4-Week Rolling Mean')\n    ax.legend()\n    ax.set_title(f'Weekly Dengue Seasonality — {district}')\n    ax.set_xlabel('Sequential Epi Week')\n    ax.set_ylabel('Dengue Cases')\n    return fig, ax",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 19.5: plot_correlation_heatmap(df, cols)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_todo_05", "To-Do 19.5", fn, "plot_correlation_heatmap", 10)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "age_years": [24, 31, 45, 52, 60, 67],
            "systolic_bp": [112, 118, 129, 138, 148, 155],
            "hemoglobin_g_dl": [13.8, 12.1, 13.4, 11.9, 12.6, 11.2],
            "extra_ignored": [1, 2, 1, 2, 1, 2],
        }
    )
    cols = ["age_years", "systolic_bp", "hemoglobin_g_dl"]
    expected_corr = df[cols].corr().to_numpy(dtype=float)

    fig = None
    try:
        try:
            res = fn(df, cols)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_todo_05",
                False,
                "failed",
                "To-Do 19.5 — Runtime Error in `plot_correlation_heatmap`",
                f"Calling `plot_correlation_heatmap(df, cols)` raised {type(exc).__name__}: {exc}",
            )

        fig, ax, err_res = _unpack_fig_ax(
            res, "ch19_todo_05", "To-Do 19.5", "plot_correlation_heatmap(df, cols)", 10
        )
        if err_res is not None:
            return err_res

        errors = []
        if not ax.collections:
            errors.append("Expected a Seaborn heatmap (`sns.heatmap(df[cols].corr(), annot=True, center=0, ax=ax)`), found no QuadMesh collection.")
        else:
            mesh = ax.collections[0]
            arr = np.asarray(mesh.get_array(), dtype=float)
            if arr.size != expected_corr.size:
                errors.append(
                    f"Heatmap has {arr.size} cells; expected {expected_corr.size} for {len(cols)}x{len(cols)} correlation matrix `df[cols].corr()`."
                )
            elif not np.allclose(arr.reshape(expected_corr.shape), expected_corr, atol=1e-4):
                errors.append("Heatmap cell values do not match `df[cols].corr()`.")

            if hasattr(mesh, "norm") and mesh.norm is not None:
                vmin = float(mesh.norm.vmin)
                vmax = float(mesh.norm.vmax)
                cmap_name = getattr(getattr(mesh, "cmap", None), "name", "")
                is_centered = (cmap_name == "unnamed") or math.isclose(vmin + vmax, 0.0, abs_tol=1e-4)
                if not is_centered:
                    errors.append(
                        f"Colormap is not centered at 0 (found vmin={vmin:.2f}, vmax={vmax:.2f}, cmap={cmap_name!r}) — pass `center=0` to `sns.heatmap(...)`!"
                    )

        if len(ax.texts) < len(cols) * len(cols):
            errors.append(
                f"Expected numeric cell annotations (`annot=True`) for all {len(cols) * len(cols)} cells, found {len(ax.texts)} text label(s)."
            )

        if not (ax.get_title() or "").strip():
            errors.append("Missing heatmap title — call `ax.set_title(...)`.")

        if errors:
            return CheckResult(
                "ch19_todo_05",
                False,
                "failed",
                "To-Do 19.5 — Check Seaborn Correlation Heatmap Settings",
                "Plot `sns.heatmap(df[cols].corr(), annot=True, center=0, ax=ax)` with a clear title and return `(fig, ax)`.",
                details=errors,
            )

        return CheckResult(
            "ch19_todo_05",
            True,
            "passed",
            "To-Do 19.5 Passed — Centered Annotated Correlation Heatmap Verified!",
            "Verified 3x3 correlation values, numeric annotations (`annot=True`), zero-centered diverging scale (`center=0`), and title.",
            tip="Always pass `center=0` with a diverging colormap (like `'coolwarm'` or `'RdBu_r'`) on correlation matrices so zero correlation maps to neutral white.",
        )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_todo_05",
        chapter="ch19",
        title="To-Do 19.5: Annotated Zero-Centered Correlation Heatmap (`plot_correlation_heatmap`)",
        book_section="Section 19.3.5 (Seaborn Statistical Matrices & Heatmaps)",
        points=10,
        hints={
            1: "Compute `corr = df[cols].corr()`, create `fig, ax = plt.subplots(...)`, call `sns.heatmap(corr, annot=True, center=0, cmap='coolwarm', ax=ax)`, set `ax.set_title(...)`, and `return fig, ax`.",
            2: "Blueprint:\n1. `corr = df[cols].corr()`\n2. `fig, ax = plt.subplots(figsize=(5.8, 4.8))`\n3. `sns.heatmap(corr, annot=True, fmt='.2f', center=0, cmap='coolwarm', ax=ax)`\n4. `ax.set_title('Clinical Feature Correlation Matrix')`\n5. `return fig, ax`",
            3: "def plot_correlation_heatmap(df, cols):\n    corr = df[cols].corr()\n    fig, ax = plt.subplots(figsize=(5.8, 4.8))\n    sns.heatmap(corr, annot=___, fmt='.2f', center=___, cmap='coolwarm', ax=ax)\n    ax.set_title('Clinical Feature Correlation Matrix')\n    return fig, ax",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 19.6: save_publication_figure(fig, out_path, dpi=300)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_todo_06", "To-Do 19.6", fn, "save_publication_figure", 10)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.plot([1, 2, 3], [2, 4, 3])
        ax.set_title("Publication Export Test")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        with tempfile.TemporaryDirectory() as tmpdir:
            png_300 = Path(tmpdir) / "fig_300.png"
            png_72 = Path(tmpdir) / "fig_72.png"
            pdf_out = Path(tmpdir) / "fig_vec.pdf"

            try:
                size_300 = fn(fig, png_300, dpi=300)
            except NotImplementedError:
                raise
            except Exception as exc:
                return CheckResult(
                    "ch19_todo_06",
                    False,
                    "failed",
                    "To-Do 19.6 — Runtime Error in `save_publication_figure`",
                    f"Calling `save_publication_figure(fig, out_path, dpi=300)` raised {type(exc).__name__}: {exc}",
                )

            if size_300 is None:
                return _none_return_trap(
                    "ch19_todo_06", "To-Do 19.6", "save_publication_figure(fig, out_path, dpi=300)", "int (file size in bytes)", 10
                )

            errors = []
            if not png_300.exists():
                errors.append(f"Expected file to be written to {png_300.name}, but it does not exist.")
            else:
                actual_300 = png_300.stat().st_size
                if not isinstance(size_300, int) or isinstance(size_300, bool):
                    errors.append(f"Return value must be an `int` (bytes), got {type(size_300).__name__} ({size_300!r}).")
                elif size_300 != actual_300 or size_300 <= 0:
                    errors.append(f"Returned {size_300} bytes, but file on disk is {actual_300} bytes.")

                w300, h300 = _png_dimensions(png_300)
                if w300 == 0 or h300 == 0:
                    errors.append("Saved PNG file did not contain valid PNG header bytes.")
                elif w300 < 900:
                    errors.append(
                        f"Expected 300-DPI width >= 900 px for a 4-inch figure, got {w300} px — pass `dpi=dpi` to `fig.savefig(...)`!"
                    )

            # Check custom dpi=72 scales pixel width down cleanly
            try:
                size_72 = fn(fig, png_72, dpi=72)
                size_pdf = fn(fig, pdf_out, dpi=300)
            except NotImplementedError:
                raise
            except Exception as exc:
                errors.append(f"Secondary export check (dpi=72 / PDF) raised {type(exc).__name__}: {exc}")
            else:
                if png_72.exists() and png_300.exists():
                    w72, _ = _png_dimensions(png_72)
                    w300, _ = _png_dimensions(png_300)
                    if w72 >= w300:
                        errors.append(
                            f"PNG width at dpi=72 ({w72}px) was not smaller than dpi=300 ({w300}px) — forward `dpi=dpi` rather than hardcoding!"
                        )
                if not pdf_out.exists() or not pdf_out.read_bytes().startswith(b"%PDF"):
                    errors.append("Saving a `.pdf` path should produce a valid vector PDF starting with `%PDF`.")
                elif not isinstance(size_pdf, int) or size_pdf <= 0:
                    errors.append(f"PDF export returned invalid byte size: {size_pdf!r}.")

            if errors:
                return CheckResult(
                    "ch19_todo_06",
                    False,
                    "failed",
                    "To-Do 19.6 — Check Publication Figure Export & Byte Count",
                    "Call `fig.savefig(out_path, dpi=dpi, bbox_inches='tight')` and return `int(Path(out_path).stat().st_size)`.",
                    details=errors,
                )

            return CheckResult(
                "ch19_todo_06",
                True,
                "passed",
                "To-Do 19.6 Passed — 300-DPI & Vector PDF Export Verified!",
                f"Exported crisp 300-DPI PNG ({size_300:,} bytes) and vector PDF cleanly inside a temporary sandbox.",
                tip="Journals require >=300 DPI for raster PNG/TIFF figures and vector PDF/SVG for line art — `bbox_inches='tight'` prevents clipped axis labels.",
            )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_todo_06",
        chapter="ch19",
        title="To-Do 19.6: Publication-Grade Figure Exporter (`save_publication_figure`)",
        book_section="Section 19.3.6 (300-DPI Raster & Vector PDF Export)",
        points=10,
        hints={
            1: "Convert `out_path` to `Path(out_path)`, call `fig.savefig(path, dpi=dpi, bbox_inches='tight')`, and return `int(path.stat().st_size)`.",
            2: "Blueprint:\n1. `path = Path(out_path)`\n2. `path.parent.mkdir(parents=True, exist_ok=True)`\n3. `fig.savefig(path, dpi=dpi, bbox_inches='tight')`\n4. `return int(path.stat().st_size)`",
            3: "from pathlib import Path\n\ndef save_publication_figure(fig, out_path, dpi=300):\n    path = Path(out_path)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    fig.savefig(path, dpi=___, bbox_inches='tight')\n    return int(path.stat().st_size)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 19.1 (35 pts): build_mch_equity_panel(mch_df)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_exam_01", "Exam Task 19.1", fn, "build_mch_equity_panel", 35)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    rows = []
    # Construct deterministic survey data with unmistakable percentages per quintile
    spec_data = [
        ("Poorest", 10, 4, 6),   # 40% stunted, 60% anemia
        ("Poorer", 10, 3, 5),    # 30% stunted, 50% anemia
        ("Middle", 10, 2, 4),    # 20% stunted, 40% anemia
        ("Richer", 10, 1, 3),    # 10% stunted, 30% anemia
        ("Richest", 10, 1, 2),   # 10% stunted, 20% anemia
    ]
    for q, n, n_stunt, n_anem in spec_data:
        for i in range(n):
            rows.append(
                {
                    "wealth_quintile": q,
                    "child_stunted": 1 if i < n_stunt else 0,
                    "maternal_anemia": 1 if i < n_anem else 0,
                }
            )
    mch_df = pd.DataFrame(rows)
    snap = mch_df.copy(deep=True)
    exp_stunt = {"Poorest": 40.0, "Poorer": 30.0, "Middle": 20.0, "Richer": 10.0, "Richest": 10.0}
    exp_anem = {"Poorest": 60.0, "Poorer": 50.0, "Middle": 40.0, "Richer": 30.0, "Richest": 20.0}

    fig = None
    try:
        try:
            res = fn(mch_df)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_exam_01",
                False,
                "failed",
                "Exam Task 19.1 — Runtime Error in `build_mch_equity_panel`",
                f"Calling `build_mch_equity_panel(mch_df)` raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        fig, axes, err_res = _unpack_fig_axes(
            res, 2, "ch19_exam_01", "Exam Task 19.1", "build_mch_equity_panel(mch_df)", 35
        )
        if err_res is not None:
            return err_res

        errors = []
        if not mch_df.equals(snap):
            errors.append("Do NOT mutate `mch_df` inside `build_mch_equity_panel`.")

        geom = axes[0].get_subplotspec().get_gridspec().get_geometry()
        if geom != (1, 2):
            errors.append(f"Expected a 1x2 side-by-side grid `plt.subplots(1, 2, ...)`, got geometry {geom}.")

        suptitle = (fig.get_suptitle() or "").strip()
        if not suptitle:
            errors.append("Missing figure-level super-title — call `fig.suptitle(...)`.")

        def _extract_bar_vals(ax_obj: Any) -> list[float]:
            vals = []
            for p in ax_obj.patches:
                w, h = p.get_width(), p.get_height()
                # Pick the data-value dimension (bar height for vertical, bar width for horizontal)
                vals.append(h if h > w else w)
            return vals

        for idx, (panel_name, exp_map) in enumerate(
            [("Left panel (axes[0] — stunting %)", exp_stunt), ("Right panel (axes[1] — anemia %)", exp_anem)]
        ):
            ax_i = axes[idx]
            for msg in _check_labels(ax_i):
                errors.append(f"{panel_name}: {msg}")

            bar_vals = _extract_bar_vals(ax_i)
            if len(bar_vals) != 5:
                errors.append(f"{panel_name}: expected 5 bars (one per wealth quintile), found {len(bar_vals)}.")
            else:
                if max(bar_vals) <= 1.05:
                    errors.append(
                        f"{panel_name}: bar values look like 0–1 proportions ({[round(v, 2) for v in bar_vals]}); multiply by 100 to plot percentages (%)!"
                    )
                elif not np.allclose(sorted(bar_vals), sorted(exp_map.values()), atol=1e-2):
                    errors.append(
                        f"{panel_name}: bar values {[round(v, 1) for v in bar_vals]} did not match expected percentages {list(exp_map.values())}."
                    )

        if errors:
            return CheckResult(
                "ch19_exam_01",
                False,
                "failed",
                "Exam Task 19.1 — Check 1x2 Equity Panel Layout, Percentages & Labels",
                "Create a 1x2 figure (`plt.subplots(1, 2, ...)`), plot stunting % on `axes[0]` and anemia % on `axes[1]` across wealth quintiles, add panel titles/labels + `fig.suptitle(...)`, and return `(fig, axes)`.",
                details=errors,
                points_earned=0,
                points_possible=35,
            )

        return CheckResult(
            "ch19_exam_01",
            True,
            "passed",
            "Exam Task 19.1 Passed (35/35) — 1x2 MCH Health Equity Panel Verified!",
            "Built 1x2 multi-panel equity figure with accurate stunting % and anemia % across wealth quintiles, labeled panels, and figure suptitle.",
            points_earned=35,
            points_possible=35,
            tip="Side-by-side 1x2 equity panels with consistent wealth-quintile ordering are a staple of Lancet and BMJ Global Health papers.",
        )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_exam_01",
        chapter="ch19",
        title="Exam Task 19.1: 1x2 MCH Equity Panel (`build_mch_equity_panel`)",
        book_section="Section 19.3.4 (Multi-Panel Subplots & Shared Grids)",
        points=35,
        is_exam=True,
        hints={
            1: "Group `mch_df` by `'wealth_quintile'`, compute mean of `'child_stunted'` and `'maternal_anemia'` times `100`, reindex in `['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']` order, plot on `fig, axes = plt.subplots(1, 2, ...)`, add titles/labels + `fig.suptitle(...)`, and return `(fig, axes)`.",
            2: "Blueprint:\n1. `order = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']`\n2. `summary = mch_df.groupby('wealth_quintile')[['child_stunted', 'maternal_anemia']].mean().reindex(order) * 100`\n3. `fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))`\n4. Plot `summary['child_stunted']` on `axes[0]` and `summary['maternal_anemia']` on `axes[1]`, label both panels, call `fig.suptitle(...)`, `return fig, axes`",
            3: "def build_mch_equity_panel(mch_df):\n    order = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']\n    summary = mch_df.groupby('wealth_quintile')[['child_stunted', 'maternal_anemia']].mean().reindex(order) * 100\n    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))\n    axes[0].bar(summary.index, summary['child_stunted'], color='#0072B2')\n    axes[0].set_title('Child Stunting by Wealth Quintile')\n    axes[0].set_xlabel('Wealth Quintile')\n    axes[0].set_ylabel('Prevalence (%)')\n    axes[1].bar(summary.index, summary['maternal_anemia'], color='#D55E00')\n    axes[1].set_title('Maternal Anemia by Wealth Quintile')\n    axes[1].set_xlabel('Wealth Quintile')\n    axes[1].set_ylabel('Prevalence (%)')\n    fig.suptitle('Maternal & Child Health Equity Across Wealth Quintiles')\n    return fig, axes",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 19.2 (35 pts): build_outbreak_dashboard_figure(epi_df, district)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_exam_02", "Exam Task 19.2", fn, "build_outbreak_dashboard_figure", 35)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    epi_df = pd.DataFrame(
        {
            "year": [2024] * 6 + [2024] * 6,
            "epi_week": [1, 2, 3, 4, 5, 6] * 2,
            "district": ["Dhaka"] * 6 + ["Chattogram"] * 6,
            "dengue_cases": [15, 25, 45, 75, 115, 95, 10, 14, 18, 26, 30, 22],
            "rainfall_mm": [12.0, 28.0, 64.0, 120.0, 95.0, 40.0, 30.0, 55.0, 90.0, 110.0, 85.0, 50.0],
        }
    )
    snap = epi_df.copy(deep=True)

    fig = None
    try:
        try:
            res = fn(epi_df, "Dhaka")
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch19_exam_02",
                False,
                "failed",
                "Exam Task 19.2 — Runtime Error in `build_outbreak_dashboard_figure`",
                f"Calling `build_outbreak_dashboard_figure(epi_df, 'Dhaka')` raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        fig, axes, err_res = _unpack_fig_axes(
            res, 2, "ch19_exam_02", "Exam Task 19.2", "build_outbreak_dashboard_figure(epi_df, 'Dhaka')", 35
        )
        if err_res is not None:
            return err_res

        errors = []
        if not epi_df.equals(snap):
            errors.append("Do NOT mutate `epi_df` inside `build_outbreak_dashboard_figure`.")

        geom = axes[0].get_subplotspec().get_gridspec().get_geometry()
        if geom != (2, 1):
            errors.append(f"Expected a 2x1 stacked layout `plt.subplots(2, 1, sharex=True, ...)`, got geometry {geom}.")

        if not axes[0].get_shared_x_axes().joined(axes[0], axes[1]):
            errors.append("Top and bottom panels must share the x-axis — pass `sharex=True` to `plt.subplots(2, 1, sharex=True, ...)`.")

        # Top panel checks: dengue_cases + 4-week rolling mean + legend + title + ylabel
        ax_top, ax_bot = axes[0], axes[1]
        exp_cases = np.array([15, 25, 45, 75, 115, 95], dtype=float)
        exp_roll_tail = pd.Series(exp_cases).rolling(4).mean().to_numpy()[3:]  # [40.0, 65.0, 82.5]

        if len(ax_top.lines) < 2:
            errors.append(f"Top panel (`axes[0]`): expected >= 2 lines (weekly dengue cases + 4-week rolling mean), found {len(ax_top.lines)}.")
        else:
            raw_ok = any(
                np.asarray(ln.get_ydata(), dtype=float).size == exp_cases.size
                and np.allclose(np.asarray(ln.get_ydata(), dtype=float), exp_cases, atol=1e-4)
                for ln in ax_top.lines
            )
            roll_ok = any(
                np.asarray(ln.get_ydata(), dtype=float).size == exp_cases.size
                and np.allclose(np.asarray(ln.get_ydata(), dtype=float)[3:], exp_roll_tail, atol=1e-4)
                for ln in ax_top.lines
            )
            if not raw_ok:
                errors.append("Top panel (`axes[0]`): could not match filtered weekly `dengue_cases` for Dhaka.")
            if not roll_ok:
                errors.append("Top panel (`axes[0]`): could not match 4-week rolling mean of `dengue_cases`.")

        top_legend = ax_top.get_legend()
        if top_legend is None or len(top_legend.get_texts()) < 2:
            errors.append("Top panel (`axes[0]`): add a legend identifying weekly cases and the 4-week rolling mean.")

        for msg in _check_labels(ax_top, require_title=True, require_x=False, require_y=True):
            errors.append(f"Top panel (`axes[0]`): {msg}")

        # Bottom panel checks: rainfall_mm bar plot + title + xlabel + ylabel
        exp_rain = np.array([12.0, 28.0, 64.0, 120.0, 95.0, 40.0], dtype=float)
        bot_patches = list(ax_bot.patches)
        if len(bot_patches) != len(exp_rain):
            errors.append(f"Bottom panel (`axes[1]`): expected {len(exp_rain)} rainfall bars via `axes[1].bar(...)`, found {len(bot_patches)}.")
        else:
            heights = np.array([p.get_height() for p in bot_patches], dtype=float)
            if not np.allclose(heights, exp_rain, atol=1e-4):
                errors.append(
                    f"Bottom panel (`axes[1]`): bar heights {[round(float(h), 1) for h in heights]} do not match Dhaka `rainfall_mm` {exp_rain.tolist()}."
                )

        for msg in _check_labels(ax_bot, require_title=True, require_x=True, require_y=True):
            errors.append(f"Bottom panel (`axes[1]`): {msg}")

        if errors:
            return CheckResult(
                "ch19_exam_02",
                False,
                "failed",
                "Exam Task 19.2 — Check 2x1 Shared-X Outbreak Dashboard",
                "Use `plt.subplots(2, 1, sharex=True, ...)`: plot dengue cases + 4-week rolling mean (with legend) on `axes[0]` and `rainfall_mm` bars on `axes[1]`.",
                details=errors,
                points_earned=0,
                points_possible=35,
            )

        return CheckResult(
            "ch19_exam_02",
            True,
            "passed",
            "Exam Task 19.2 Passed (35/35) — 2x1 Shared-X Outbreak Dashboard Verified!",
            "Aligned weekly dengue incidence + 4-week trend over monsoon rainfall bars with shared x-axis, legend, and clean labels.",
            points_earned=35,
            points_possible=35,
            tip="Stacking climate drivers (`rainfall_mm`) directly below disease incidence with `sharex=True` reveals lag dynamics without misleading dual y-axes!",
        )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_exam_02",
        chapter="ch19",
        title="Exam Task 19.2: 2x1 Outbreak Dashboard (`build_outbreak_dashboard_figure`)",
        book_section="Section 19.3.4 (Multi-Panel Subplots & Shared-X Alignment)",
        points=35,
        is_exam=True,
        hints={
            1: "Filter `epi_df` to `district` sorted by `['year', 'epi_week']`, build `fig, axes = plt.subplots(2, 1, sharex=True, ...)`, plot weekly cases + 4-week rolling mean with legend on `axes[0]`, and plot `rainfall_mm` bars on `axes[1]`.",
            2: "Blueprint:\n1. `sub = epi_df[epi_df['district'] == district].sort_values(['year', 'epi_week']).reset_index(drop=True)`\n2. `x = range(1, len(sub) + 1)`\n3. `roll = sub['dengue_cases'].rolling(4, min_periods=1).mean()`\n4. `fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.0, 5.5))`\n5. Top `axes[0]`: plot cases + roll + legend + title + ylabel; Bottom `axes[1]`: `bar(x, sub['rainfall_mm'])` + title + xlabel + ylabel",
            3: "def build_outbreak_dashboard_figure(epi_df, district):\n    sub = epi_df[epi_df['district'] == district].sort_values(['year', 'epi_week']).reset_index(drop=True)\n    x = range(1, len(sub) + 1)\n    roll = sub['dengue_cases'].rolling(4, min_periods=1).mean()\n    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.0, 5.5))\n    axes[0].plot(x, sub['dengue_cases'], color='#56B4E9', alpha=0.65, label='Weekly Cases')\n    axes[0].plot(x, roll, color='#D55E00', linewidth=2.2, label='4-Week Rolling Mean')\n    axes[0].legend()\n    axes[0].set_title(f'Dengue Incidence — {district}')\n    axes[0].set_ylabel('Dengue Cases')\n    axes[1].bar(x, sub['rainfall_mm'], color='#0072B2', alpha=0.8)\n    axes[1].set_title(f'Weekly Rainfall — {district}')\n    axes[1].set_xlabel('Sequential Epi Week')\n    axes[1].set_ylabel('Rainfall (mm)')\n    return fig, axes",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 19.3 (30 pts): export_figure_bundle(fig, out_dir, stem, dpi=300)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch19_exam_03", "Exam Task 19.3", fn, "export_figure_bundle", 30)
    if bad is not None:
        return bad

    import matplotlib.pyplot as plt

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
        ax.bar(["A", "B", "C"], [10, 22, 16])
        ax.set_title("Bundle Export Check")
        ax.set_xlabel("Group")
        ax.set_ylabel("Value")

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "manuscript" / "figures"
            low_dir = Path(tmpdir) / "low_dpi"

            try:
                bundle = fn(fig, nested_dir, "fig1_equity", dpi=300)
            except NotImplementedError:
                raise
            except Exception as exc:
                return CheckResult(
                    "ch19_exam_03",
                    False,
                    "failed",
                    "Exam Task 19.3 — Runtime Error in `export_figure_bundle`",
                    f"Calling `export_figure_bundle(fig, out_dir, 'fig1_equity', dpi=300)` raised {type(exc).__name__}: {exc}",
                    points_earned=0,
                    points_possible=30,
                )

            if bundle is None:
                return _none_return_trap(
                    "ch19_exam_03",
                    "Exam Task 19.3",
                    "export_figure_bundle(fig, out_dir, 'fig1_equity', dpi=300)",
                    "dict with keys {'png', 'pdf', 'png_bytes', 'pdf_bytes'}",
                    30,
                )

            errors = []
            expected_keys = {"png", "pdf", "png_bytes", "pdf_bytes"}
            if not isinstance(bundle, dict):
                errors.append(f"Expected a `dict`, got `{type(bundle).__name__}`.")
            else:
                if set(bundle.keys()) != expected_keys:
                    errors.append(
                        f"Expected exact keys {sorted(expected_keys)}, got {sorted(bundle.keys())}."
                    )
                else:
                    png_p = bundle["png"]
                    pdf_p = bundle["pdf"]
                    png_b = bundle["png_bytes"]
                    pdf_b = bundle["pdf_bytes"]

                    if not isinstance(png_p, Path):
                        errors.append(f"`bundle['png']` must be a `pathlib.Path`, got `{type(png_p).__name__}`.")
                    elif png_p.name != "fig1_equity.png" or not png_p.exists():
                        errors.append(f"`bundle['png']` should point to existing `fig1_equity.png`, got {png_p!r}.")
                    else:
                        if not png_p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                            errors.append("Exported PNG does not have a valid PNG file header.")
                        actual_png_b = png_p.stat().st_size
                        if not isinstance(png_b, int) or isinstance(png_b, bool) or png_b != actual_png_b:
                            errors.append(f"`png_bytes` must be `int` equal to file size ({actual_png_b}), got {png_b!r}.")

                    if not isinstance(pdf_p, Path):
                        errors.append(f"`bundle['pdf']` must be a `pathlib.Path`, got `{type(pdf_p).__name__}`.")
                    elif pdf_p.name != "fig1_equity.pdf" or not pdf_p.exists():
                        errors.append(f"`bundle['pdf']` should point to existing `fig1_equity.pdf`, got {pdf_p!r}.")
                    else:
                        if not pdf_p.read_bytes().startswith(b"%PDF"):
                            errors.append("Exported PDF does not have a valid `%PDF` header.")
                        actual_pdf_b = pdf_p.stat().st_size
                        if not isinstance(pdf_b, int) or isinstance(pdf_b, bool) or pdf_b != actual_pdf_b:
                            errors.append(f"`pdf_bytes` must be `int` equal to file size ({actual_pdf_b}), got {pdf_b!r}.")

                    # Verify dpi parameter scales raster resolution
                    if isinstance(png_p, Path) and png_p.exists():
                        try:
                            low_bundle = fn(fig, low_dir, "fig1_equity", dpi=72)
                        except NotImplementedError:
                            raise
                        except Exception as exc:
                            errors.append(f"DPI scaling check (dpi=72) raised {type(exc).__name__}: {exc}")
                        else:
                            if isinstance(low_bundle, dict) and isinstance(low_bundle.get("png"), Path) and low_bundle["png"].exists():
                                w300, _ = _png_dimensions(png_p)
                                w72, _ = _png_dimensions(low_bundle["png"])
                                if w300 <= w72:
                                    errors.append(
                                        f"300-DPI PNG width ({w300}px) was not larger than 72-DPI width ({w72}px) — pass `dpi=dpi` to `fig.savefig(...)`!"
                                    )

            if errors:
                return CheckResult(
                    "ch19_exam_03",
                    False,
                    "failed",
                    "Exam Task 19.3 — Check Dual PNG + PDF Export Bundle",
                    "Create `out_dir`, save `<stem>.png` and `<stem>.pdf` at `dpi` with `bbox_inches='tight'`, and return `{'png': Path, 'pdf': Path, 'png_bytes': int, 'pdf_bytes': int}`.",
                    details=errors,
                    points_earned=0,
                    points_possible=30,
                )

            return CheckResult(
                "ch19_exam_03",
                True,
                "passed",
                "Exam Task 19.3 Passed (30/30) — Reproducible Figure Bundle Exporter Verified!",
                f"Created nested directory, saved 300-DPI PNG ({bundle['png_bytes']:,} bytes) + vector PDF ({bundle['pdf_bytes']:,} bytes), and returned full Path/byte metadata.",
                points_earned=30,
                points_possible=30,
                tip="Always export BOTH a 300-DPI PNG (for slides/previews) and a vector PDF (for LaTeX/Quarto camera-ready manuscripts) in one automated step!",
            )
    finally:
        if fig is not None:
            plt.close(fig)


register_exercise(
    ExerciseSpec(
        exercise_id="ch19_exam_03",
        chapter="ch19",
        title="Exam Task 19.3: Dual PNG + PDF Figure Bundle (`export_figure_bundle`)",
        book_section="Section 19.4 (Publication Export Pipelines)",
        points=30,
        is_exam=True,
        hints={
            1: "Ensure `out_path = Path(out_dir)` exists via `out_path.mkdir(parents=True, exist_ok=True)`, save `out_path / f'{stem}.png'` and `out_path / f'{stem}.pdf'` with `dpi=dpi, bbox_inches='tight'`, and return the 4-key dictionary.",
            2: "Blueprint:\n1. `out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)`\n2. `png_path = out / f'{stem}.png'`; `pdf_path = out / f'{stem}.pdf'`\n3. `fig.savefig(png_path, dpi=dpi, bbox_inches='tight')` and `fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')`\n4. Return `{'png': png_path, 'pdf': pdf_path, 'png_bytes': int(png_path.stat().st_size), 'pdf_bytes': int(pdf_path.stat().st_size)}`",
            3: "from pathlib import Path\n\ndef export_figure_bundle(fig, out_dir, stem, dpi=300):\n    out = Path(out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    png_path = out / f'{stem}.png'\n    pdf_path = out / f'{stem}.pdf'\n    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')\n    fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight')\n    return {\n        'png': png_path,\n        'pdf': pdf_path,\n        'png_bytes': int(png_path.stat().st_size),\n        'pdf_bytes': int(pdf_path.stat().st_size),\n    }",
        },
        verifier=_verify_exam_03,
    )
)
