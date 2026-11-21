"""Layered diagnostic verification & progressive hint engine for sharukh_tutor."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from typing import Any, Callable, Dict, List, Optional

try:
    from IPython.display import HTML, display as _ipython_display  # type: ignore

    def _in_notebook() -> bool:
        try:
            from IPython import get_ipython  # type: ignore

            shell = get_ipython()
            return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
        except Exception:
            return False

except Exception:
    HTML = None  # type: ignore
    _ipython_display = None

    def _in_notebook() -> bool:
        return False


class UnattemptedError(Exception):
    """Sentinel raised when a learner leaves the starter stub unmodified."""


@dataclass
class CheckResult:
    exercise_id: str
    passed: bool
    status: str  # "passed" | "unattempted" | "failed" | "error"
    title: str
    message: str
    details: List[str] = field(default_factory=list)
    points_earned: int = 0
    points_possible: int = 0
    tip: str = ""


@dataclass
class ExerciseSpec:
    exercise_id: str
    chapter: str
    title: str
    book_section: str
    hints: Dict[int, str]
    verifier: Callable[[Any], CheckResult]
    points: int = 10
    is_exam: bool = False


_REGISTRY: Dict[str, ExerciseSpec] = {}


def register_exercise(spec: ExerciseSpec) -> ExerciseSpec:
    """Register an exercise specification into the global tutor registry."""
    _REGISTRY[spec.exercise_id] = spec
    return spec


def _render_card(kind: str, title: str, body: str, details: Optional[List[str]] = None, footer: str = "") -> None:
    """Render a clean styled card in Jupyter/Colab or formatted text in terminal."""
    palette = {
        "pass": ("#1b5e20", "#e8f5e9", "#2e7d32", "PASS"),
        "fail": ("#b71c1c", "#ffebee", "#c62828", "CHECK"),
        "hint": ("#0d47a1", "#e3f2fd", "#1565c0", "HINT"),
        "info": ("#37474f", "#eceff1", "#455a64", "INFO"),
        "warn": ("#e65100", "#fff3e0", "#ef6c00", "TODO"),
    }
    border, bg, heading_color, badge = palette.get(kind, palette["info"])
    details = details or []

    if _in_notebook() and HTML is not None and _ipython_display is not None:
        items_html = ""
        if details:
            lis = "".join(f"<li style='margin:4px 0;'>{html.escape(d)}</li>" for d in details)
            items_html = f"<ul style='margin:8px 0 6px 18px;padding:0;'>{lis}</ul>"
        footer_html = (
            f"<div style='margin-top:8px;padding-top:6px;border-top:1px solid rgba(0,0,0,0.1);"
            f"font-size:0.9em;color:#37474f;'><strong>Researcher Tip:</strong> {html.escape(footer)}</div>"
            if footer
            else ""
        )
        card = (
            f"<div style='border-left:5px solid {border};background:{bg};padding:12px 16px;"
            f"border-radius:6px;margin:8px 0;font-family:system-ui,-apple-system,sans-serif;color:#1a1a1a;'>"
            f"<div style='font-weight:700;color:{heading_color};font-size:1.02em;margin-bottom:4px;'>"
            f"[{badge}] {html.escape(title)}</div>"
            f"<div style='line-height:1.5;white-space:pre-line;'>{html.escape(body)}</div>"
            f"{items_html}{footer_html}</div>"
        )
        _ipython_display(HTML(card))
    else:
        print(f"\n[{badge}] {title}")
        print("-" * max(40, len(title) + len(badge) + 3))
        print(body)
        for item in details:
            print(f"  - {item}")
        if footer:
            print(f"  Tip: {footer}")


class TutorSession:
    """Tracks learner progress across a single notebook session."""

    def __init__(self, chapter: str = "", notebook: str = "") -> None:
        self.chapter = chapter
        self.notebook = notebook
        self.results: Dict[str, CheckResult] = {}
        self.hints_used: Dict[str, int] = {}

    def record(self, result: CheckResult) -> None:
        self.results[result.exercise_id] = result

    def record_hint(self, exercise_id: str, level: int) -> None:
        prev = self.hints_used.get(exercise_id, 0)
        self.hints_used[exercise_id] = max(prev, level)


_ACTIVE_SESSION = TutorSession()


def get_session() -> TutorSession:
    return _ACTIVE_SESSION


def activate(chapter: str, notebook: str = "todos", quiet: bool = False) -> TutorSession:
    """Initialize a fresh interactive tutor session for a chapter notebook."""
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = TutorSession(chapter=chapter, notebook=notebook)
    available = [k for k, v in _REGISTRY.items() if v.chapter == chapter]
    if not quiet:
        _render_card(
            "info",
            f"Interactive Tutor Activated ({chapter.upper()} — {notebook})",
            f"Loaded {len(available)} verification check(s) for {chapter.upper()}.\n"
            f"Use tutor.hint('<exercise_id>', level=1..3) whenever you want a progressive nudge, "
            f"and tutor.check('<exercise_id>', your_answer) to verify your work.",
            footer="Remember our discipline: read the Problem Blueprint before writing code!",
        )
    return _ACTIVE_SESSION


def hint(exercise_id: str, level: int = 1) -> str:
    """Display a 3-tier progressive hint (1=Concept, 2=Blueprint, 3=Skeleton)."""
    if exercise_id not in _REGISTRY:
        msg = f"Unknown exercise ID '{exercise_id}'. Double-check the spelling from the exercise prompt."
        _render_card("warn", "Exercise Not Found", msg)
        return msg

    spec = _REGISTRY[exercise_id]
    clamped = max(1, min(3, int(level)))
    text = spec.hints.get(clamped, spec.hints.get(max(spec.hints.keys()), "No hint available."))
    _ACTIVE_SESSION.record_hint(exercise_id, clamped)

    level_names = {
        1: "Level 1 — Conceptual Nudge",
        2: "Level 2 — Step-by-Step Algorithm Blueprint",
        3: "Level 3 — Structural Code Skeleton",
    }
    next_tip = (
        f"Still stuck? Try tutor.hint('{exercise_id}', level={clamped + 1}) or revisit Book {spec.book_section}."
        if clamped < 3
        else f"Review Book {spec.book_section}, or compare with the paired _SOLUTION.ipynb notebook after trying!"
    )
    _render_card(
        "hint",
        f"{spec.title} ({level_names[clamped]})",
        text,
        footer=next_tip,
    )
    return text


def check(exercise_id: str, submission: Any = None, **kwargs: Any) -> CheckResult:
    """Run layered diagnostic verification on a learner's variable, value, or function."""
    if exercise_id not in _REGISTRY:
        res = CheckResult(
            exercise_id=exercise_id,
            passed=False,
            status="error",
            title="Unknown Exercise ID",
            message=f"Could not find checks for '{exercise_id}'.",
        )
        _render_card("fail", res.title, res.message)
        return res

    spec = _REGISTRY[exercise_id]
    try:
        if submission is None and not kwargs:
            res = CheckResult(
                exercise_id=exercise_id,
                passed=False,
                status="unattempted",
                title=f"{spec.title} — Not Yet Completed",
                message="Your variable or function currently evaluates to None (nothing returned or assigned yet).",
                details=[
                    "Check if you replaced '# YOUR CODE HERE' in the cell above.",
                    "If this exercise asks for a function, make sure you used 'return' instead of only 'print()'.",
                ],
                points_earned=0,
                points_possible=spec.points,
                tip=f"Try tutor.hint('{exercise_id}', level=1) for a gentle starting nudge.",
            )
        else:
            payload = kwargs if kwargs and submission is None else submission
            res = spec.verifier(payload)
            if res.points_possible == 0:
                res.points_possible = spec.points
            if res.passed and res.points_earned == 0:
                res.points_earned = spec.points
    except NotImplementedError:
        res = CheckResult(
            exercise_id=exercise_id,
            passed=False,
            status="unattempted",
            title=f"{spec.title} — Ready for Your Solution",
            message="You haven't replaced the starter 'NotImplementedError' line yet.",
            details=["Delete the 'raise NotImplementedError(...)' line and write your code under '# YOUR CODE HERE'."],
            points_earned=0,
            points_possible=spec.points,
            tip=f"Run tutor.hint('{exercise_id}', level=1) if you want a conceptual clue first.",
        )
    except Exception as exc:
        res = CheckResult(
            exercise_id=exercise_id,
            passed=False,
            status="error",
            title=f"{spec.title} — Runtime Error During Testing",
            message=f"Your code raised {type(exc).__name__}: {exc}",
            details=[
                "Read the error type calmly — every error message is a diagnostic clue.",
                "Check variable names, parentheses, quotes, and data types (int/float vs. str).",
            ],
            points_earned=0,
            points_possible=spec.points,
            tip=f"Revisit Book {spec.book_section} or run tutor.hint('{exercise_id}', level=2).",
        )

    _ACTIVE_SESSION.record(res)
    kind = "pass" if res.passed else ("warn" if res.status == "unattempted" else "fail")
    _render_card(kind, res.title, res.message, res.details, res.tip)
    return res


def check_exam(chapter: str, submissions: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a chapter coding exam and render a 100-point diagnostic scoreboard."""
    exam_specs = [s for s in _REGISTRY.values() if s.chapter == chapter and s.is_exam]
    total_possible = sum(s.points for s in exam_specs)
    total_earned = 0
    lines: List[str] = []

    for spec in exam_specs:
        sub = submissions.get(spec.exercise_id)
        res = check(spec.exercise_id, sub)
        total_earned += res.points_earned
        mark = "PASS" if res.passed else "NEEDS WORK"
        lines.append(f"[{mark}] {spec.title}: {res.points_earned}/{spec.points} pts")

    pct = round((total_earned / total_possible) * 100, 1) if total_possible else 0.0
    if pct >= 90:
        verdict = "Mastery Achieved! You are ready to advance to the next chapter."
        kind = "pass"
    elif pct >= 70:
        verdict = "Strong Progress! Review the missed edge cases in the Debugging Post-Mortem before moving on."
        kind = "warn"
    else:
        verdict = "Keep Building! Revisit the Problem-Structuring Blueprints and re-run the Guided To-Do labs."
        kind = "fail"

    _render_card(
        kind,
        f"{chapter.upper()} Coding Exam Scoreboard — {total_earned}/{total_possible} ({pct}%)",
        verdict,
        details=lines,
        footer="Compare your approach line-by-line with the annotated _SOLUTION.ipynb after achieving a pass.",
    )
    return {
        "chapter": chapter,
        "earned": total_earned,
        "possible": total_possible,
        "percent": pct,
        "passed_all": total_earned == total_possible and total_possible > 0,
    }


def summary() -> Dict[str, Any]:
    """Display a live session progress report across checked exercises."""
    sess = _ACTIVE_SESSION
    total = len(sess.results)
    passed = sum(1 for r in sess.results.values() if r.passed)
    earned = sum(r.points_earned for r in sess.results.values())
    possible = sum(r.points_possible for r in sess.results.values())

    details = [
        f"{'PASS' if r.passed else 'OPEN'} | {ex_id}: {r.points_earned}/{r.points_possible} pts "
        f"(hints used: {sess.hints_used.get(ex_id, 0)})"
        for ex_id, r in sess.results.items()
    ]
    if not details:
        details = ["No checks have been run in this session yet."]

    _render_card(
        "pass" if (total > 0 and passed == total) else "info",
        f"Session Progress Summary ({passed}/{total} checks passed)",
        f"Score: {earned}/{possible} points across {total} evaluated exercise(s).",
        details=details,
        footer="Before closing Colab, run Runtime -> Restart session and run all to confirm reproducibility.",
    )
    return {
        "passed": passed,
        "total": total,
        "earned": earned,
        "possible": possible,
    }
