"""sharukh_tutor: Interactive tutor & diagnostic verification engine.

Companion package for 'From First Line to First Paper: Mastering Python,
Statistical Data Science, and Applied Deep Learning' by Mezanur Rahman Sharukh.
"""

from .engine import (
    CheckResult,
    ExerciseSpec,
    TutorSession,
    activate,
    check,
    check_exam,
    get_session,
    hint,
    register_exercise,
    summary,
)
from .datasets import list_datasets, load_data

# Auto-load registered chapter check modules
from .checks import ch01 as _ch01  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "CheckResult",
    "ExerciseSpec",
    "TutorSession",
    "activate",
    "check",
    "check_exam",
    "get_session",
    "hint",
    "list_datasets",
    "load_data",
    "register_exercise",
    "summary",
    "__version__",
]
