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

# Auto-load registered chapter check modules (Parts I & II Complete: Chapters 1–11)
from .checks import ch01 as _ch01  # noqa: F401
from .checks import ch02 as _ch02  # noqa: F401
from .checks import ch03 as _ch03  # noqa: F401
from .checks import ch04 as _ch04  # noqa: F401
from .checks import ch05 as _ch05  # noqa: F401
from .checks import ch06 as _ch06  # noqa: F401
from .checks import ch07 as _ch07  # noqa: F401
from .checks import ch08 as _ch08  # noqa: F401
from .checks import ch09 as _ch09  # noqa: F401
from .checks import ch10 as _ch10  # noqa: F401
from .checks import ch11 as _ch11  # noqa: F401

__version__ = "0.3.0"

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
