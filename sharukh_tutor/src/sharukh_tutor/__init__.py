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

# Auto-load registered chapter check modules (Complete Edition: Parts I–VII, Chapters 1–36)
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
from .checks import ch12 as _ch12  # noqa: F401
from .checks import ch13 as _ch13  # noqa: F401
from .checks import ch14 as _ch14  # noqa: F401
from .checks import ch15 as _ch15  # noqa: F401
from .checks import ch16 as _ch16  # noqa: F401
from .checks import ch17 as _ch17  # noqa: F401
from .checks import ch18 as _ch18  # noqa: F401
from .checks import ch19 as _ch19  # noqa: F401
from .checks import ch20 as _ch20  # noqa: F401
from .checks import ch21 as _ch21  # noqa: F401
from .checks import ch22 as _ch22  # noqa: F401
from .checks import ch23 as _ch23  # noqa: F401
from .checks import ch24 as _ch24  # noqa: F401
from .checks import ch25 as _ch25  # noqa: F401
from .checks import ch26 as _ch26  # noqa: F401
from .checks import ch27 as _ch27  # noqa: F401
from .checks import ch28 as _ch28  # noqa: F401
from .checks import ch29 as _ch29  # noqa: F401
from .checks import ch30 as _ch30  # noqa: F401
from .checks import ch31 as _ch31  # noqa: F401
from .checks import ch32 as _ch32  # noqa: F401
from .checks import ch33 as _ch33  # noqa: F401
from .checks import ch34 as _ch34  # noqa: F401
from .checks import ch35 as _ch35  # noqa: F401
from .checks import ch36 as _ch36  # noqa: F401

__version__ = "1.0.1"

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
