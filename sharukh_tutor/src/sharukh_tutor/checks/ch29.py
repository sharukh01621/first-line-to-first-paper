"""Chapter 29 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch29_todo_01..06) and the Chapter 29 Coding Exam
(ch29_exam_01..03) for Training Dynamics, Optimizers, Regularization,
BatchNorm, and Early Stopping.
Every verifier lazy-imports PyTorch/NumPy/Pandas/Scikit-Learn inside the
check function, validates return structures, rounding, deep-copy hygiene,
and training dynamics, while letting NotImplementedError propagate cleanly
for starter stubs.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_np_or_py_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, (bool, np.bool_)
    )


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
                f"Pass the symbol itself: tutor.check('{ex_id}', {expected_name})",
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
            "Remember: print() only displays text; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Training utilities should return structured dicts, lists, models, or DataFrames.",
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
# GUIDED TO-DO 29.1: init_linear_weights(layer, method="kaiming")
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch29_todo_01", "To-Do 29.1", fn, "init_linear_weights", 10)
    if bad is not None:
        return bad

    errors = []

    # Case 1: Kaiming normal (default)
    torch.manual_seed(42)
    layer_k = nn.Linear(256, 256)
    with torch.no_grad():
        layer_k.bias.fill_(5.0)
    try:
        ret_k = fn(layer_k, method="kaiming")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_todo_01",
            False,
            "failed",
            "To-Do 29.1 — Runtime Error on method='kaiming'",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if ret_k is None:
        return _none_return_trap("ch29_todo_01", "To-Do 29.1", "init_linear_weights(layer)", 10)
    if ret_k is not layer_k:
        errors.append("[kaiming] Must return the initialized `layer` object itself (`return layer`).")
    k_std = float(layer_k.weight.detach().std().item())
    k_theory = math.sqrt(2.0 / 256.0)
    if not math.isclose(k_std, k_theory, abs_tol=0.015):
        errors.append(
            f"[kaiming] Weight std={k_std:.4f}, expected ~{k_theory:.4f} "
            "(use `nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')`)."
        )
    if float(layer_k.bias.detach().abs().max().item()) > 1e-7:
        errors.append("[kaiming] `layer.bias` was not zeroed with `nn.init.zeros_(layer.bias)`.")

    # Case 2: Xavier uniform
    torch.manual_seed(42)
    layer_x = nn.Linear(256, 256)
    with torch.no_grad():
        layer_x.bias.fill_(-3.0)
    try:
        ret_x = fn(layer_x, method="xavier")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_todo_01",
            False,
            "failed",
            "To-Do 29.1 — Runtime Error on method='xavier'",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if ret_x is not layer_x:
        errors.append("[xavier] Must return the initialized `layer` object itself.")
    x_bound = math.sqrt(6.0 / (256.0 + 256.0))
    x_std = float(layer_x.weight.detach().std().item())
    x_theory_std = math.sqrt(2.0 / (256.0 + 256.0))
    x_max = float(layer_x.weight.detach().abs().max().item())
    if not math.isclose(x_std, x_theory_std, abs_tol=0.015) or x_max > x_bound + 1e-4:
        errors.append(
            f"[xavier] Weight std={x_std:.4f} (expected ~{x_theory_std:.4f}), "
            f"max|w|={x_max:.4f} (expected <= {x_bound:.4f}); use `nn.init.xavier_uniform_(layer.weight)`."
        )
    if float(layer_x.bias.detach().abs().max().item()) > 1e-7:
        errors.append("[xavier] `layer.bias` was not zeroed.")

    # Case 3: bias=False guard
    layer_nb = nn.Linear(32, 16, bias=False)
    try:
        ret_nb = fn(layer_nb)
        if ret_nb is not layer_nb:
            errors.append("[bias=False] Must return `layer` even when `layer.bias is None`.")
    except Exception as exc:
        errors.append(
            f"[bias=False] Crashed when `layer.bias is None` ({type(exc).__name__}: {exc}). "
            "Guard with `if layer.bias is not None: nn.init.zeros_(layer.bias)`."
        )

    # Case 4: Unknown method must raise ValueError
    layer_bad = nn.Linear(8, 4)
    raised_val_err = False
    try:
        fn(layer_bad, method="unknown_scheme")
    except ValueError:
        raised_val_err = True
    except Exception as exc:
        errors.append(
            f"[unknown method] Expected ValueError, got {type(exc).__name__}: {exc}."
        )
    if not raised_val_err:
        errors.append("[unknown method] Must raise `ValueError` when `method` is not 'kaiming' or 'xavier'.")

    if errors:
        return CheckResult(
            "ch29_todo_01",
            False,
            "failed",
            "To-Do 29.1 — Check Weight Initialization, Bias Guard & Validation",
            "Apply Kaiming normal (`nonlinearity='relu'`) or Xavier uniform, zero bias when present, raise `ValueError` for unknown methods, and return `layer`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch29_todo_01",
        True,
        "passed",
        "To-Do 29.1 Passed — Kaiming & Xavier Initialization Verified!",
        "Verified Kaiming ReLU variance, Xavier uniform bounds, `bias=False` safety, and `ValueError` on unknown methods.",
        tip="Use Kaiming (He) initialization with ReLU layers (factor 2 compensates for zeroed negative activations) and Xavier (Glorot) with tanh/sigmoid.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_todo_01",
        chapter="ch29",
        title="To-Do 29.1: Kaiming & Xavier Linear Initializer (init_linear_weights)",
        book_section="Section 29.3.1 (Weight Initialization — Kaiming vs. Xavier)",
        points=10,
        hints={
            1: "Branch on `method == 'kaiming'` vs `method == 'xavier'`, and raise `ValueError` for any other string.",
            2: "Call `nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')` or `nn.init.xavier_uniform_(layer.weight)`, then check `if layer.bias is not None: nn.init.zeros_(layer.bias)` and `return layer`.",
            3: "def init_linear_weights(layer, method='kaiming'):\n    if method == 'kaiming':\n        nn.init.kaiming_normal_(layer.weight, nonlinearity='___')\n    elif method == 'xavier':\n        nn.init.xavier_uniform_(layer.weight)\n    else:\n        raise ValueError(f'Unknown method: {method}')\n    if layer.bias is not None:\n        nn.init.zeros_(layer.bias)\n    return layer",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 29.2: RegularizedClinicalMLP (nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_02(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        return CheckResult(
            "ch29_todo_02",
            False,
            "failed",
            "To-Do 29.2 — Pass the `RegularizedClinicalMLP` Class (Inheriting `nn.Module`)",
            f"Expected a subclass of `torch.nn.Module`, got {type(cls).__name__} ({cls!r}).",
            details=[
                "Define `class RegularizedClinicalMLP(nn.Module):` and pass the class itself: `tutor.check('ch29_todo_02', RegularizedClinicalMLP)`."
            ],
            points_earned=0,
            points_possible=10,
        )

    errors = []

    try:
        model_def = cls(in_dim=6)
        model_cus = cls(in_dim=4, hidden_dim=24, dropout_p=0.5)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_todo_02",
            False,
            "failed",
            "To-Do 29.2 — Error Instantiating `RegularizedClinicalMLP`",
            f"Raised {type(exc).__name__}: {exc}",
            details=["Check `__init__(self, in_dim, hidden_dim=32, dropout_p=0.3)` and call `super().__init__()`."],
            points_earned=0,
            points_possible=10,
        )

    for model, in_d, hid_d, drop_p, desc in [
        (model_def, 6, 32, 0.3, "default (in_dim=6, hidden_dim=32, dropout_p=0.3)"),
        (model_cus, 4, 24, 0.5, "custom (in_dim=4, hidden_dim=24, dropout_p=0.5)"),
    ]:
        mods = [m for m in model.modules() if not isinstance(m, (cls, nn.Sequential))]
        linears = [m for m in mods if isinstance(m, nn.Linear)]
        bns = [m for m in mods if isinstance(m, nn.BatchNorm1d)]
        drops = [m for m in mods if isinstance(m, nn.Dropout)]
        sigmoids = [m for m in mods if isinstance(m, nn.Sigmoid)]

        if sigmoids:
            errors.append(f"[{desc}] Found `nn.Sigmoid`! Return raw logits so `BCEWithLogitsLoss` stays numerically stable.")
        if len(linears) != 2:
            errors.append(f"[{desc}] Expected 2 `nn.Linear` layers, found {len(linears)}.")
        else:
            if (linears[0].in_features, linears[0].out_features) != (in_d, hid_d):
                errors.append(
                    f"[{desc}] First Linear should be ({in_d}, {hid_d}), "
                    f"got ({linears[0].in_features}, {linears[0].out_features})."
                )
            if (linears[1].in_features, linears[1].out_features) != (hid_d, 1):
                errors.append(
                    f"[{desc}] Output Linear should be ({hid_d}, 1), "
                    f"got ({linears[1].in_features}, {linears[1].out_features})."
                )
        if len(bns) != 1 or bns[0].num_features != hid_d:
            errors.append(f"[{desc}] Expected one `nn.BatchNorm1d({hid_d})`.")
        if len(drops) != 1 or not math.isclose(float(drops[0].p), drop_p, abs_tol=1e-6):
            errors.append(f"[{desc}] Expected one `nn.Dropout(p={drop_p})`.")

        torch.manual_seed(42)
        x = torch.randn(16, in_d)
        try:
            model.train()
            out_t1 = model(x)
            out_t2 = model(x)
            model.eval()
            with torch.no_grad():
                out_e1 = model(x)
                out_e2 = model(x)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"[{desc}] Forward pass raised {type(exc).__name__}: {exc}")
            continue

        if out_t1 is None or out_e1 is None:
            return _none_return_trap("ch29_todo_02", "To-Do 29.2", "model(x)", 10)
        if tuple(out_e1.shape) != (16, 1):
            errors.append(f"[{desc}] Output shape should be (16, 1), got {tuple(out_e1.shape)}.")
        if not torch.isfinite(out_e1).all():
            errors.append(f"[{desc}] Forward pass produced non-finite values.")
        if torch.allclose(out_t1, out_t2):
            errors.append(f"[{desc}] Two train() passes were identical — make sure Dropout is applied in `forward()`!")
        if not torch.allclose(out_e1, out_e2):
            errors.append(f"[{desc}] Two eval() passes differed — eval() must disable Dropout.")

        # Verify raw logits (not squashed into [0, 1]) by scaling output layer
        if len(linears) == 2:
            with torch.no_grad():
                linears[1].weight.mul_(20.0)
                logits_big = model(x)
            if float(logits_big.min().item()) >= 0.0 and float(logits_big.max().item()) <= 1.0:
                errors.append(
                    f"[{desc}] Output stays trapped in [0, 1] even with scaled weights — do NOT apply sigmoid inside `forward()`!"
                )

    if errors:
        return CheckResult(
            "ch29_todo_02",
            False,
            "failed",
            "To-Do 29.2 — Check `RegularizedClinicalMLP` Layers & Raw Logits Output",
            "Stack `Linear(in_dim, hidden_dim) -> BatchNorm1d(hidden_dim) -> ReLU -> Dropout(dropout_p) -> Linear(hidden_dim, 1)` returning raw `(batch, 1)` logits.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch29_todo_02",
        True,
        "passed",
        "To-Do 29.2 Passed — `RegularizedClinicalMLP` Verified!",
        "Confirmed `Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear` topology, raw `(batch, 1)` logits, stochastic train() mode, and deterministic eval() mode.",
        tip="Returning raw logits instead of sigmoid probabilities lets `nn.BCEWithLogitsLoss()` use the log-sum-exp trick for numerical stability.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_todo_02",
        chapter="ch29",
        title="To-Do 29.2: Regularized Clinical MLP (RegularizedClinicalMLP)",
        book_section="Section 29.3.4 (BatchNorm1d & Dropout Architecture)",
        points=10,
        hints={
            1: "Subclass `nn.Module`, call `super().__init__()`, and build `self.net = nn.Sequential(...)` with 5 layers ending in `nn.Linear(hidden_dim, 1)` (no Sigmoid!).",
            2: "Exact layer order: `nn.Linear(in_dim, hidden_dim)`, `nn.BatchNorm1d(hidden_dim)`, `nn.ReLU()`, `nn.Dropout(dropout_p)`, `nn.Linear(hidden_dim, 1)`.",
            3: "class RegularizedClinicalMLP(nn.Module):\n    def __init__(self, in_dim, hidden_dim=32, dropout_p=0.3):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(in_dim, hidden_dim),\n            nn.BatchNorm1d(hidden_dim),\n            nn.ReLU(),\n            nn.Dropout(___),\n            nn.Linear(hidden_dim, 1),\n        )\n    def forward(self, x):\n        return self.net(x)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 29.3: verify_dropout_train_vs_eval(model, x_batch)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch29_todo_03", "To-Do 29.3", fn, "verify_dropout_train_vs_eval", 10)
    if bad is not None:
        return bad

    torch.manual_seed(42)
    x_batch = torch.randn(16, 6)
    m_drop = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1))
    m_plain = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))

    cases = [
        (m_drop, True, True, "model WITH Dropout(0.5)"),
        (m_plain, False, True, "deterministic model WITHOUT Dropout"),
    ]
    expected_keys = ["train_stochastic", "eval_deterministic"]
    errors = []

    for model, exp_train_stoch, exp_eval_det, desc in cases:
        try:
            got = fn(model, x_batch)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch29_todo_03",
                False,
                "failed",
                "To-Do 29.3 — Runtime Error in `verify_dropout_train_vs_eval`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch29_todo_03", "To-Do 29.3", "verify_dropout_train_vs_eval(model, x_batch)", 10
            )
        key_err = _check_dict_keys("ch29_todo_03", "To-Do 29.3", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if not isinstance(got["train_stochastic"], bool) or not isinstance(got["eval_deterministic"], bool):
            errors.append(f"[{desc}] Both values must be native Python `bool`, got {got!r}.")
            continue

        if got["train_stochastic"] is not exp_train_stoch:
            errors.append(
                f"[{desc}] `train_stochastic`: expected {exp_train_stoch}, got {got['train_stochastic']} "
                "(Did you actually call `model.train()` and compare two passes with `not torch.allclose(t1, t2)`?)."
            )
        if got["eval_deterministic"] is not exp_eval_det:
            errors.append(
                f"[{desc}] `eval_deterministic`: expected {exp_eval_det}, got {got['eval_deterministic']} "
                "(Did you call `model.eval()` before the two evaluation passes?)."
            )

    if errors:
        return CheckResult(
            "ch29_todo_03",
            False,
            "failed",
            "To-Do 29.3 — Check `model.train()` vs. `model.eval()` Verification",
            "Under `torch.no_grad()`, run 2 passes in `model.train()` and 2 passes in `model.eval()`, returning `{'train_stochastic': bool, 'eval_deterministic': bool}`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch29_todo_03",
        True,
        "passed",
        "To-Do 29.3 Passed — Dropout Train vs. Eval Behavior Verified!",
        "Confirmed stochastic predictions under `model.train()` (when Dropout is present) and identical deterministic predictions under `model.eval()`.",
        tip="Forgetting `model.eval()` before validation or deployment leaves Dropout randomly masking neurons during clinical inference!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_todo_03",
        chapter="ch29",
        title="To-Do 29.3: Dropout Train vs. Eval Mode Audit (verify_dropout_train_vs_eval)",
        book_section="Section 29.3.3 (The Dropout train() vs. eval() Trap)",
        points=10,
        hints={
            1: "Wrap all 4 forward passes inside `with torch.no_grad():` so no unnecessary autograd graph is built.",
            2: "Call `model.train()`, compute `t1 = model(x_batch)` and `t2 = model(x_batch)`. Then call `model.eval()`, compute `e1 = model(x_batch)` and `e2 = model(x_batch)`.",
            3: "def verify_dropout_train_vs_eval(model, x_batch):\n    with torch.no_grad():\n        model.train()\n        t1, t2 = model(x_batch), model(x_batch)\n        model.eval()\n        e1, e2 = model(x_batch), model(x_batch)\n    return {\n        'train_stochastic': bool(not torch.allclose(t1, t2)),\n        'eval_deterministic': bool(torch.allclose(e1, e2)),\n    }",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 29.4: step_with_grad_clipping(model, loss, optimizer, max_norm=1.0)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch29_todo_04", "To-Do 29.4", fn, "step_with_grad_clipping", 10)
    if bad is not None:
        return bad

    expected_keys = ["grad_norm_before", "grad_norm_after"]
    errors = []

    cases = [
        (50.0, 1.0, True, "exploding gradient clipped to max_norm=1.0"),
        (0.05, 10.0, False, "small gradient below max_norm=10.0"),
    ]

    for scale, max_norm, expect_clipped, desc in cases:
        torch.manual_seed(42)
        m_ref = nn.Linear(4, 1)
        m_stu = nn.Linear(4, 1)
        m_stu.load_state_dict(copy.deepcopy(m_ref.state_dict()))

        x = torch.randn(12, 4)
        y = torch.randn(12, 1)

        # Reference calculation
        opt_ref = torch.optim.SGD(m_ref.parameters(), lr=0.1)
        loss_ref = ((m_ref(x) - y) ** 2).mean() * scale
        opt_ref.zero_grad()
        loss_ref.backward()
        exp_before = float(nn.utils.clip_grad_norm_(m_ref.parameters(), max_norm))
        exp_after = math.sqrt(
            sum(float(p.grad.detach().pow(2).sum().item()) for p in m_ref.parameters() if p.grad is not None)
        )
        opt_ref.step()

        # Student call — inject huge stale grad first to verify optimizer.zero_grad()
        opt_stu = torch.optim.SGD(m_stu.parameters(), lr=0.1)
        for p in m_stu.parameters():
            p.grad = torch.full_like(p, 500.0)
        w_before = m_stu.weight.detach().clone()
        loss_stu = ((m_stu(x) - y) ** 2).mean() * scale

        try:
            got = fn(m_stu, loss_stu, opt_stu, max_norm=max_norm)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch29_todo_04",
                False,
                "failed",
                "To-Do 29.4 — Runtime Error in `step_with_grad_clipping`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch29_todo_04", "To-Do 29.4", "step_with_grad_clipping(model, loss, optimizer)", 10
            )
        key_err = _check_dict_keys("ch29_todo_04", "To-Do 29.4", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        gb, ga = got["grad_norm_before"], got["grad_norm_after"]
        if not (_is_np_or_py_number(gb) and _is_np_or_py_number(ga)):
            errors.append(f"[{desc}] Expected numeric norms, got {got!r}.")
            continue

        if not math.isclose(float(gb), round(exp_before, 4), abs_tol=1e-3):
            extra = " (Did you forget `optimizer.zero_grad()` before `loss.backward()`?)" if float(gb) > exp_before + 100 else ""
            errors.append(f"[{desc}] `grad_norm_before`: expected {round(exp_before, 4)}, got {gb!r}{extra}")
        if float(ga) > max_norm + 1e-3:
            errors.append(f"[{desc}] `grad_norm_after` ({ga}) exceeded `max_norm` ({max_norm})!")
        if not math.isclose(float(ga), round(exp_after, 4), abs_tol=1e-3):
            errors.append(f"[{desc}] `grad_norm_after`: expected {round(exp_after, 4)}, got {ga!r}")
        if expect_clipped and not (float(gb) > float(ga)):
            errors.append(f"[{desc}] Expected `grad_norm_before` > `grad_norm_after` when gradient exceeds `max_norm`.")
        if torch.allclose(w_before, m_stu.weight.detach()):
            errors.append(f"[{desc}] Model weights did not change — did you call `optimizer.step()`?")

    if errors:
        return CheckResult(
            "ch29_todo_04",
            False,
            "failed",
            "To-Do 29.4 — Check Gradient Clipping & Optimizer Step Sequence",
            "Run `optimizer.zero_grad()`, `loss.backward()`, `nn.utils.clip_grad_norm_(model.parameters(), max_norm)`, compute post-clip L2 norm, and call `optimizer.step()`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch29_todo_04",
        True,
        "passed",
        "To-Do 29.4 Passed — Gradient Norm Clipping Step Verified!",
        "Cleared stale gradients, clipped exploding global L2 norm to `<= max_norm`, left benign gradients untouched, and stepped optimizer weights.",
        tip="Global norm clipping rescales all parameter gradients by the same factor, preserving the update direction while preventing NaN explosions!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_todo_04",
        chapter="ch29",
        title="To-Do 29.4: Gradient Norm Clipping Step (step_with_grad_clipping)",
        book_section="Section 29.3.5 (Gradient Clipping & Exploding Gradients)",
        points=10,
        hints={
            1: "Follow the strict 5-step order: `optimizer.zero_grad()` -> `loss.backward()` -> `clip_grad_norm_` -> measure post-clip norm -> `optimizer.step()`.",
            2: "`nn.utils.clip_grad_norm_(model.parameters(), max_norm)` returns the pre-clip total L2 norm as a tensor. Afterward, compute `math.sqrt(sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None))`.",
            3: "def step_with_grad_clipping(model, loss, optimizer, max_norm=1.0):\n    optimizer.zero_grad()\n    loss.backward()\n    before = float(nn.utils.clip_grad_norm_(model.parameters(), max_norm))\n    after = math.sqrt(sum(float(p.grad.detach().pow(2).sum().item()) for p in model.parameters() if p.grad is not None))\n    optimizer.step()\n    return {'grad_norm_before': round(before, 4), 'grad_norm_after': round(after, 4)}",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 29.5: EarlyStoppingTracker
# ---------------------------------------------------------------------------
def _verify_todo_05(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    if not isinstance(cls, type):
        return CheckResult(
            "ch29_todo_05",
            False,
            "failed",
            "To-Do 29.5 — Pass the `EarlyStoppingTracker` Class Without Parentheses",
            f"Expected a class, got {type(cls).__name__} ({cls!r}).",
            points_earned=0,
            points_possible=10,
        )

    errors = []
    try:
        tracker = cls(patience=3, min_delta=0.01)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_todo_05",
            False,
            "failed",
            "To-Do 29.5 — Error Instantiating `EarlyStoppingTracker`",
            f"Raised {type(exc).__name__}: {exc}",
            details=["Check `__init__(self, patience=5, min_delta=0.0)`."],
            points_earned=0,
            points_possible=10,
        )

    for attr, exp_v in [
        ("best_loss", float("inf")),
        ("counter", 0),
        ("should_stop", False),
        ("best_state_dict", None),
    ]:
        if not hasattr(tracker, attr):
            errors.append(f"Missing required attribute `self.{attr}` after `__init__`.")
        elif getattr(tracker, attr) != exp_v:
            errors.append(f"Initial `self.{attr}` expected {exp_v!r}, got {getattr(tracker, attr)!r}.")

    if errors:
        return CheckResult(
            "ch29_todo_05",
            False,
            "failed",
            "To-Do 29.5 — Check `EarlyStoppingTracker.__init__` Attributes",
            "Initialize `best_loss=float('inf')`, `counter=0`, `should_stop=False`, and `best_state_dict=None`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    torch.manual_seed(42)
    model = nn.Linear(3, 1)
    w_snap = model.weight.detach().clone()

    try:
        s1 = tracker.step(0.80, model)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_todo_05",
            False,
            "failed",
            "To-Do 29.5 — Runtime Error in `EarlyStoppingTracker.step`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if s1 is not False:
        errors.append(f"Step 1 (loss=0.80 improvement) should return False, got {s1!r}.")
    if not math.isclose(float(tracker.best_loss), 0.80, abs_tol=1e-6) or tracker.counter != 0:
        errors.append(f"After step 1: expected best_loss=0.80 & counter=0, got {tracker.best_loss}, {tracker.counter}.")
    if tracker.best_state_dict is None:
        errors.append("`best_state_dict` was not saved on improvement!")
    else:
        # Mutate live model weights in place to catch shallow state_dict() aliasing!
        with torch.no_grad():
            model.weight.fill_(99.0)
        if not torch.allclose(tracker.best_state_dict["weight"], w_snap):
            errors.append(
                "`best_state_dict` mutated when the live model changed! "
                "Use `copy.deepcopy(model.state_dict())` — `model.state_dict()` alone is a shallow reference!"
            )

    # Tiny improvement smaller than min_delta=0.01 must NOT reset counter
    s2 = tracker.step(0.795, model)
    if s2 is not False or tracker.counter != 1 or not math.isclose(float(tracker.best_loss), 0.80, abs_tol=1e-6):
        errors.append(
            f"Step 2 (loss=0.795 vs best=0.80 with min_delta=0.01) should NOT count as improvement: "
            f"expected counter=1, best_loss=0.80, got counter={tracker.counter}, best_loss={tracker.best_loss}."
        )

    # Genuine improvement resets counter to 0
    s3 = tracker.step(0.75, model)
    if s3 is not False or tracker.counter != 0 or not math.isclose(float(tracker.best_loss), 0.75, abs_tol=1e-6):
        errors.append(
            f"Step 3 (loss=0.75 genuine improvement) should reset counter=0 and best_loss=0.75, "
            f"got counter={tracker.counter}, best_loss={tracker.best_loss}."
        )

    # Three consecutive non-improvements trigger should_stop=True on patience=3
    s4 = tracker.step(0.76, model)
    s5 = tracker.step(0.77, model)
    s6 = tracker.step(0.78, model)
    if s4 is not False or s5 is not False or s6 is not True or tracker.should_stop is not True:
        errors.append(
            f"With patience=3, three non-improving steps should return (False, False, True), "
            f"got ({s4!r}, {s5!r}, {s6!r}) with should_stop={tracker.should_stop!r}."
        )

    if errors:
        return CheckResult(
            "ch29_todo_05",
            False,
            "failed",
            "To-Do 29.5 — Check `EarlyStoppingTracker` Patience, `min_delta` & Deep Copy",
            "Improve only when `val_loss < self.best_loss - self.min_delta` (deep-copying `model.state_dict()` and resetting `counter = 0`); otherwise increment `counter` and set `should_stop = counter >= patience`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch29_todo_05",
        True,
        "passed",
        "To-Do 29.5 Passed — `EarlyStoppingTracker` Verified!",
        "Passed initial state, `copy.deepcopy(model.state_dict())` aliasing guard, `min_delta` threshold, counter reset, and exact `patience` stop trigger.",
        tip="Always deep-copy `model.state_dict()` when checkpointing in memory — PyTorch's `state_dict()` returns live tensor views that overwrite on the next optimizer step!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_todo_05",
        chapter="ch29",
        title="To-Do 29.5: Early Stopping & Best-Weight Checkpointer (EarlyStoppingTracker)",
        book_section="Section 29.3.7 (Early Stopping & Checkpoint Restoration)",
        points=10,
        hints={
            1: "Store `patience`, `min_delta`, `best_loss = float('inf')`, `counter = 0`, `should_stop = False`, and `best_state_dict = None` in `__init__`.",
            2: "In `step(self, val_loss, model)`, check `if val_loss < self.best_loss - self.min_delta:`. On improvement, save `copy.deepcopy(model.state_dict())` and reset `self.counter = 0`; else increment `self.counter` and set `self.should_stop = self.counter >= self.patience`.",
            3: "class EarlyStoppingTracker:\n    def __init__(self, patience=5, min_delta=0.0):\n        self.patience, self.min_delta = patience, min_delta\n        self.best_loss, self.counter = float('inf'), 0\n        self.should_stop, self.best_state_dict = False, None\n    def step(self, val_loss, model):\n        if val_loss < self.best_loss - self.min_delta:\n            self.best_loss = float(val_loss)\n            self.counter = 0\n            self.best_state_dict = copy.deepcopy(model.___())\n        else:\n            self.counter += 1\n            if self.counter >= self.patience:\n                self.should_stop = True\n        return self.should_stop",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 29.6: cosine_lr_schedule_trace(base_lr=0.01, t_max=20)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch29_todo_06", "To-Do 29.6", fn, "cosine_lr_schedule_trace", 10)
    if bad is not None:
        return bad

    cases = [
        (0.01, 20, "default (base_lr=0.01, t_max=20)"),
        (0.05, 10, "custom (base_lr=0.05, t_max=10)"),
    ]
    errors = []

    for base_lr, t_max, desc in cases:
        try:
            got = fn(base_lr=base_lr, t_max=t_max)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch29_todo_06",
                False,
                "failed",
                "To-Do 29.6 — Runtime Error in `cosine_lr_schedule_trace`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch29_todo_06", "To-Do 29.6", "cosine_lr_schedule_trace()", 10)

        if not isinstance(got, list) or len(got) != t_max:
            errors.append(
                f"[{desc}] Expected a Python list of length {t_max}, "
                f"got {type(got).__name__} of length {len(got) if hasattr(got, '__len__') else 'N/A'}."
            )
            continue

        # Build PyTorch reference trace (recording LR before step, and after step for helpful hint)
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=base_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        ref_before = []
        ref_after = []
        for _ in range(t_max):
            ref_before.append(round(float(opt.param_groups[0]["lr"]), 6))
            opt.step()
            sched.step()
            ref_after.append(round(float(opt.param_groups[0]["lr"]), 6))

        if not all(_is_np_or_py_number(v) for v in got):
            errors.append(f"[{desc}] All elements must be floats rounded to 6 decimals.")
            continue

        got_f = [float(v) for v in got]
        matches_before = all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got_f, ref_before))
        matches_after = all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got_f, ref_after))
        if not matches_before:
            if matches_after:
                errors.append(
                    f"[{desc}] Recorded LR *after* `sched.step()` instead of *before* it! "
                    "Append `round(opt.param_groups[0]['lr'], 6)` first so step 0 equals `base_lr`, then call `opt.step(); sched.step()`."
                )
            else:
                errors.append(
                    f"[{desc}] Schedule values did not match `CosineAnnealingLR(opt, T_max={t_max})`. "
                    f"Expected start={ref_before[:3]} ... end={ref_before[-2:]}, got start={got_f[:3]} ... end={got_f[-2:]}."
                )

    if errors:
        return CheckResult(
            "ch29_todo_06",
            False,
            "failed",
            "To-Do 29.6 — Check `CosineAnnealingLR` Schedule Trace",
            "Initialize a dummy `nn.Parameter`, `SGD([p], lr=base_lr)`, and `CosineAnnealingLR(opt, T_max=t_max)`; across `t_max` steps, record `round(opt.param_groups[0]['lr'], 6)` then call `opt.step(); sched.step()`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch29_todo_06",
        True,
        "passed",
        "To-Do 29.6 Passed — Cosine Annealing Learning-Rate Trace Verified!",
        "Produced exact smooth cosine decay trajectories starting at `base_lr` across both default and custom `t_max` schedules.",
        tip="Always call `optimizer.step()` before `scheduler.step()` in PyTorch, and log the LR used for that epoch cleanly.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_todo_06",
        chapter="ch29",
        title="To-Do 29.6: Cosine Annealing LR Trace (cosine_lr_schedule_trace)",
        book_section="Section 29.3.6 (Learning-Rate Schedules & Cosine Annealing)",
        points=10,
        hints={
            1: "Create a 1-element dummy `nn.Parameter(torch.zeros(1))`, wrap it in `torch.optim.SGD([param], lr=base_lr)`, and attach `torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)`.",
            2: "Loop `for _ in range(t_max):`: append `round(float(opt.param_groups[0]['lr']), 6)` to a list, then call `opt.step()` and `sched.step()`.",
            3: "def cosine_lr_schedule_trace(base_lr=0.01, t_max=20):\n    param = nn.Parameter(torch.zeros(1))\n    opt = torch.optim.SGD([param], lr=base_lr)\n    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=___)\n    lrs = []\n    for _ in range(t_max):\n        lrs.append(round(float(opt.param_groups[0]['lr']), 6))\n        opt.step()\n        sched.step()\n    return lrs",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 29.1 (30 pts): compare_optimizers_benchmark(...)
# ---------------------------------------------------------------------------
def _make_synthetic_cohort(seed: int = 42):
    import numpy as np
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(320, 6)).astype(np.float32)
    logits = 1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.9 * X[:, 2] * X[:, 3] - 0.7 * X[:, 4]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs + rng.normal(0.0, 0.12, size=320) >= 0.5).astype(int)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)


def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    import torch

    bad = _check_callable("ch29_exam_01", "Exam Task 29.1", fn, "compare_optimizers_benchmark", 30)
    if bad is not None:
        return bad

    torch.set_num_threads(1)
    X_tr, X_va, y_tr, y_va = _make_synthetic_cohort(42)
    X_tr_c, y_tr_c = X_tr.copy(), y_tr.copy()
    X_va_c, y_va_c = X_va.copy(), y_va.copy()

    expected_cols = ["optimizer", "final_train_loss", "final_val_loss", "val_auc"]
    expected_opts = ["SGD", "SGD+Momentum", "AdamW"]
    errors = []

    try:
        df1 = fn(X_tr_c, y_tr_c, X_va_c, y_va_c, epochs=35, seed=42)
        df2 = fn(X_tr.copy(), y_tr.copy(), X_va.copy(), y_va.copy(), epochs=35, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_exam_01",
            False,
            "failed",
            "Exam Task 29.1 — Runtime Error in `compare_optimizers_benchmark`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if df1 is None:
        return _none_return_trap(
            "ch29_exam_01", "Exam Task 29.1", "compare_optimizers_benchmark(...)", 30
        )
    if not isinstance(df1, pd.DataFrame):
        return CheckResult(
            "ch29_exam_01",
            False,
            "failed",
            "Exam Task 29.1 — Return a Pandas DataFrame",
            f"Expected a pd.DataFrame, got {type(df1).__name__}.",
            points_earned=0,
            points_possible=30,
        )

    if not (np.array_equal(X_tr_c, X_tr) and np.array_equal(y_tr_c, y_tr)):
        errors.append("Mutated caller training arrays in place!")

    if list(df1.columns) != expected_cols:
        errors.append(f"Expected columns {expected_cols}, got {list(df1.columns)}.")
    elif len(df1) != 3:
        errors.append(f"Expected 3 rows (SGD, SGD+Momentum, AdamW), got {len(df1)}.")
    else:
        df1_r = df1.reset_index(drop=True)
        df2_r = df2.reset_index(drop=True)
        got_opts = list(df1_r["optimizer"].astype(str))
        if got_opts != expected_opts:
            errors.append(f"Expected optimizer labels {expected_opts}, got {got_opts}.")
        for col in ("final_train_loss", "final_val_loss", "val_auc"):
            vals = [float(v) for v in df1_r[col]]
            if not all(math.isfinite(v) for v in vals):
                errors.append(f"Non-finite values in `{col}`: {vals}.")
            if not all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(vals, [float(v) for v in df2_r[col]])):
                errors.append(f"Results in `{col}` were NOT reproducible with the same `seed=42` — call `torch.manual_seed(seed)`!")

        if got_opts == expected_opts:
            by_opt = {row["optimizer"]: row for _, row in df1_r.iterrows()}
            sgd_tr = float(by_opt["SGD"]["final_train_loss"])
            mom_tr = float(by_opt["SGD+Momentum"]["final_train_loss"])
            adam_tr = float(by_opt["AdamW"]["final_train_loss"])
            adam_auc = float(by_opt["AdamW"]["val_auc"])
            if not (mom_tr < sgd_tr and adam_tr < sgd_tr):
                errors.append(
                    f"Expected SGD+Momentum ({mom_tr:.4f}) and AdamW ({adam_tr:.4f}) to reach lower train loss than plain SGD ({sgd_tr:.4f}) from identical initial weights!"
                )
            if adam_auc < 0.85:
                errors.append(
                    f"Expected AdamW `val_auc >= 0.85` on the separable benchmark, got {adam_auc:.4f}."
                )

    if errors:
        return CheckResult(
            "ch29_exam_01",
            False,
            "failed",
            "Exam Task 29.1 — Check Optimizer Benchmark Cloning, Seeding & Columns",
            "Clone identical initial weights (`copy.deepcopy(base.state_dict())`) for `SGD(lr=0.05)`, `SGD(lr=0.05, momentum=0.9)`, and `AdamW(lr=0.01, weight_decay=1e-2)` using `BCEWithLogitsLoss`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch29_exam_01",
        True,
        "passed",
        "Exam Task 29.1 Passed (30/30 pts) — Controlled Optimizer Benchmark Verified!",
        "Cloned identical initial weights cleanly, demonstrated faster convergence for SGD+Momentum and AdamW over plain SGD, and verified seed reproducibility.",
        points_earned=30,
        points_possible=30,
        tip="Fair optimizer ablations ALWAYS clone the exact same initial `state_dict()` so random weight luck cannot masquerade as an optimizer win.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_exam_01",
        chapter="ch29",
        title="Exam 29.1: Controlled Optimizer Benchmark (compare_optimizers_benchmark)",
        book_section="Chapter 29 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Seed with `torch.manual_seed(seed)`, construct a base MLP `nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))`, and save `base_state = copy.deepcopy(base.state_dict())`.",
            2: "For each of `('SGD', lr=0.05)`, `('SGD+Momentum', lr=0.05, momentum=0.9)`, and `('AdamW', lr=0.01, weight_decay=1e-2)`, load `copy.deepcopy(base_state)`, train `epochs` steps with `BCEWithLogitsLoss()`, and evaluate train loss, val loss, and `roc_auc_score` in `eval()` + `torch.no_grad()`.",
            3: "def compare_optimizers_benchmark(X_train, y_train, X_val, y_val, epochs=35, seed=42):\n    torch.manual_seed(seed)\n    in_dim = X_train.shape[1]\n    base = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))\n    base_sd = copy.deepcopy(base.state_dict())\n    # loop over [('SGD', ...), ('SGD+Momentum', ...), ('AdamW', ...)]\n    return pd.DataFrame(rows, columns=['optimizer', 'final_train_loss', 'final_val_loss', 'val_auc'])",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 29.2 (35 pts): audit_weight_decay_regularization(...)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import pandas as pd
    import torch

    bad = _check_callable("ch29_exam_02", "Exam Task 29.2", fn, "audit_weight_decay_regularization", 35)
    if bad is not None:
        return bad

    torch.set_num_threads(1)
    X_tr, X_va, y_tr, y_va = _make_synthetic_cohort(42)
    wds = [0.0, 0.05, 0.5, 2.0]
    expected_keys = ["best_wd", "table"]
    expected_cols = ["weight_decay", "val_loss", "val_auc", "l2_weight_norm"]
    errors = []

    try:
        got = fn(X_tr.copy(), y_tr.copy(), X_va.copy(), y_va.copy(), wds, epochs=40, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch29_exam_02",
            False,
            "failed",
            "Exam Task 29.2 — Runtime Error in `audit_weight_decay_regularization`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap(
            "ch29_exam_02", "Exam Task 29.2", "audit_weight_decay_regularization(...)", 35
        )
    key_err = _check_dict_keys("ch29_exam_02", "Exam Task 29.2", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    tbl = got["table"]
    if not isinstance(tbl, pd.DataFrame):
        errors.append(f"`table` must be a pd.DataFrame, got {type(tbl).__name__}.")
    elif list(tbl.columns) != expected_cols:
        errors.append(f"`table` columns mismatch: expected {expected_cols}, got {list(tbl.columns)}.")
    elif len(tbl) != len(wds):
        errors.append(f"`table` row count mismatch: expected {len(wds)}, got {len(tbl)}.")
    else:
        tbl_r = tbl.reset_index(drop=True)
        norms = [float(v) for v in tbl_r["l2_weight_norm"]]
        vlosses = [float(v) for v in tbl_r["val_loss"]]
        vaucs = [float(v) for v in tbl_r["val_auc"]]

        if not all(math.isfinite(v) for v in norms + vlosses + vaucs):
            errors.append("Found non-finite values in `table`.")
        if not all(norms[i] > norms[i + 1] for i in range(len(norms) - 1)):
            errors.append(
                f"Expected `l2_weight_norm` to shrink strictly as `weight_decay` increases, got {norms} "
                "(Check that you pass `weight_decay=wd` to `AdamW` and clone identical initial weights for each run!)."
            )
        if max(vaucs) < 0.85:
            errors.append(f"Expected top `val_auc >= 0.85` on the benchmark cohort, got max={max(vaucs):.4f}.")

        best_idx = int(tbl_r["val_loss"].idxmin())
        exp_best_wd = float(tbl_r.loc[best_idx, "weight_decay"])
        if not _is_np_or_py_number(got["best_wd"]) or not math.isclose(float(got["best_wd"]), exp_best_wd, abs_tol=1e-4):
            errors.append(
                f"`best_wd` expected {exp_best_wd} (lowest `val_loss` row), got {got['best_wd']!r}."
            )

    if errors:
        return CheckResult(
            "ch29_exam_02",
            False,
            "failed",
            "Exam Task 29.2 — Check Weight Decay Sweep, L2 Norm & `best_wd`",
            "Clone identical initial weights for each `wd`, train with `AdamW(lr=0.01, weight_decay=wd)`, compute L2 norm across weight matrices (`p.ndim >= 2`), and return `{'best_wd': float, 'table': DataFrame}`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch29_exam_02",
        True,
        "passed",
        "Exam Task 29.2 Passed (35/35 pts) — Decoupled Weight Decay Audit Verified!",
        "Confirmed strictly monotonic L2 weight-norm shrinkage as `AdamW` weight decay increases and accurate `best_wd` selection.",
        points_earned=35,
        points_possible=35,
        tip="In AdamW, decoupled weight decay directly shrinks weight magnitudes on every step without distorting adaptive gradient scaling.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_exam_02",
        chapter="ch29",
        title="Exam 29.2: AdamW Weight Decay Regularization Audit (audit_weight_decay_regularization)",
        book_section="Chapter 29 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Seed with `torch.manual_seed(seed)`, snapshot `base_sd = copy.deepcopy(base.state_dict())`, and reload it cleanly for every `wd` in `weight_decays`.",
            2: "After training 40 steps with `AdamW(model.parameters(), lr=0.01, weight_decay=wd)`, compute weight-matrix L2 norm `math.sqrt(sum(float(p.detach().pow(2).sum().item()) for p in model.parameters() if p.ndim >= 2))`.",
            3: "def audit_weight_decay_regularization(X_train, y_train, X_val, y_val, weight_decays, epochs=40, seed=42):\n    # build base MLP, clone state_dict for each wd, record val_loss, val_auc, l2_weight_norm\n    tbl = pd.DataFrame(rows, columns=['weight_decay', 'val_loss', 'val_auc', 'l2_weight_norm'])\n    best_wd = float(tbl.loc[tbl['val_loss'].idxmin(), 'weight_decay'])\n    return {'best_wd': best_wd, 'table': tbl}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 29.3 (35 pts): train_with_early_stopping_mch(...)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import torch

    from ..datasets import load_data

    bad = _check_callable("ch29_exam_03", "Exam Task 29.3", fn, "train_with_early_stopping_mch", 35)
    if bad is not None:
        return bad

    torch.set_num_threads(1)
    df_mch = load_data("mch_survey_sample", as_pandas=True)
    expected_keys = ["epochs_run", "best_val_loss", "test_auc", "stopped_early"]
    errors = []

    test_cases = [
        (8, 80, 42, True, "default (patience=8, max_epochs=80, seed=42)"),
        (8, 5, 42, False, "short budget (patience=8, max_epochs=5, seed=42) cannot trigger early stop"),
    ]

    for patience, max_epochs, seed, expect_stop, desc in test_cases:
        df_copy = df_mch.copy(deep=True)
        try:
            got = fn(df_copy, patience=patience, max_epochs=max_epochs, seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch29_exam_03",
                False,
                "failed",
                "Exam Task 29.3 — Runtime Error in `train_with_early_stopping_mch`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch29_exam_03", "Exam Task 29.3", "train_with_early_stopping_mch(mch_df)", 35
            )
        key_err = _check_dict_keys("ch29_exam_03", "Exam Task 29.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_mch):
            errors.append(f"[{desc}] Mutated caller `mch_df`! Start with `work = mch_df.copy()`.")

        ep_run = got["epochs_run"]
        b_loss = got["best_val_loss"]
        t_auc = got["test_auc"]
        stopped = got["stopped_early"]

        if not isinstance(ep_run, int) or isinstance(ep_run, bool) or not (1 <= ep_run <= max_epochs):
            errors.append(f"[{desc}] `epochs_run` must be int in [1, {max_epochs}], got {ep_run!r}.")
        if not isinstance(stopped, bool):
            errors.append(f"[{desc}] `stopped_early` must be a Python `bool`, got {type(stopped).__name__}.")
        elif stopped is not expect_stop:
            errors.append(
                f"[{desc}] `stopped_early` expected {expect_stop}, got {stopped} "
                f"(epochs_run={ep_run}, patience={patience}, max_epochs={max_epochs})."
            )
        if expect_stop and isinstance(ep_run, int) and ep_run >= max_epochs:
            errors.append(f"[{desc}] `stopped_early=True` requires stopping before `max_epochs={max_epochs}`.")
        if not _is_np_or_py_number(b_loss) or not (0.20 < float(b_loss) < 1.20):
            errors.append(f"[{desc}] `best_val_loss` ({b_loss!r}) looks out of expected clinical BCE range (0.20..1.20).")
        if not _is_np_or_py_number(t_auc) or not (0.50 <= float(t_auc) <= 0.90):
            extra = " (AUC > 0.90 suggests target leakage — exclude `maternal_hb_g_dl`!)" if _is_np_or_py_number(t_auc) and float(t_auc) > 0.90 else ""
            errors.append(f"[{desc}] `test_auc` ({t_auc!r}) outside expected leak-free range [0.50, 0.90]{extra}.")

    if errors:
        return CheckResult(
            "ch29_exam_03",
            False,
            "failed",
            "Exam Task 29.3 — Check MCH Cleaning, 60/20/20 Split, Scaler & Early Stopping",
            "Clean `mch_survey_sample` without mutation or target leakage, perform stratified 60/20/20 split with train-only `StandardScaler`, train `RegularizedClinicalMLP` with `AdamW` + `CosineAnnealingLR` + `EarlyStoppingTracker`, restore `best_state_dict`, and evaluate `test_auc` in `eval()` mode.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch29_exam_03",
        True,
        "passed",
        "Exam Task 29.3 Passed (35/35 pts) — Full Regularized MCH Early-Stopping Pipeline Verified!",
        "Passed leak-free cleaning, stratified 60/20/20 splitting, train-only scaling, CosineAnnealingLR, early-stopping restoration, and short-budget edge case.",
        points_earned=35,
        points_possible=35,
        tip="Restoring `best_state_dict` before evaluating `X_test` ensures your reported test AUC reflects the validation optimum—not the overfit epoch where patience expired!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch29_exam_03",
        chapter="ch29",
        title="Exam 29.3: End-to-End Regularized MCH Pipeline with Early Stopping (train_with_early_stopping_mch)",
        book_section="Chapter 29 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Copy `mch_df`, clean `residence`, replace `anc_visits == 999` with `np.nan`, encode `is_urban` and `wealth_num` (1..5), drop NaNs on `['maternal_anemia', 'maternal_age_years', 'anc_visits', 'wealth_num', 'is_urban']`, and split stratified 60/20/20.",
            2: "Fit `StandardScaler()` strictly on `X_train`, transform `X_val` and `X_test`. Seed `torch.manual_seed(seed)`, train `RegularizedClinicalMLP(4, 32, 0.3)` with `AdamW(lr=0.02, weight_decay=1e-2)`, `CosineAnnealingLR(opt, T_max=max_epochs)`, and `EarlyStoppingTracker(patience=patience)`.",
            3: "def train_with_early_stopping_mch(mch_df, patience=8, max_epochs=80, seed=42):\n    # clean -> 60/20/20 split -> train-only StandardScaler\n    # train loop: train() step -> sched.step() -> eval() val_loss -> if tracker.step(val_loss, model): break\n    model.load_state_dict(tracker.best_state_dict)\n    model.eval()\n    with torch.no_grad():\n        probs = torch.sigmoid(model(Xte_t)).squeeze(1).numpy()\n    return {'epochs_run': int(ep), 'best_val_loss': round(float(tracker.best_loss), 5), 'test_auc': round(float(roc_auc_score(y_te, probs)), 4), 'stopped_early': bool(tracker.should_stop)}",
        },
        verifier=_verify_exam_03,
    )
)
