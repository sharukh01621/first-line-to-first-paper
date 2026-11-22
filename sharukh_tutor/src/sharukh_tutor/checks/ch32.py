"""Chapter 32 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch32_todo_01..06) and the Part VI Capstone Coding Exam
(ch32_exam_01..03) for Attention, Transformers, Fine-Tuning, and Modern
Foundation Models. Every verifier lazy-imports PyTorch/NumPy/Scikit-Learn
inside the check function, runs 100% offline on CPU in seconds with zero
external downloads, and lets NotImplementedError propagate cleanly for
starter stubs.
"""

from __future__ import annotations

import math
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
        f"Calling {call_str} returned None instead of the expected result.",
        details=[
            "Remember: print() only displays text; functions must end with `return ...`.",
            f"Ensure {call_str} explicitly returns the required tensor, tuple, or dictionary.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="PyTorch building blocks return tensors or (output, attn_weights) tuples so autograd can backpropagate through them.",
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


def _ref_sdpa(Q: Any, K: Any, V: Any, mask: Any = None) -> tuple[Any, Any]:
    import torch

    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


def _ref_sinusoidal_pe(max_len: int, d_model: int) -> Any:
    import torch

    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe


# ---------------------------------------------------------------------------
# GUIDED TO-DO 32.1: scaled_dot_product_attention(Q, K, V, mask=None)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import torch

    bad = _check_callable(
        "ch32_todo_01", "To-Do 32.1", fn, "scaled_dot_product_attention", 10
    )
    if bad is not None:
        return bad

    torch.manual_seed(42)
    Q3 = torch.randn(2, 4, 8)
    K3 = torch.randn(2, 4, 8)
    V3 = torch.randn(2, 4, 6)
    Q3_orig = Q3.clone()

    try:
        res3 = fn(Q3, K3, V3)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_01",
            False,
            "failed",
            "To-Do 32.1 — Runtime Error in `scaled_dot_product_attention`",
            f"Calling scaled_dot_product_attention(Q, K, V) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `K.transpose(-2, -1)` so the function works for both 3D (B, T, d_k) and 4D (B, H, T, d_k) tensors.",
                "Divide dot products by `math.sqrt(Q.size(-1))` before `torch.softmax(scores, dim=-1)`.",
            ],
            points_earned=0,
            points_possible=10,
        )

    if res3 is None:
        return _none_return_trap(
            "ch32_todo_01", "To-Do 32.1", "scaled_dot_product_attention(Q, K, V)", 10
        )

    if not isinstance(res3, tuple) or len(res3) != 2:
        return CheckResult(
            "ch32_todo_01",
            False,
            "failed",
            "To-Do 32.1 — Return a `(context, attn_weights)` Tuple",
            f"Expected a 2-tuple `(attn @ V, attn)`, got {type(res3).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    out3, attn3 = res3
    errors: list[str] = []

    if not torch.equal(Q3, Q3_orig):
        errors.append("Input tensor `Q` was mutated in place! Avoid in-place edits on inputs.")

    if not isinstance(out3, torch.Tensor) or not isinstance(attn3, torch.Tensor):
        errors.append("Both returned elements must be `torch.Tensor` objects.")
    else:
        exp_out3, exp_attn3 = _ref_sdpa(Q3, K3, V3)
        if tuple(out3.shape) != (2, 4, 6):
            errors.append(f"3D output shape mismatch: expected (2, 4, 6), got {tuple(out3.shape)}.")
        if tuple(attn3.shape) != (2, 4, 4):
            errors.append(f"3D attention shape mismatch: expected (2, 4, 4), got {tuple(attn3.shape)}.")

        # Detect missing 1/sqrt(d_k) scaling specifically
        unscaled_attn = torch.softmax(Q3 @ K3.transpose(-2, -1), dim=-1)
        if tuple(attn3.shape) == tuple(unscaled_attn.shape) and torch.allclose(
            attn3, unscaled_attn, atol=1e-4
        ):
            errors.append(
                "Attention weights match UNSCALED `Q @ K^T`! "
                "Divide scores by `math.sqrt(Q.size(-1))` to prevent softmax saturation."
            )
        elif tuple(attn3.shape) == tuple(exp_attn3.shape) and not torch.allclose(
            attn3, exp_attn3, atol=1e-5
        ):
            errors.append("Unmasked 3D attention weights do not match `softmax((Q @ K^T) / sqrt(d_k), dim=-1)`.")

        if tuple(out3.shape) == tuple(exp_out3.shape) and not torch.allclose(
            out3, exp_out3, atol=1e-5
        ):
            errors.append("Unmasked 3D context output does not match `attn @ V`.")

    # 4D multi-head test with causal mask (1=allowed, 0=masked)
    torch.manual_seed(7)
    Q4 = torch.randn(2, 3, 5, 4)
    K4 = torch.randn(2, 3, 5, 4)
    V4 = torch.randn(2, 3, 5, 4)
    causal_mask = torch.tril(torch.ones(1, 1, 5, 5))

    try:
        out4, attn4 = fn(Q4, K4, V4, mask=causal_mask)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"4D masked attention raised {type(exc).__name__}: {exc}")
    else:
        exp_out4, exp_attn4 = _ref_sdpa(Q4, K4, V4, mask=causal_mask)
        if tuple(out4.shape) != (2, 3, 5, 4) or tuple(attn4.shape) != (2, 3, 5, 5):
            errors.append(
                f"4D masked shapes mismatch: expected out=(2, 3, 5, 4), attn=(2, 3, 5, 5), "
                f"got out={tuple(out4.shape)}, attn={tuple(attn4.shape)}."
            )
        else:
            upper = torch.triu(attn4, diagonal=1).abs().max().item()
            if upper > 1e-6:
                errors.append(
                    f"Masked future positions have non-zero attention weight (max={upper:.4f})! "
                    "Use `scores = scores.masked_fill(mask == 0, float('-inf'))` BEFORE softmax."
                )
            row_err = (attn4.sum(dim=-1) - 1.0).abs().max().item()
            if row_err > 1e-5:
                errors.append(f"Attention rows do not sum to 1.0 (max error={row_err:.5f}).")
            if not torch.allclose(out4, exp_out4, atol=1e-5) or not torch.allclose(
                attn4, exp_attn4, atol=1e-5
            ):
                errors.append("4D masked output/attention does not match reference calculation.")

    if errors:
        return CheckResult(
            "ch32_todo_01",
            False,
            "failed",
            "To-Do 32.1 — Check Scaled Dot-Product Attention Math & Masking",
            "Compute `scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)`, mask `mask == 0` with `float('-inf')`, apply `torch.softmax(scores, dim=-1)`, and return `(attn @ V, attn)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch32_todo_01",
        True,
        "passed",
        "To-Do 32.1 Passed — `scaled_dot_product_attention` Verified!",
        "Passed 3D (d_v != d_k), 4D multi-head, sqrt(d_k) variance scaling, causal mask == 0 zeroing, and row-stochastic softmax checks.",
        tip="Dividing by sqrt(d_k) keeps dot-product variance near 1.0 so softmax gradients stay healthy instead of saturating to 0.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_todo_01",
        chapter="ch32",
        title="To-Do 32.1: Scaled Dot-Product Attention (scaled_dot_product_attention)",
        book_section="Section 32.3.1 (Scaled Dot-Product Attention & Masking)",
        points=10,
        hints={
            1: "Compute `d_k = Q.size(-1)` and `scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)`.",
            2: "If `mask is not None`, fill blocked positions via `scores = scores.masked_fill(mask == 0, float('-inf'))`. Then apply `attn = torch.softmax(scores, dim=-1)` and return `(attn @ V, attn)`.",
            3: "def scaled_dot_product_attention(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(___)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    attn = torch.softmax(scores, dim=___)\n    return attn @ V, attn",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 32.2: sinusoidal_positional_encoding(max_len, d_model)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import torch

    bad = _check_callable(
        "ch32_todo_02", "To-Do 32.2", fn, "sinusoidal_positional_encoding", 10
    )
    if bad is not None:
        return bad

    errors: list[str] = []
    for max_len, d_model in [(16, 8), (32, 16)]:
        try:
            got = fn(max_len, d_model)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch32_todo_02",
                False,
                "failed",
                "To-Do 32.2 — Runtime Error in `sinusoidal_positional_encoding`",
                f"Calling sinusoidal_positional_encoding({max_len}, {d_model}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch32_todo_02",
                "To-Do 32.2",
                f"sinusoidal_positional_encoding({max_len}, {d_model})",
                10,
            )

        if not isinstance(got, torch.Tensor):
            errors.append(f"Expected torch.Tensor, got {type(got).__name__}.")
            continue
        if got.dtype != torch.float32:
            errors.append(
                f"Expected dtype torch.float32, got {got.dtype} (use `dtype=torch.float32`)."
            )
        if tuple(got.shape) != (max_len, d_model):
            errors.append(
                f"Shape mismatch for ({max_len}, {d_model}): got {tuple(got.shape)}."
            )
            continue

        ref = _ref_sinusoidal_pe(max_len, d_model)
        if not torch.allclose(got[0, 0::2], torch.zeros(d_model // 2), atol=1e-5):
            errors.append(
                f"[({max_len}, {d_model})] Position 0 even dimensions should be sin(0)=0.0."
            )
        if not torch.allclose(got[0, 1::2], torch.ones(d_model // 2), atol=1e-5):
            errors.append(
                f"[({max_len}, {d_model})] Position 0 odd dimensions should be cos(0)=1.0."
            )
        if not torch.allclose(got, ref, atol=1e-5):
            errors.append(
                f"[({max_len}, {d_model})] Values do not match Vaswani et al. (2017) sin/cos formula "
                "(check `div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))`)."
            )

    if errors:
        return CheckResult(
            "ch32_todo_02",
            False,
            "failed",
            "To-Do 32.2 — Check Sinusoidal Positional Encoding Formula",
            "Return a float32 tensor of shape `(max_len, d_model)` with `sin(pos / 10000^(2i/d_model))` on even indices `0::2` and `cos(...)` on odd indices `1::2`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch32_todo_02",
        True,
        "passed",
        "To-Do 32.2 Passed — `sinusoidal_positional_encoding` Verified!",
        "Verified exact Vaswani et al. (2017) multi-frequency sine/cosine encodings in torch.float32 across (16, 8) and (32, 16).",
        tip="Self-attention is permutation-equivariant by itself; adding positional encodings lets a Transformer distinguish 'no chest pain' from 'chest pain no'.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_todo_02",
        chapter="ch32",
        title="To-Do 32.2: Sinusoidal Positional Encoding (sinusoidal_positional_encoding)",
        book_section="Section 32.3.2 (Sinusoidal Positional Encoding)",
        points=10,
        hints={
            1: "Create `pe = torch.zeros(max_len, d_model, dtype=torch.float32)` and column vector `pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)`.",
            2: "Compute `div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))`, then assign `pe[:, 0::2] = torch.sin(pos * div_term)` and `pe[:, 1::2] = torch.cos(pos * div_term)`.",
            3: "def sinusoidal_positional_encoding(max_len, d_model):\n    pe = torch.zeros(max_len, d_model, dtype=torch.float32)\n    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)\n    div_term = torch.exp(\n        torch.arange(0, d_model, 2, dtype=torch.float32)\n        * (-math.log(10000.0) / d_model)\n    )\n    pe[:, 0::2] = torch.sin(pos * ___)\n    pe[:, 1::2] = torch.cos(pos * div_term[: d_model // 2])\n    return pe",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 32.3: MultiHeadSelfAttention (nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_03(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    if not isinstance(cls, type):
        return CheckResult(
            "ch32_todo_03",
            False,
            "failed",
            "To-Do 32.3 — Pass the Class Name `MultiHeadSelfAttention`",
            f"Expected a class (type), got {type(cls).__name__} ({cls!r}).",
            details=["Pass the class itself: tutor.check('ch32_todo_03', MultiHeadSelfAttention)"],
            points_earned=0,
            points_possible=10,
        )

    if not issubclass(cls, nn.Module):
        return CheckResult(
            "ch32_todo_03",
            False,
            "failed",
            "To-Do 32.3 — Inherit From `nn.Module`",
            "Define `class MultiHeadSelfAttention(nn.Module):` and call `super().__init__()`.",
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []

    # Check divisibility validation
    try:
        _ = cls(d_model=30, num_heads=4)
    except NotImplementedError:
        raise
    except (ValueError, AssertionError):
        pass
    except Exception as exc:
        errors.append(
            f"Invalid (d_model=30, num_heads=4) raised unexpected {type(exc).__name__}; expected ValueError or AssertionError."
        )
    else:
        errors.append(
            "MultiHeadSelfAttention(d_model=30, num_heads=4) did NOT fail! "
            "Check `d_model % num_heads == 0` in `__init__` and raise ValueError (or assert)."
        )

    torch.manual_seed(11)
    try:
        mha = cls(d_model=16, num_heads=4)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_03",
            False,
            "failed",
            "To-Do 32.3 — Error Instantiating `MultiHeadSelfAttention(16, 4)`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    for attr in ("W_q", "W_k", "W_v", "W_o"):
        proj = getattr(mha, attr, None)
        if not isinstance(proj, nn.Linear) or proj.in_features != 16 or proj.out_features != 16:
            errors.append(
                f"Expected `self.{attr} = nn.Linear(d_model, d_model)` (16 -> 16), got {proj!r}."
            )

    x = torch.randn(2, 5, 16)
    mask = torch.tril(torch.ones(1, 1, 5, 5))
    try:
        res = mha(x, mask=mask)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_03",
            False,
            "failed",
            "To-Do 32.3 — Runtime Error in `MultiHeadSelfAttention.forward`",
            f"Calling mha(x, mask=mask) raised {type(exc).__name__}: {exc}",
            details=[
                "Reshape projected tensors via `.view(B, T, num_heads, d_k).transpose(1, 2)`.",
                "After attention, recombine heads via `context.transpose(1, 2).contiguous().view(B, T, d_model)` before `self.W_o(...)`.",
            ],
            points_earned=0,
            points_possible=10,
        )

    if res is None:
        return _none_return_trap("ch32_todo_03", "To-Do 32.3", "mha(x, mask=mask)", 10)

    if not isinstance(res, tuple) or len(res) != 2:
        return CheckResult(
            "ch32_todo_03",
            False,
            "failed",
            "To-Do 32.3 — Return `(out, attn_weights)` Tuple",
            f"Expected `(out[B, T, d_model], attn_weights[B, H, T, T])`, got {type(res).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    out, attn = res
    if tuple(out.shape) != (2, 5, 16):
        errors.append(f"Output shape mismatch: expected (2, 5, 16), got {tuple(out.shape)}.")
    if tuple(attn.shape) != (2, 4, 5, 5):
        errors.append(f"Attention shape mismatch: expected (2, 4, 5, 5), got {tuple(attn.shape)}.")

    if not errors:
        # Verify exact mathematical equivalence using student's own W_q, W_k, W_v, W_o
        B, T, D = x.shape
        H, d_k = 4, 4
        Q = mha.W_q(x).view(B, T, H, d_k).transpose(1, 2)
        K = mha.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        V = mha.W_v(x).view(B, T, H, d_k).transpose(1, 2)
        ctx, ref_attn = _ref_sdpa(Q, K, V, mask=mask)
        ref_out = mha.W_o(ctx.transpose(1, 2).contiguous().view(B, T, D))

        if not torch.allclose(attn, ref_attn, atol=1e-5):
            errors.append(
                "Multi-head attention weights do not match reference split-head SDPA "
                "(check `.view(B, T, H, d_k).transpose(1, 2)` ordering!)."
            )
        if not torch.allclose(out, ref_out, atol=1e-5):
            errors.append(
                "Output projection does not match `W_o(concat(heads))` "
                "(check `.transpose(1, 2).contiguous().view(B, T, d_model)` before `self.W_o`)."
            )

        # Verify gradient flow
        loss = out.pow(2).mean()
        loss.backward()
        for attr in ("W_q", "W_k", "W_v", "W_o"):
            if getattr(mha, attr).weight.grad is None:
                errors.append(f"No gradient flowed to `self.{attr}.weight` during `.backward()`.")

    if errors:
        return CheckResult(
            "ch32_todo_03",
            False,
            "failed",
            "To-Do 32.3 — Check `MultiHeadSelfAttention` Implementation",
            "Verify `d_model % num_heads == 0`, projections `W_q, W_k, W_v, W_o`, split/merge head shapes `(B, H, T, d_k)`, and return `(out, attn_weights)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch32_todo_03",
        True,
        "passed",
        "To-Do 32.3 Passed — `MultiHeadSelfAttention` Verified!",
        "Passed divisibility guard, W_q/W_k/W_v/W_o projections, exact multi-head split/merge numerical parity, causal mask broadcasting, and full gradient flow.",
        tip="Never call `.view(B, H, T, d_k)` directly on `(B, T, d_model)` — always `.view(B, T, H, d_k).transpose(1, 2)` so token boundaries stay intact!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_todo_03",
        chapter="ch32",
        title="To-Do 32.3: Multi-Head Self-Attention (MultiHeadSelfAttention)",
        book_section="Section 32.3.3 (Multi-Head Self-Attention)",
        points=10,
        hints={
            1: "In `__init__(self, d_model, num_heads)`, verify `d_model % num_heads == 0`, set `self.d_k = d_model // num_heads`, and create `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o` as `nn.Linear(d_model, d_model)`.",
            2: "In `forward(self, x, mask=None)`, split heads via `Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)` (same for K, V), call `scaled_dot_product_attention(Q, K, V, mask=mask)`, then merge via `ctx.transpose(1, 2).contiguous().view(B, T, self.d_model)`.",
            3: "class MultiHeadSelfAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        if d_model % num_heads != 0:\n            raise ValueError('d_model must be divisible by num_heads')\n        self.d_model, self.num_heads = d_model, num_heads\n        self.d_k = d_model // num_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n    def forward(self, x, mask=None):\n        B, T, _ = x.shape\n        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)\n        ctx, attn = scaled_dot_product_attention(Q, K, V, mask=___)\n        merged = ctx.transpose(1, 2).contiguous().view(B, T, self.d_model)\n        return self.W_o(merged), attn",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 32.4: TransformerEncoderBlock (nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_04(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    if not isinstance(cls, type):
        return CheckResult(
            "ch32_todo_04",
            False,
            "failed",
            "To-Do 32.4 — Pass the Class Name `TransformerEncoderBlock`",
            f"Expected a class (type), got {type(cls).__name__} ({cls!r}).",
            details=["Pass the class itself: tutor.check('ch32_todo_04', TransformerEncoderBlock)"],
            points_earned=0,
            points_possible=10,
        )

    if not issubclass(cls, nn.Module):
        return CheckResult(
            "ch32_todo_04",
            False,
            "failed",
            "To-Do 32.4 — Inherit From `nn.Module`",
            "Define `class TransformerEncoderBlock(nn.Module):` and call `super().__init__()`.",
            points_earned=0,
            points_possible=10,
        )

    torch.manual_seed(17)
    try:
        blk = cls(d_model=16, num_heads=4, d_ff=32, dropout=0.1)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_04",
            False,
            "failed",
            "To-Do 32.4 — Error Instantiating `TransformerEncoderBlock`",
            f"TransformerEncoderBlock(16, 4, 32, dropout=0.1) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []
    submodules = list(blk.modules())
    n_ln = sum(1 for m in submodules if isinstance(m, nn.LayerNorm))
    if n_ln < 2:
        errors.append(
            f"Expected at least 2 `nn.LayerNorm` layers (post-attention & post-FFN), found {n_ln}."
        )
    has_ffn_up = any(
        isinstance(m, nn.Linear) and m.in_features == 16 and m.out_features == 32
        for m in submodules
    )
    has_ffn_down = any(
        isinstance(m, nn.Linear) and m.in_features == 32 and m.out_features == 16
        for m in submodules
    )
    if not (has_ffn_up and has_ffn_down):
        errors.append(
            "Expected a 2-layer feed-forward network with `Linear(d_model, d_ff)` -> `GELU` -> `Linear(d_ff, d_model)`."
        )
    has_gelu = any(isinstance(m, nn.GELU) for m in submodules)
    if not has_gelu:
        errors.append("Expected an `nn.GELU()` activation inside the feed-forward sublayer.")

    blk.eval()
    x = torch.randn(2, 6, 16)
    pad_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float32
    ).view(2, 1, 1, 6)

    try:
        res = blk(x, mask=pad_mask)
        res_repeat = blk(x, mask=pad_mask)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_04",
            False,
            "failed",
            "To-Do 32.4 — Runtime Error in `TransformerEncoderBlock.forward`",
            f"Calling blk(x, mask=pad_mask) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if res is None:
        return _none_return_trap("ch32_todo_04", "To-Do 32.4", "blk(x, mask=pad_mask)", 10)

    if not isinstance(res, tuple) or len(res) != 2:
        return CheckResult(
            "ch32_todo_04",
            False,
            "failed",
            "To-Do 32.4 — Return `(out, attn_weights)` Tuple",
            f"Expected `(out[B, T, d_model], attn_weights[B, H, T, T])`, got {type(res).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    out, attn = res
    if tuple(out.shape) != (2, 6, 16):
        errors.append(f"Output shape mismatch: expected (2, 6, 16), got {tuple(out.shape)}.")
    if tuple(attn.shape) != (2, 4, 6, 6):
        errors.append(f"Attention shape mismatch: expected (2, 4, 6, 6), got {tuple(attn.shape)}.")
    if not torch.isfinite(out).all():
        errors.append("Output tensor contains NaN or Inf values.")
    if not torch.allclose(out, res_repeat[0], atol=1e-6):
        errors.append("Output is non-deterministic in `.eval()` mode (check dropout handling).")

    if tuple(attn.shape) == (2, 4, 6, 6):
        pad_leak_0 = attn[0, :, :4, 4:].abs().max().item()
        pad_leak_1 = attn[1, :, :5, 5:].abs().max().item()
        if max(pad_leak_0, pad_leak_1) > 1e-6:
            errors.append(
                "Padding positions received non-zero attention weight! Forward `mask` into `MultiHeadSelfAttention`."
            )

    # Verify gradient flow in train mode
    blk.train()
    out_tr, _ = blk(x, mask=pad_mask)
    out_tr.pow(2).mean().backward()
    no_grad = [name for name, p in blk.named_parameters() if p.requires_grad and p.grad is None]
    if no_grad:
        errors.append(f"Parameters received no gradient during `.backward()`: {no_grad[:4]}.")

    if errors:
        return CheckResult(
            "ch32_todo_04",
            False,
            "failed",
            "To-Do 32.4 — Check `TransformerEncoderBlock` Architecture",
            "Combine `MultiHeadSelfAttention`, two `nn.LayerNorm` layers, residual connections, dropout, and `Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch32_todo_04",
        True,
        "passed",
        "To-Do 32.4 Passed — `TransformerEncoderBlock` Verified!",
        "Verified MHA + 2 LayerNorms + GELU FFN + residual connections, padding-mask propagation, eval determinism, and full backpropagation.",
        tip="Residual connections (`x + sublayer(x)`) plus LayerNorm are what allow stacking 12–96 Transformer blocks cleanly without vanishing gradients.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_todo_04",
        chapter="ch32",
        title="To-Do 32.4: Transformer Encoder Block (TransformerEncoderBlock)",
        book_section="Section 32.3.4 (Transformer Encoder Block)",
        points=10,
        hints={
            1: "In `__init__(self, d_model, num_heads, d_ff, dropout=0.1)`, store `self.attn = MultiHeadSelfAttention(d_model, num_heads)`, `self.norm1 = nn.LayerNorm(d_model)`, `self.norm2 = nn.LayerNorm(d_model)`, `self.drop = nn.Dropout(dropout)`, and a 2-layer `self.ffn` with `nn.GELU()`.",
            2: "In `forward(self, x, mask=None)`, run `attn_out, weights = self.attn(x, mask=mask)`, update `x = self.norm1(x + self.drop(attn_out))`, then `x = self.norm2(x + self.drop(self.ffn(x)))`, and return `(x, weights)`.",
            3: "class TransformerEncoderBlock(nn.Module):\n    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):\n        super().__init__()\n        self.attn = MultiHeadSelfAttention(d_model, num_heads)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)\n        self.drop = nn.Dropout(dropout)\n        self.ffn = nn.Sequential(\n            nn.Linear(d_model, d_ff),\n            nn.GELU(),\n            nn.Dropout(dropout),\n            nn.Linear(d_ff, d_model),\n        )\n    def forward(self, x, mask=None):\n        attn_out, weights = self.attn(x, mask=___)\n        x = self.norm1(x + self.drop(attn_out))\n        x = self.norm2(x + self.drop(self.ffn(x)))\n        return x, weights",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 32.5: ClinicalVocabTokenizer
# ---------------------------------------------------------------------------
def _verify_todo_05(cls: Any) -> CheckResult:
    if not isinstance(cls, type):
        return CheckResult(
            "ch32_todo_05",
            False,
            "failed",
            "To-Do 32.5 — Pass the Class Name `ClinicalVocabTokenizer`",
            f"Expected a class (type), got {type(cls).__name__} ({cls!r}).",
            details=["Pass the class itself: tutor.check('ch32_todo_05', ClinicalVocabTokenizer)"],
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []
    if getattr(cls, "PAD_ID", None) != 0:
        errors.append(f"Class constant `PAD_ID` must be 0, got {getattr(cls, 'PAD_ID', None)!r}.")
    if getattr(cls, "UNK_ID", None) != 1:
        errors.append(f"Class constant `UNK_ID` must be 1, got {getattr(cls, 'UNK_ID', None)!r}.")

    notes_a = ["Severe Chest Pain", "mild cough stable", "fever chest"]
    notes_b = list(reversed(notes_a))

    try:
        tok_a = cls()
        fit_ret = tok_a.fit(notes_a)
        tok_b = cls()
        tok_b.fit(notes_b)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_05",
            False,
            "failed",
            "To-Do 32.5 — Runtime Error in `ClinicalVocabTokenizer.fit`",
            f"fit(texts) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if fit_ret is not tok_a:
        errors.append("`.fit(texts)` should return `self` so chaining `ClinicalVocabTokenizer().fit(texts)` works.")

    # Unique sorted tokens: chest, cough, fever, mild, pain, severe, stable -> 7 + 2 = 9
    try:
        vlen = len(tok_a)
    except Exception as exc:
        errors.append(f"`len(tokenizer)` failed ({type(exc).__name__}: {exc}) — implement `__len__(self)`.")
        vlen = -1

    if vlen != 9:
        errors.append(
            f"Expected `len(tokenizer) == 9` (2 special tokens + 7 unique lowercased words), got {vlen}."
        )

    try:
        enc_short = tok_a.encode("CHEST pain", max_len=5)
        enc_short_b = tok_b.encode("CHEST pain", max_len=5)
        enc_oov = tok_a.encode("chest hypoxia", max_len=4)
        enc_trunc = tok_a.encode("severe chest pain fever mild cough", max_len=3)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_05",
            False,
            "failed",
            "To-Do 32.5 — Runtime Error in `ClinicalVocabTokenizer.encode`",
            f"encode(text, max_len) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if enc_short is None:
        return _none_return_trap("ch32_todo_05", "To-Do 32.5", "tokenizer.encode(text, max_len)", 10)

    if not isinstance(enc_short, list) or not all(isinstance(i, int) for i in enc_short):
        errors.append(f"`encode()` must return `list[int]`, got {enc_short!r}.")
    else:
        if len(enc_short) != 5 or enc_short[2:] != [0, 0, 0]:
            errors.append(
                f"Expected length 5 with right `<PAD>` (0) padding for 'CHEST pain', got {enc_short!r}."
            )
        # Sorted vocab: 2=chest, 3=cough, 4=fever, 5=mild, 6=pain, 7=severe, 8=stable
        if enc_short != [2, 6, 0, 0, 0]:
            errors.append(
                f"Deterministic sorted IDs expected `[2, 6, 0, 0, 0]` for 'CHEST pain', got {enc_short!r}."
            )
        if enc_short != enc_short_b:
            errors.append(
                "Token IDs depended on input text order! Sort unique tokens alphabetically in `.fit()`."
            )
        if enc_oov != [2, 1, 0, 0]:
            errors.append(
                f"Expected `[2, 1, 0, 0]` (OOV 'hypoxia' -> UNK_ID=1, padded with 0), got {enc_oov!r}."
            )
        if len(enc_trunc) != 3 or enc_trunc != [7, 2, 6]:
            errors.append(
                f"Expected truncation to length 3 `[7, 2, 6]`, got {enc_trunc!r}."
            )

    if errors:
        return CheckResult(
            "ch32_todo_05",
            False,
            "failed",
            "To-Do 32.5 — Check `ClinicalVocabTokenizer` Rules",
            "Define `PAD_ID = 0`, `UNK_ID = 1`, sort lowercased whitespace tokens in `.fit(texts)`, implement `__len__`, and pad/truncate to exact `max_len` in `.encode(text, max_len)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch32_todo_05",
        True,
        "passed",
        "To-Do 32.5 Passed — `ClinicalVocabTokenizer` Verified!",
        "Passed deterministic sorted vocabulary, case normalization, OOV -> UNK_ID=1 mapping, right PAD_ID=0 padding, truncation, and `len(tokenizer)`.",
        tip="Fit your tokenizer strictly on training notes only—just like Scikit-Learn's OneHotEncoder—so unseen validation jargon safely maps to `<UNK>` instead of crashing `nn.Embedding`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_todo_05",
        chapter="ch32",
        title="To-Do 32.5: Clinical Text Tokenizer (ClinicalVocabTokenizer)",
        book_section="Section 32.3.5 (Clinical Note Tokenization & Embeddings)",
        points=10,
        hints={
            1: "Set class constants `PAD_ID = 0` and `UNK_ID = 1`. In `__init__`, initialize `self.token_to_id = {'<PAD>': 0, '<UNK>': 1}`.",
            2: "In `fit(self, texts)`, gather `unique = sorted({w for t in texts for w in t.lower().split()})`, reset `self.token_to_id = {'<PAD>': 0, '<UNK>': 1}`, assign IDs `2, 3, ...`, and `return self`.",
            3: "class ClinicalVocabTokenizer:\n    PAD_ID = 0\n    UNK_ID = 1\n    def __init__(self):\n        self.token_to_id = {'<PAD>': 0, '<UNK>': 1}\n    def fit(self, texts):\n        words = sorted({w for t in texts for w in t.lower().split()})\n        self.token_to_id = {'<PAD>': 0, '<UNK>': 1}\n        for idx, w in enumerate(words, start=2):\n            self.token_to_id[w] = idx\n        return self\n    def encode(self, text, max_len):\n        ids = [self.token_to_id.get(w, self.UNK_ID) for w in text.lower().split()][:max_len]\n        return ids + [self.PAD_ID] * (max_len - len(ids))\n    def __len__(self):\n        return len(self.token_to_id)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 32.6: LoRALinear (nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_06(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    if not isinstance(cls, type):
        return CheckResult(
            "ch32_todo_06",
            False,
            "failed",
            "To-Do 32.6 — Pass the Class Name `LoRALinear`",
            f"Expected a class (type), got {type(cls).__name__} ({cls!r}).",
            details=["Pass the class itself: tutor.check('ch32_todo_06', LoRALinear)"],
            points_earned=0,
            points_possible=10,
        )

    if not issubclass(cls, nn.Module):
        return CheckResult(
            "ch32_todo_06",
            False,
            "failed",
            "To-Do 32.6 — Inherit From `nn.Module`",
            "Define `class LoRALinear(nn.Module):` and call `super().__init__()`.",
            points_earned=0,
            points_possible=10,
        )

    torch.manual_seed(23)
    base = nn.Linear(16, 12)
    try:
        lora = cls(base, rank=4, alpha=8.0)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_06",
            False,
            "failed",
            "To-Do 32.6 — Error Instantiating `LoRALinear`",
            f"LoRALinear(base, rank=4, alpha=8.0) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    errors: list[str] = []
    if base.weight.requires_grad:
        errors.append("`base_linear.weight.requires_grad` is still True! Freeze all `base_linear.parameters()`.")
    if base.bias is not None and base.bias.requires_grad:
        errors.append("`base_linear.bias.requires_grad` is still True! Freeze all `base_linear.parameters()`.")

    A = getattr(lora, "lora_A", None)
    B = getattr(lora, "lora_B", None)
    if not isinstance(A, nn.Parameter) or tuple(A.shape) != (4, 16) or not A.requires_grad:
        errors.append(
            f"Expected trainable `nn.Parameter` `self.lora_A` of shape (4, 16), got {type(A).__name__} shape={getattr(A, 'shape', None)}."
        )
    elif float(A.abs().max().item()) == 0.0:
        errors.append("`self.lora_A` is all zeros! Initialize `lora_A` with small random weights (e.g., `torch.randn(rank, in_features) * 0.02`).")

    if not isinstance(B, nn.Parameter) or tuple(B.shape) != (12, 4) or not B.requires_grad:
        errors.append(
            f"Expected trainable `nn.Parameter` `self.lora_B` of shape (12, 4), got {type(B).__name__} shape={getattr(B, 'shape', None)}."
        )
    elif float(B.abs().max().item()) > 1e-8:
        errors.append("`self.lora_B` must be initialized to ALL ZEROS (`torch.zeros(out_features, rank)`) so initial forward matches `base_linear(x)` identically.")

    x2 = torch.randn(3, 16)
    x3 = torch.randn(2, 5, 16)
    try:
        y2 = lora(x2)
        y3 = lora(x3)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch32_todo_06",
            False,
            "failed",
            "To-Do 32.6 — Runtime Error in `LoRALinear.forward`",
            f"Calling lora(x) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if y2 is None:
        return _none_return_trap("ch32_todo_06", "To-Do 32.6", "lora(x)", 10)

    if not torch.allclose(y2, base(x2), atol=1e-6):
        errors.append("Initial 2D output `lora(x)` does not match `base_linear(x)` identically.")
    if not torch.allclose(y3, base(x3), atol=1e-6):
        errors.append("Initial 3D `(B, T, d)` output `lora(x)` does not match `base_linear(x)`.")

    if not errors:
        # Perturb lora_B and test exact alpha/rank scaling
        with torch.no_grad():
            B.fill_(0.5)
        y2_pert = lora(x2)
        expected_pert = base(x2) + (8.0 / 4) * (x2 @ A.T @ B.T)
        if not torch.allclose(y2_pert, expected_pert, atol=1e-5):
            errors.append(
                "Perturbed LoRA output does not match `base_linear(x) + (alpha / rank) * (x @ lora_A.T @ lora_B.T)` "
                "(check your `alpha / rank` scaling factor!)."
            )

        # Verify backward pass freezes base and trains A & B
        loss = y2_pert.pow(2).mean()
        loss.backward()
        if base.weight.grad is not None:
            errors.append("`base_linear.weight.grad` should be None because base weights are frozen.")
        if A.grad is None or B.grad is None:
            errors.append("Both `lora_A.grad` and `lora_B.grad` must be populated after `.backward()`.")

    if errors:
        return CheckResult(
            "ch32_todo_06",
            False,
            "failed",
            "To-Do 32.6 — Check `LoRALinear` Low-Rank Adapter Implementation",
            "Freeze `base_linear`, define `lora_A` (rank, in_features) and zero-init `lora_B` (out_features, rank), and return `base_linear(x) + (alpha / rank) * (x @ lora_A.T @ lora_B.T)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch32_todo_06",
        True,
        "passed",
        "To-Do 32.6 Passed — `LoRALinear` Parameter-Efficient Adapter Verified!",
        "Passed base-weight freezing, zero-init output parity, exact `(alpha / rank) * (x @ A^T @ B^T)` scaling on 2D/3D tensors, and selective gradient flow.",
        tip="Zero-initializing `lora_B` guarantees LoRA starts at the exact pretrained foundation model solution—never destroying pretrained clinical knowledge on step 0!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_todo_06",
        chapter="ch32",
        title="To-Do 32.6: Low-Rank Adaptation Layer (LoRALinear)",
        book_section="Section 32.3.7 (Parameter-Efficient Fine-Tuning With LoRA)",
        points=10,
        hints={
            1: "Freeze `base_linear` with `for p in base_linear.parameters(): p.requires_grad = False`. Save `self.scaling = alpha / rank`.",
            2: "Create `self.lora_A = nn.Parameter(torch.randn(rank, base_linear.in_features) * 0.02)` and `self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))`.",
            3: "class LoRALinear(nn.Module):\n    def __init__(self, base_linear, rank=4, alpha=8.0):\n        super().__init__()\n        self.base_linear = base_linear\n        for p in self.base_linear.parameters():\n            p.requires_grad = False\n        self.scaling = float(alpha) / rank\n        self.lora_A = nn.Parameter(torch.randn(rank, base_linear.in_features) * 0.02)\n        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))\n    def forward(self, x):\n        return self.base_linear(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 32.1 (30 pts): verify_causal_attention_mask(seq_len)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch32_exam_01", "Exam Task 32.1", fn, "verify_causal_attention_mask", 30
    )
    if bad is not None:
        return bad

    expected_keys = ["upper_triangle_max", "row_sum_error", "is_causal"]
    errors: list[str] = []

    for seq_len in (4, 8, 10):
        try:
            got = fn(seq_len)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch32_exam_01",
                False,
                "failed",
                "Exam Task 32.1 — Runtime Error in `verify_causal_attention_mask`",
                f"verify_causal_attention_mask({seq_len}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch32_exam_01",
                "Exam Task 32.1",
                f"verify_causal_attention_mask({seq_len})",
                30,
            )

        key_err = _check_dict_keys(
            "ch32_exam_01", "Exam Task 32.1", got, expected_keys, 30
        )
        if key_err is not None:
            return key_err

        utm = got["upper_triangle_max"]
        rse = got["row_sum_error"]
        isc = got["is_causal"]

        if not _is_number(utm) or float(utm) < 0.0 or float(utm) >= 1e-6:
            errors.append(
                f"[seq_len={seq_len}] upper_triangle_max must be a float < 1e-6, got {utm!r}."
            )
        if not _is_number(rse) or float(rse) < 0.0 or float(rse) >= 1e-5:
            errors.append(
                f"[seq_len={seq_len}] row_sum_error must be a float < 1e-5, got {rse!r}."
            )
        if not _is_bool_like(isc) or not bool(isc):
            errors.append(f"[seq_len={seq_len}] is_causal must be True, got {isc!r}.")

    if errors:
        return CheckResult(
            "ch32_exam_01",
            False,
            "failed",
            "Exam Task 32.1 — Check Causal Mask Construction & Metrics",
            "Build `mask = torch.tril(torch.ones(1, seq_len, seq_len))`, run `scaled_dot_product_attention` on random `(1, seq_len, 8)` Q/K/V, and verify upper-triangle weights are 0 and rows sum to 1.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch32_exam_01",
        True,
        "passed",
        "Exam Task 32.1 Passed (30/30 pts) — Causal Autoregressive Mask Verified!",
        "Verified zero future-token leakage (`upper_triangle_max < 1e-6`) and row-stochastic normalization (`row_sum_error < 1e-5`) across seq_len = 4, 8, and 10.",
        points_earned=30,
        points_possible=30,
        tip="This exact lower-triangular mask is the core architectural distinction between bidirectional encoders (BERT/BioClinicalBERT) and autoregressive decoders (GPT/Gemini).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_exam_01",
        chapter="ch32",
        title="Exam 32.1: Causal Attention Mask Audit (verify_causal_attention_mask)",
        book_section="Chapter 32 Part VI Capstone Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Build `mask = torch.tril(torch.ones(1, seq_len, seq_len))` and generate random `(1, seq_len, 8)` tensors `Q, K, V`.",
            2: "Call `_, attn = scaled_dot_product_attention(Q, K, V, mask=mask)`. Extract strictly upper-triangular weights with `torch.triu(attn[0], diagonal=1).max().item()` and row-sum error with `(attn.sum(dim=-1) - 1.0).abs().max().item()`.",
            3: "def verify_causal_attention_mask(seq_len):\n    torch.manual_seed(0)\n    Q = torch.randn(1, seq_len, 8)\n    K = torch.randn(1, seq_len, 8)\n    V = torch.randn(1, seq_len, 8)\n    mask = torch.tril(torch.ones(1, seq_len, seq_len))\n    _, attn = scaled_dot_product_attention(Q, K, V, mask=mask)\n    upper_max = float(torch.triu(attn[0], diagonal=1).max().item()) if seq_len > 1 else 0.0\n    row_err = float((attn.sum(dim=-1) - 1.0).abs().max().item())\n    return {\n        'upper_triangle_max': upper_max,\n        'row_sum_error': row_err,\n        'is_causal': bool(upper_max < 1e-6 and row_err < 1e-5),\n    }",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 32.2 (35 pts): train_clinical_note_transformer(seed=42, epochs=25)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch32_exam_02", "Exam Task 32.2", fn, "train_clinical_note_transformer", 35
    )
    if bad is not None:
        return bad

    expected_keys = ["val_acc", "val_auc", "attention_shape"]
    errors: list[str] = []
    results_by_seed: list[dict[str, Any]] = []

    for seed in (42, 7):
        try:
            got = fn(seed=seed, epochs=25)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch32_exam_02",
                False,
                "failed",
                "Exam Task 32.2 — Runtime Error in `train_clinical_note_transformer`",
                f"train_clinical_note_transformer(seed={seed}, epochs=25) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch32_exam_02",
                "Exam Task 32.2",
                f"train_clinical_note_transformer(seed={seed}, epochs=25)",
                35,
            )

        key_err = _check_dict_keys(
            "ch32_exam_02", "Exam Task 32.2", got, expected_keys, 35
        )
        if key_err is not None:
            return key_err

        acc = got["val_acc"]
        auc = got["val_auc"]
        ashape = got["attention_shape"]

        if not _is_number(acc) or not (0.0 <= float(acc) <= 1.0):
            errors.append(f"[seed={seed}] val_acc must be a float in [0, 1], got {acc!r}.")
        elif float(acc) < 0.85:
            errors.append(f"[seed={seed}] val_acc={acc} is below 0.85; check optimizer/loss/masked pooling.")

        if not _is_number(auc) or not (0.0 <= float(auc) <= 1.0):
            errors.append(f"[seed={seed}] val_auc must be a float in [0, 1], got {auc!r}.")
        elif float(auc) < 0.90:
            errors.append(
                f"[seed={seed}] val_auc={auc} is below required threshold 0.90! "
                "Verify BCEWithLogitsLoss, masked mean pooling over non-PAD tokens, and Adam lr."
            )

        if (
            not isinstance(ashape, tuple)
            or len(ashape) != 4
            or not all(isinstance(d, int) and d > 0 for d in ashape)
        ):
            errors.append(
                f"[seed={seed}] attention_shape must be a 4-tuple of positive ints `(N_val, H, T, T)`, got {ashape!r}."
            )
        else:
            n_val, n_heads, t1, t2 = ashape
            if n_val != 45:
                errors.append(
                    f"[seed={seed}] Expected 25% of 180 notes = 45 validation notes in attention_shape[0], got {n_val}."
                )
            if n_heads < 2:
                errors.append(
                    f"[seed={seed}] Expected multi-head attention (`num_heads >= 2`), got {n_heads}."
                )
            if t1 != 10 or t2 != 10:
                errors.append(
                    f"[seed={seed}] Expected `max_len=10` so attention_shape[2:] == (10, 10), got {(t1, t2)}."
                )

        results_by_seed.append(got)

    if errors:
        return CheckResult(
            "ch32_exam_02",
            False,
            "failed",
            "Exam Task 32.2 — Check Clinical Note Transformer Training & Metrics",
            "Build a 180-note Routine vs. Urgent dataset, tokenize with `ClinicalVocabTokenizer(max_len=10)`, train Embedding + PE + TransformerEncoderBlock + MaskedMeanPool + Linear(d_model, 1) on a stratified 75/25 split, and achieve `val_auc >= 0.90`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    r42 = results_by_seed[0]
    return CheckResult(
        "ch32_exam_02",
        True,
        "passed",
        "Exam Task 32.2 Passed (35/35 pts) — Clinical Note Transformer Verified!",
        f"Achieved val_acc={r42['val_acc']:.4f}, val_auc={r42['val_auc']:.4f} (threshold >= 0.90) with attention_shape={r42['attention_shape']} across multiple seeds.",
        points_earned=35,
        points_possible=35,
        tip="Masked mean pooling averages only over real clinical tokens (excluding `<PAD>`), preventing short notes from being diluted by zero-padding vectors.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_exam_02",
        chapter="ch32",
        title="Exam 32.2: End-to-End Clinical Triage Transformer (train_clinical_note_transformer)",
        book_section="Chapter 32 Part VI Capstone Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Seed NumPy and PyTorch with `seed`, generate 180 labeled clinical triage notes (90 Routine=0, 90 Urgent=1), perform a stratified 75/25 `train_test_split`, and fit `ClinicalVocabTokenizer` on the training notes (`max_len=10`).",
            2: "Stack `nn.Embedding(vocab_size, d_model, padding_idx=0)`, `sinusoidal_positional_encoding(10, d_model)`, `TransformerEncoderBlock(d_model=16, num_heads=2, d_ff=32)`, masked mean pooling over non-PAD tokens, and `nn.Linear(16, 1)`. Train 25 epochs with `BCEWithLogitsLoss` and `Adam(lr=0.02)`.",
            3: "In eval mode on `X_val`, compute `logits, attn = model(X_val)`, probabilities `probs = torch.sigmoid(logits).squeeze(-1).numpy()`, `val_acc = round(float(((probs >= 0.5) == y_val).mean()), 4)`, `val_auc = round(float(roc_auc_score(y_val, probs)), 4)`, and `attention_shape = tuple(attn.shape)`.",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 32.3 (35 pts): audit_lora_parameter_efficiency(d_in=64, d_out=64, rank=4)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch32_exam_03", "Exam Task 32.3", fn, "audit_lora_parameter_efficiency", 35
    )
    if bad is not None:
        return bad

    expected_keys = [
        "base_params",
        "lora_trainable_params",
        "trainable_pct",
        "initial_parity_ok",
        "base_frozen_ok",
    ]
    errors: list[str] = []

    for d_in, d_out, rank in [(64, 64, 4), (32, 48, 2)]:
        try:
            got = fn(d_in=d_in, d_out=d_out, rank=rank)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch32_exam_03",
                False,
                "failed",
                "Exam Task 32.3 — Runtime Error in `audit_lora_parameter_efficiency`",
                f"audit_lora_parameter_efficiency({d_in}, {d_out}, {rank}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch32_exam_03",
                "Exam Task 32.3",
                f"audit_lora_parameter_efficiency({d_in}, {d_out}, {rank})",
                35,
            )

        key_err = _check_dict_keys(
            "ch32_exam_03", "Exam Task 32.3", got, expected_keys, 35
        )
        if key_err is not None:
            return key_err

        exp_base = d_in * d_out + d_out
        exp_lora = rank * (d_in + d_out)
        exp_pct_total = round(100.0 * exp_lora / (exp_base + exp_lora), 2)

        if not isinstance(got["base_params"], int) or isinstance(got["base_params"], bool) or got["base_params"] != exp_base:
            errors.append(
                f"[({d_in},{d_out},r={rank})] base_params: expected {exp_base} "
                f"(nn.Linear weight + bias), got {got['base_params']!r}."
            )
        if not isinstance(got["lora_trainable_params"], int) or isinstance(got["lora_trainable_params"], bool) or got["lora_trainable_params"] != exp_lora:
            errors.append(
                f"[({d_in},{d_out},r={rank})] lora_trainable_params: expected {exp_lora} "
                f"(rank * (d_in + d_out)), got {got['lora_trainable_params']!r}."
            )
        if not _is_number(got["trainable_pct"]) or not math.isclose(
            float(got["trainable_pct"]), exp_pct_total, abs_tol=0.05
        ):
            errors.append(
                f"[({d_in},{d_out},r={rank})] trainable_pct: expected {exp_pct_total} "
                f"(`round(100 * lora_trainable / (base_params + lora_trainable), 2)`), got {got['trainable_pct']!r}."
            )
        if not _is_bool_like(got["initial_parity_ok"]) or not bool(got["initial_parity_ok"]):
            errors.append(f"[({d_in},{d_out},r={rank})] initial_parity_ok must be True, got {got['initial_parity_ok']!r}.")
        if not _is_bool_like(got["base_frozen_ok"]) or not bool(got["base_frozen_ok"]):
            errors.append(f"[({d_in},{d_out},r={rank})] base_frozen_ok must be True, got {got['base_frozen_ok']!r}.")

    if errors:
        return CheckResult(
            "ch32_exam_03",
            False,
            "failed",
            "Exam Task 32.3 — Check LoRA Parameter Audit & Gradient Freeze",
            "Wrap `nn.Linear(d_in, d_out)` in `LoRALinear(..., rank=rank)`, verify initial forward parity (`max_diff < 1e-6`), run a backward pass confirming frozen base & active LoRA gradients, and report parameter efficiency.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch32_exam_03",
        True,
        "passed",
        "Exam Task 32.3 Passed (35/35 pts) — Part VI Capstone LoRA Audit Verified!",
        "Confirmed 64x64 linear layer adapts with only 512 trainable LoRA params (10.96% of 4,672 total), bit-exact step-0 parity, and frozen base gradients!",
        points_earned=35,
        points_possible=35,
        tip="Congratulations on completing the Part VI Capstone! At 7B–70B foundation model scale, rank-8 LoRA adapters routinely cut trainable parameters below 0.2% while matching full fine-tuning.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch32_exam_03",
        chapter="ch32",
        title="Exam 32.3: LoRA Parameter-Efficiency & Freeze Audit (audit_lora_parameter_efficiency)",
        book_section="Chapter 32 Part VI Capstone Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Create `base = nn.Linear(d_in, d_out)` and `lora = LoRALinear(base, rank=rank, alpha=2.0 * rank)`. Count `base_params = sum(p.numel() for p in base.parameters())` and `lora_trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)`.",
            2: "Check initial parity on a random tensor `x = torch.randn(4, d_in)` via `max_diff = (base(x) - lora(x)).abs().max().item() < 1e-6`. Then run `lora(x).pow(2).mean().backward()` and check `base.weight.grad is None and lora.lora_A.grad is not None and lora.lora_B.grad is not None`.",
            3: "def audit_lora_parameter_efficiency(d_in=64, d_out=64, rank=4):\n    torch.manual_seed(0)\n    base = nn.Linear(d_in, d_out)\n    lora = LoRALinear(base, rank=rank, alpha=2.0 * rank)\n    x = torch.randn(4, d_in)\n    with torch.no_grad():\n        parity_ok = bool((base(x) - lora(x)).abs().max().item() < 1e-6)\n    loss = lora(x).pow(2).mean()\n    loss.backward()\n    frozen_ok = bool(\n        base.weight.grad is None\n        and lora.lora_A.grad is not None\n        and lora.lora_B.grad is not None\n    )\n    base_params = sum(p.numel() for p in base.parameters())\n    lora_params = sum(p.numel() for p in lora.parameters() if p.requires_grad)\n    pct = round(100.0 * lora_params / (base_params + lora_params), 2)\n    return {\n        'base_params': int(base_params),\n        'lora_trainable_params': int(lora_params),\n        'trainable_pct': pct,\n        'initial_parity_ok': parity_ok,\n        'base_frozen_ok': frozen_ok,\n    }",
        },
        verifier=_verify_exam_03,
    )
)
