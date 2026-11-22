"""Chapter 30 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch30_todo_01..06) and the Chapter 30 Coding Exam
(ch30_exam_01..03) for Convolutional Neural Networks (CNNs) and Medical
Image Classification. Every verifier lazy-imports PyTorch, NumPy, and
Scikit-Learn inside the check function, runs 100% offline on deterministic
synthetic medical lesion patches, validates shapes, hooks, non-mutation,
and mathematical fidelity, while letting NotImplementedError propagate
cleanly for starter stubs.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_np_or_py_number(val: Any) -> bool:
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
            f"{title} — Pass the Function or Class Name Without Parentheses",
            (
                f"Expected a callable (`{expected_name}`), "
                f"but received {type(fn).__name__} ({fn!r})."
            ),
            details=[
                f"Pass the name itself: tutor.check('{ex_id}', {expected_name})",
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
        f"Calling {call_str} returned None instead of the computed CNN result.",
        details=[
            "Remember: print() only displays text; functions must end with an explicit `return ...`.",
            f"Check that {call_str} returns the required tuple, array, list, or dictionary.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Every vision helper should return clean Python/NumPy/PyTorch objects so downstream training and audit steps can compose it.",
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


def _make_ref_medical_cnn(num_classes: int = 1) -> Any:
    import torch.nn as nn

    class _RefMedicalPatchCNN(nn.Module):
        def __init__(self, n_cls: int = 1) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.fc = nn.Linear(16, n_cls)

        def forward(self, x: Any) -> Any:
            return self.fc(self.features(x))

    return _RefMedicalPatchCNN(num_classes)


def _make_ref_lesion_dataset(
    n_samples: int = 160,
    img_size: int = 16,
    seed: int = 42,
) -> tuple[Any, Any]:
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    n_neg = n_samples // 2
    n_pos = n_samples - n_neg
    labels = np.array([0] * n_neg + [1] * n_pos, dtype=np.float32)
    rng.shuffle(labels)

    yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing="ij")
    images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    for idx, lbl in enumerate(labels):
        base = rng.uniform(0.15, 0.30)
        patch = rng.normal(loc=base, scale=0.12, size=(img_size, img_size))
        if lbl == 1.0:
            cy = rng.integers(3, img_size - 3)
            cx = rng.integers(3, img_size - 3)
            amp = rng.uniform(0.95, 1.25)
            sigma = 1.5
            lesion = amp * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)
            )
            patch = patch + lesion
        images[idx, 0] = np.clip(patch, 0.0, 2.0).astype(np.float32)
    return torch.from_numpy(images), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 30.1: conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch30_todo_01", "To-Do 30.1", fn, "conv2d_output_shape", 10)
    if bad is not None:
        return bad

    valid_cases = [
        ((16, 16, 3), {}, (14, 14), "16x16 valid 3x3 conv"),
        ((16, 16, 3), {"stride": 1, "padding": 1}, (16, 16), "16x16 same-padding 3x3 conv"),
        ((32, 24, 5), {"stride": 2, "padding": 2}, (16, 12), "rectangular 32x24 stride-2 conv"),
        ((28, 28, 3), {"stride": 2, "padding": 0}, (13, 13), "even input odd kernel floor division"),
        ((224, 224, 7), {"stride": 2, "padding": 3}, (112, 112), "ResNet stem 7x7 stride-2 conv"),
    ]
    errors = []

    for args, kwargs, expected, desc in valid_cases:
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch30_todo_01",
                False,
                "failed",
                "To-Do 30.1 — Check Signature `conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0)`",
                f"Failed on {desc} with TypeError: {exc}",
                details=["Define `def conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0):`."],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch30_todo_01",
                False,
                "failed",
                "To-Do 30.1 — Runtime Error in `conv2d_output_shape`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch30_todo_01", "To-Do 30.1", "conv2d_output_shape(16, 16, 3)", 10)

        if not isinstance(got, tuple) or len(got) != 2 or not (_is_int_like(got[0]) and _is_int_like(got[1])):
            errors.append(f"[{desc}] Expected a 2-tuple of ints {expected}, got {got!r}.")
            continue

        if (int(got[0]), int(got[1])) != expected:
            errors.append(f"[{desc}] Expected {expected}, got {got!r}.")

    invalid_cases = [
        ((16, 16, 3), {"stride": 0, "padding": 0}, "stride=0 must raise ValueError"),
        ((16, 16, 3), {"stride": -1, "padding": 0}, "negative stride must raise ValueError"),
        ((2, 2, 5), {"stride": 1, "padding": 0}, "kernel larger than padded input must raise ValueError"),
        ((4, 2, 3), {"stride": 1, "padding": 0}, "width output <= 0 must raise ValueError"),
    ]
    for args, kwargs, desc in invalid_cases:
        try:
            got = fn(*args, **kwargs)
        except ValueError:
            continue
        except ZeroDivisionError:
            errors.append(
                f"[{desc}] Raised ZeroDivisionError instead of ValueError — check `if stride <= 0:` BEFORE dividing!"
            )
        except Exception as exc:
            errors.append(f"[{desc}] Expected ValueError, got {type(exc).__name__}: {exc}")
        else:
            errors.append(f"[{desc}] Expected ValueError, but returned {got!r}.")

    if errors:
        return CheckResult(
            "ch30_todo_01",
            False,
            "failed",
            "To-Do 30.1 — Check Conv2d Output Shape Formula & ValueError Guards",
            "Validate `stride > 0`, compute `(h_in + 2 * padding - kernel_size) // stride + 1` (and same for `w_in`), and raise `ValueError` if either output dimension is `<= 0`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch30_todo_01",
        True,
        "passed",
        "To-Do 30.1 Passed — Conv2d Output Shape Calculator Verified!",
        "Passed square, rectangular, strided, ResNet-stem, and invalid-input ValueError checks.",
        tip="Memorize `(H + 2P - K) // S + 1`: with odd `K` and `S=1`, setting `padding = K // 2` preserves spatial dimensions exactly!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_todo_01",
        chapter="ch30",
        title="To-Do 30.1: Conv2d Spatial Output Shape (conv2d_output_shape)",
        book_section="Section 30.3.2 (2D Convolution Output-Shape Arithmetic)",
        points=10,
        hints={
            1: "First guard `if stride <= 0: raise ValueError(...)`. Then apply `(dim + 2 * padding - kernel_size) // stride + 1` to both `h_in` and `w_in`.",
            2: "After computing `h_out` and `w_out`, check `if h_out <= 0 or w_out <= 0: raise ValueError(...)` and return `(int(h_out), int(w_out))`.",
            3: "def conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0):\n    if stride <= 0:\n        raise ValueError('stride must be positive')\n    h_out = (h_in + 2 * padding - kernel_size) // stride + 1\n    w_out = (w_in + 2 * padding - kernel_size) // stride + 1\n    if h_out <= 0 or w_out <= 0:\n        raise ValueError('output spatial dimensions must be positive')\n    return (int(h_out), int(w_out))",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 30.2: conv2d_single_channel_numpy(image_2d, kernel_2d, stride=1, padding=0)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np
    import torch
    import torch.nn.functional as F

    bad = _check_callable("ch30_todo_02", "To-Do 30.2", fn, "conv2d_single_channel_numpy", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(30)
    img1 = rng.normal(size=(5, 5)).astype(np.float32)
    ker1 = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32)

    img2 = rng.normal(size=(6, 6)).astype(np.float32)
    ker2 = np.array([[0.5, -1.2, 0.3], [2.1, 0.7, -0.4], [-0.9, 1.5, 0.2]], dtype=np.float32)

    img3 = rng.normal(size=(8, 7)).astype(np.float32)
    ker3 = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)

    cases = [
        (img1, ker1, {}, "5x5 Sobel kernel (stride=1, padding=0)"),
        (img2, ker2, {"stride": 1, "padding": 1}, "6x6 asymmetric kernel (stride=1, padding=1)"),
        (img3, ker3, {"stride": 2, "padding": 1}, "8x7 rectangular image 2x2 kernel (stride=2, padding=1)"),
    ]
    errors = []

    for img_raw, ker_raw, kwargs, desc in cases:
        img_copy = img_raw.copy()
        ker_copy = ker_raw.copy()
        stride = kwargs.get("stride", 1)
        padding = kwargs.get("padding", 0)
        try:
            got = fn(img_copy, ker_copy, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch30_todo_02",
                False,
                "failed",
                "To-Do 30.2 — Runtime Error in `conv2d_single_channel_numpy`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch30_todo_02", "To-Do 30.2", "conv2d_single_channel_numpy(image_2d, kernel_2d)", 10
            )

        if not np.array_equal(img_copy, img_raw) or not np.array_equal(ker_copy, ker_raw):
            errors.append(f"[{desc}] Mutated caller `image_2d` or `kernel_2d` in place!")

        if not isinstance(got, np.ndarray) or got.ndim != 2:
            errors.append(f"[{desc}] Expected a 2D NumPy array, got {type(got).__name__} (ndim={getattr(got, 'ndim', None)}).")
            continue

        t_img = torch.from_numpy(img_raw).unsqueeze(0).unsqueeze(0)
        t_ker = torch.from_numpy(ker_raw).unsqueeze(0).unsqueeze(0)
        exp = F.conv2d(t_img, t_ker, stride=stride, padding=padding).squeeze().numpy()

        if got.shape != exp.shape:
            errors.append(f"[{desc}] Shape mismatch: expected {exp.shape}, got {got.shape}.")
            continue

        if not np.allclose(got.astype(np.float32), exp, atol=1e-5):
            # Check if student flipped the kernel (mathematical convolution vs PyTorch cross-correlation)
            t_ker_flip = torch.from_numpy(ker_raw[::-1, ::-1].copy()).unsqueeze(0).unsqueeze(0)
            exp_flip = F.conv2d(t_img, t_ker_flip, stride=stride, padding=padding).squeeze().numpy()
            hint_extra = ""
            if np.allclose(got.astype(np.float32), exp_flip, atol=1e-5):
                hint_extra = " (You flipped the kernel! PyTorch `conv2d` computes cross-correlation WITHOUT flipping `kernel_2d`.)"
            max_diff = float(np.max(np.abs(got.astype(np.float32) - exp)))
            errors.append(f"[{desc}] Max absolute diff vs `F.conv2d` is {max_diff:.2e} (expected < 1e-5).{hint_extra}")

    if errors:
        return CheckResult(
            "ch30_todo_02",
            False,
            "failed",
            "To-Do 30.2 — Check Pure-NumPy 2D Cross-Correlation vs. `F.conv2d`",
            "Zero-pad `image_2d` with `np.pad(..., constant_values=0.0)` when `padding > 0`, slide over `(i*stride, j*stride)`, and sum `patch * kernel_2d` without flipping the kernel.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch30_todo_02",
        True,
        "passed",
        "To-Do 30.2 Passed — Pure-NumPy 2D Cross-Correlation Matches PyTorch `F.conv2d`!",
        "Matched `torch.nn.functional.conv2d` to < 1e-5 across valid, padded, asymmetric-kernel, and strided rectangular cases without mutating inputs.",
        tip="Deep learning libraries call this operation `conv2d`, though mathematically it is 2D cross-correlation (no 180-degree kernel flip) because weights are learned anyway.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_todo_02",
        chapter="ch30",
        title="To-Do 30.2: Pure-NumPy 2D Cross-Correlation (conv2d_single_channel_numpy)",
        book_section="Section 30.3.1 (2D Convolution from Scratch vs. PyTorch F.conv2d)",
        points=10,
        hints={
            1: "Convert inputs to float arrays (`np.asarray(..., dtype=np.float32)`). If `padding > 0`, zero-pad all four sides with `np.pad(img, ((padding, padding), (padding, padding)), mode='constant')`.",
            2: "Compute `h_out = (h_pad - kh) // stride + 1` and `w_out = (w_pad - kw) // stride + 1`. Loop `i` in `range(h_out)` and `j` in `range(w_out)`, slicing `patch = img[i*stride : i*stride + kh, j*stride : j*stride + kw]`.",
            3: "def conv2d_single_channel_numpy(image_2d, kernel_2d, stride=1, padding=0):\n    img = np.asarray(image_2d, dtype=np.float32)\n    ker = np.asarray(kernel_2d, dtype=np.float32)\n    if padding > 0:\n        img = np.pad(img, ((padding, padding), (padding, padding)), mode='constant')\n    kh, kw = ker.shape\n    h_out = (img.shape[0] - kh) // stride + 1\n    w_out = (img.shape[1] - kw) // stride + 1\n    out = np.zeros((h_out, w_out), dtype=np.float32)\n    for i in range(h_out):\n        for j in range(w_out):\n            patch = img[i * stride : i * stride + kh, j * stride : j * stride + kw]\n            out[i, j] = np.sum(patch * ker)\n    return out",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 30.3: MedicalPatchCNN(nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_03(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch30_todo_03", "To-Do 30.3", cls, "MedicalPatchCNN", 10)
    if bad is not None:
        return bad

    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        return CheckResult(
            "ch30_todo_03",
            False,
            "failed",
            "To-Do 30.3 — Pass the `MedicalPatchCNN` Class (`nn.Module` Subclass)",
            f"Expected a subclass of `torch.nn.Module`, got {cls!r}.",
            details=[
                "Define `class MedicalPatchCNN(nn.Module):` and pass the class name: `tutor.check('ch30_todo_03', MedicalPatchCNN)`."
            ],
            points_earned=0,
            points_possible=10,
        )

    errors = []
    try:
        model1 = cls()
        model3 = cls(num_classes=3)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch30_todo_03",
            False,
            "failed",
            "To-Do 30.3 — Error Instantiating `MedicalPatchCNN`",
            f"Failed during `MedicalPatchCNN()` or `MedicalPatchCNN(num_classes=3)` with {type(exc).__name__}: {exc}",
            details=["Make sure `__init__(self, num_classes=1)` calls `super().__init__()`."],
            points_earned=0,
            points_possible=10,
        )

    convs = [m for m in model1.modules() if isinstance(m, nn.Conv2d)]
    maxpools = [m for m in model1.modules() if isinstance(m, nn.MaxPool2d)]
    adapts = [m for m in model1.modules() if isinstance(m, nn.AdaptiveAvgPool2d)]

    if len(convs) < 2:
        errors.append(f"Expected 2 `nn.Conv2d` layers (`1->8` and `8->16`), found {len(convs)}.")
    else:
        c1, c2 = convs[0], convs[1]
        if (c1.in_channels, c1.out_channels, c1.kernel_size, c1.padding) != (1, 8, (3, 3), (1, 1)):
            errors.append(
                f"First Conv2d should be `Conv2d(1, 8, kernel_size=3, padding=1)`, "
                f"got in={c1.in_channels}, out={c1.out_channels}, k={c1.kernel_size}, pad={c1.padding}."
            )
        if (c2.in_channels, c2.out_channels, c2.kernel_size, c2.padding) != (8, 16, (3, 3), (1, 1)):
            errors.append(
                f"Second Conv2d should be `Conv2d(8, 16, kernel_size=3, padding=1)`, "
                f"got in={c2.in_channels}, out={c2.out_channels}, k={c2.kernel_size}, pad={c2.padding}."
            )

    if not maxpools:
        errors.append("Missing `nn.MaxPool2d(2)` spatial downsampling layer.")
    if not adapts:
        errors.append("Missing `nn.AdaptiveAvgPool2d(1)` global average pooling layer.")

    fc = getattr(model1, "fc", None)
    if not isinstance(fc, nn.Linear) or (fc.in_features, fc.out_features) != (16, 1):
        errors.append(
            "Classification head must be named `self.fc = nn.Linear(16, num_classes)` "
            "(required so transfer-learning freezing on `head_name='fc'` works cleanly!)."
        )

    param_names = [n for n, _ in model1.named_parameters()]
    if not any(n.startswith("fc.") for n in param_names):
        errors.append("No parameters starting with `'fc.'` found in `model.named_parameters()`.")

    try:
        out_16 = model1(torch.randn(4, 1, 16, 16))
        out_24 = model1(torch.randn(2, 1, 24, 24))
        out_3c = model3(torch.randn(3, 1, 16, 16))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch30_todo_03",
            False,
            "failed",
            "To-Do 30.3 — Forward Pass Error in `MedicalPatchCNN`",
            f"Forward pass failed with {type(exc).__name__}: {exc}",
            details=[
                "Check layer order: Conv2d(1,8,3,padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(8,16,3,padding=1) -> ReLU -> AdaptiveAvgPool2d(1) -> Flatten -> self.fc."
            ],
            points_earned=0,
            points_possible=10,
        )

    if tuple(out_16.shape) != (4, 1):
        errors.append(f"For input (4, 1, 16, 16) with num_classes=1, expected shape (4, 1), got {tuple(out_16.shape)}.")
    if tuple(out_24.shape) != (2, 1):
        errors.append(
            f"For variable-size input (2, 1, 24, 24), expected shape (2, 1), got {tuple(out_24.shape)} "
            "(Ensure `AdaptiveAvgPool2d(1)` runs before `Flatten`!)."
        )
    if tuple(out_3c.shape) != (3, 3):
        errors.append(f"For input (3, 1, 16, 16) with num_classes=3, expected shape (3, 3), got {tuple(out_3c.shape)}.")

    # Verify gradient flow
    try:
        model1.zero_grad()
        out_16.sum().backward()
        if any(p.grad is None for p in model1.parameters()):
            errors.append("Some parameters received `grad=None` after backward pass — ensure every layer is used in `forward()`.")
    except Exception as exc:
        errors.append(f"Backward pass failed: {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch30_todo_03",
            False,
            "failed",
            "To-Do 30.3 — Check `MedicalPatchCNN` Layers, `self.fc` Head & Forward Shapes",
            "Build `Conv2d(1,8,3,padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(8,16,3,padding=1) -> ReLU -> AdaptiveAvgPool2d(1) -> Flatten -> self.fc` (`Linear(16, num_classes)`).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch30_todo_03",
        True,
        "passed",
        "To-Do 30.3 Passed — `MedicalPatchCNN` Architecture Verified!",
        "Verified 1->8->16 convolutional hierarchy, resolution-agnostic `AdaptiveAvgPool2d(1)`, `self.fc` head naming, and clean gradient flow.",
        tip="Global average pooling (`AdaptiveAvgPool2d(1)`) eliminates brittle hardcoded flatten sizes, letting one CNN score 16x16, 24x24, or 64x64 patches seamlessly!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_todo_03",
        chapter="ch30",
        title="To-Do 30.3: Compact Medical Patch CNN (MedicalPatchCNN)",
        book_section="Section 30.3.3 (Pooling, Adaptive Global Average Pooling & MedicalPatchCNN)",
        points=10,
        hints={
            1: "Subclass `nn.Module`, call `super().__init__()`, stack the convolutional feature extractor in `self.features = nn.Sequential(...)`, and name the final linear classifier `self.fc = nn.Linear(16, num_classes)`.",
            2: "Inside `self.features`: `Conv2d(1, 8, 3, padding=1)`, `ReLU()`, `MaxPool2d(2)`, `Conv2d(8, 16, 3, padding=1)`, `ReLU()`, `AdaptiveAvgPool2d(1)`, `Flatten()`. In `forward(self, x)`, return `self.fc(self.features(x))`.",
            3: "class MedicalPatchCNN(nn.Module):\n    def __init__(self, num_classes=1):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(1, 8, kernel_size=3, padding=1),\n            nn.ReLU(),\n            nn.MaxPool2d(kernel_size=2),\n            nn.Conv2d(8, 16, kernel_size=3, padding=1),\n            nn.ReLU(),\n            nn.AdaptiveAvgPool2d(1),\n            nn.Flatten(),\n        )\n        self.fc = nn.Linear(16, ___)\n    def forward(self, x):\n        return self.fc(self.features(x))",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 30.4: ResidualConvBlock(nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_04(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch30_todo_04", "To-Do 30.4", cls, "ResidualConvBlock", 10)
    if bad is not None:
        return bad

    if not (inspect.isclass(cls) and issubclass(cls, nn.Module)):
        return CheckResult(
            "ch30_todo_04",
            False,
            "failed",
            "To-Do 30.4 — Pass the `ResidualConvBlock` Class (`nn.Module` Subclass)",
            f"Expected a subclass of `torch.nn.Module`, got {cls!r}.",
            points_earned=0,
            points_possible=10,
        )

    errors = []
    try:
        blk8 = cls(8)
        blk16 = cls(16)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch30_todo_04",
            False,
            "failed",
            "To-Do 30.4 — Error Instantiating `ResidualConvBlock(channels)`",
            f"Failed during `ResidualConvBlock(8)` with {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    convs = [m for m in blk8.modules() if isinstance(m, nn.Conv2d)]
    bns = [m for m in blk8.modules() if isinstance(m, nn.BatchNorm2d)]
    if len(convs) != 2:
        errors.append(f"Expected 2 `nn.Conv2d(c, c, 3, padding=1)` layers, found {len(convs)}.")
    else:
        for i, c in enumerate(convs, 1):
            if (c.in_channels, c.out_channels, c.kernel_size, c.padding) != (8, 8, (3, 3), (1, 1)):
                errors.append(
                    f"Conv #{i} should be `Conv2d(8, 8, kernel_size=3, padding=1)`, "
                    f"got ({c.in_channels}, {c.out_channels}, k={c.kernel_size}, pad={c.padding})."
                )
    if len(bns) != 2:
        errors.append(f"Expected 2 `nn.BatchNorm2d(c)` layers, found {len(bns)}.")
    else:
        for i, b in enumerate(bns, 1):
            if b.num_features != 8:
                errors.append(f"BatchNorm2d #{i} should have `num_features=8`, got {b.num_features}.")

    try:
        x8 = torch.randn(3, 8, 16, 16)
        y8 = blk8(x8)
        x16 = torch.randn(2, 16, 10, 12)
        y16 = blk16(x16)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch30_todo_04",
            False,
            "failed",
            "To-Do 30.4 — Forward Pass Error in `ResidualConvBlock`",
            f"Forward pass failed with {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if tuple(y8.shape) != (3, 8, 16, 16):
        errors.append(f"Expected exact shape (3, 8, 16, 16), got {tuple(y8.shape)}.")
    if tuple(y16.shape) != (2, 16, 10, 12):
        errors.append(f"Expected exact shape (2, 16, 10, 12), got {tuple(y16.shape)}.")

    # Skip-connection & post-addition ReLU test: zero all BatchNorm weights/biases so F(x) == 0
    blk_test = cls(4)
    blk_test.eval()
    with torch.no_grad():
        for m in blk_test.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.zero_()
                m.bias.zero_()
        pos_in = torch.full((2, 4, 6, 6), 2.5)
        pos_out = blk_test(pos_in)
        if not torch.allclose(pos_out, pos_in, atol=1e-5):
            errors.append(
                "Skip connection check failed! When the residual branch outputs zeros, `F.relu(0 + x)` "
                "must equal `x` for positive inputs (`2.5`). Did you forget `+ x`?"
            )
        neg_in = torch.full((2, 4, 6, 6), -2.5)
        neg_out = blk_test(neg_in)
        if not torch.allclose(neg_out, torch.zeros_like(neg_in), atol=1e-5):
            errors.append(
                "Final ReLU check failed! Expected `F.relu(out + x)` to zero out negative values (`-2.5 -> 0.0`)."
            )

    if errors:
        return CheckResult(
            "ch30_todo_04",
            False,
            "failed",
            "To-Do 30.4 — Check Residual Block Layers, Shape Preservation & Skip Connection",
            "Apply `Conv2d(c,c,3,padding=1) -> BatchNorm2d(c) -> ReLU -> Conv2d(c,c,3,padding=1) -> BatchNorm2d(c)` and return `F.relu(out + x)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch30_todo_04",
        True,
        "passed",
        "To-Do 30.4 Passed — `ResidualConvBlock` & Identity Skip Highway Verified!",
        "Confirmed 2x Conv2d+BatchNorm2d layers, exact (B, C, H, W) preservation, identity skip path (`out + x`), and post-addition ReLU.",
        tip="When `F(x)` is initialized near zero, `x + F(x)` starts as an identity highway—letting gradients flow unimpeded through 50+ layers!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_todo_04",
        chapter="ch30",
        title="To-Do 30.4: ResNet-Style Residual Conv Block (ResidualConvBlock)",
        book_section="Section 30.3.4 (Residual Skip Connections & Vanishing Gradients)",
        points=10,
        hints={
            1: "Create two `nn.Conv2d(channels, channels, kernel_size=3, padding=1)` layers and two `nn.BatchNorm2d(channels)` layers in `__init__`.",
            2: "In `forward(self, x)`, compute `out = F.relu(self.bn1(self.conv1(x)))`, then `out = self.bn2(self.conv2(out))`, and finally return `F.relu(out + x)`.",
            3: "class ResidualConvBlock(nn.Module):\n    def __init__(self, channels):\n        super().__init__()\n        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n        self.bn1 = nn.BatchNorm2d(channels)\n        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n        self.bn2 = nn.BatchNorm2d(channels)\n    def forward(self, x):\n        out = F.relu(self.bn1(self.conv1(x)))\n        out = self.bn2(self.conv2(out))\n        return F.relu(out + ___)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 30.5: freeze_backbone_for_transfer(model, head_name="fc")
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import torch.nn as nn

    bad = _check_callable("ch30_todo_05", "To-Do 30.5", fn, "freeze_backbone_for_transfer", 10)
    if bad is not None:
        return bad

    class _CustomHeadNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Conv2d(1, 4, kernel_size=3, padding=1)
            self.classifier = nn.Linear(4, 2)

        def forward(self, x: Any) -> Any:
            return self.classifier(self.stem(x).mean(dim=(2, 3)))

    m1 = _make_ref_medical_cnn(num_classes=1)
    # Start m1 with all params frozen to verify head params get unfrozen too!
    for p in m1.parameters():
        p.requires_grad = False

    m2 = _CustomHeadNet()

    cases = [
        (m1, {}, "fc", 17, 1248, "MedicalPatchCNN with default head_name='fc'"),
        (m2, {"head_name": "classifier"}, "classifier", 10, 40, "custom network with head_name='classifier'"),
    ]
    expected_keys = ["trainable_params", "frozen_params"]
    errors = []

    for model, kwargs, head_name, exp_train, exp_frozen, desc in cases:
        try:
            got = fn(model, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch30_todo_05",
                False,
                "failed",
                "To-Do 30.5 — Runtime Error in `freeze_backbone_for_transfer`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch30_todo_05", "To-Do 30.5", "freeze_backbone_for_transfer(model)", 10
            )

        key_err = _check_dict_keys("ch30_todo_05", "To-Do 30.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        for name, param in model.named_parameters():
            expected_rg = name.startswith(f"{head_name}.")
            if param.requires_grad != expected_rg:
                errors.append(
                    f"[{desc}] Parameter `{name}` has requires_grad={param.requires_grad}, expected {expected_rg}."
                )

        if not _is_int_like(got["trainable_params"]) or int(got["trainable_params"]) != exp_train:
            errors.append(
                f"[{desc}] `trainable_params`: expected {exp_train} (scalar elements via `p.numel()`), got {got['trainable_params']!r}."
            )
        if not _is_int_like(got["frozen_params"]) or int(got["frozen_params"]) != exp_frozen:
            errors.append(
                f"[{desc}] `frozen_params`: expected {exp_frozen} (scalar elements via `p.numel()`), got {got['frozen_params']!r}."
            )

    if errors:
        return CheckResult(
            "ch30_todo_05",
            False,
            "failed",
            "To-Do 30.5 — Check Transfer-Learning Freezing & Parameter Element Counts",
            "Iterate `for name, p in model.named_parameters():`, set `p.requires_grad = name.startswith(f'{head_name}.')`, and sum `p.numel()` for trainable vs. frozen parameters.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch30_todo_05",
        True,
        "passed",
        "To-Do 30.5 Passed — Transfer-Learning Backbone Freezing Verified!",
        "Toggled `requires_grad` cleanly across default (`fc`) and custom (`classifier`) heads and tallied exact `p.numel()` parameter counts.",
        tip="In medical imaging transfer learning, freezing the convolutional backbone and training only `fc` prevents small hospital cohorts (< 500 scans) from overfitting!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_todo_05",
        chapter="ch30",
        title="To-Do 30.5: Transfer-Learning Backbone Freezer (freeze_backbone_for_transfer)",
        book_section="Section 30.3.6 (Transfer Learning & Backbone Freezing)",
        points=10,
        hints={
            1: "Loop over `for name, param in model.named_parameters():` and set `param.requires_grad = name.startswith(f'{head_name}.')`.",
            2: "Count scalar elements with `param.numel()`: add to `trainable_params` when `param.requires_grad` is True, otherwise add to `frozen_params`.",
            3: "def freeze_backbone_for_transfer(model, head_name='fc'):\n    prefix = f'{head_name}.'\n    trainable, frozen = 0, 0\n    for name, p in model.named_parameters():\n        p.requires_grad = name.startswith(prefix)\n        if p.requires_grad:\n            trainable += p.numel()\n        else:\n            frozen += p.numel()\n    return {'trainable_params': int(trainable), 'frozen_params': int(frozen)}",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 30.6: generate_synthetic_lesion_dataset(n_samples=160, img_size=16, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np
    import torch

    bad = _check_callable("ch30_todo_06", "To-Do 30.6", fn, "generate_synthetic_lesion_dataset", 10)
    if bad is not None:
        return bad

    errors = []
    cases = [
        ({}, 160, 16, 42, "default (n_samples=160, img_size=16, seed=42)"),
        ({"n_samples": 60, "img_size": 20, "seed": 7}, 60, 20, 7, "custom (n_samples=60, img_size=20, seed=7)"),
    ]

    for kwargs, n_s, sz, seed, desc in cases:
        try:
            got = fn(**kwargs)
            got_repeat = fn(**kwargs)
            got_diff = fn(n_samples=n_s, img_size=sz, seed=seed + 100)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch30_todo_06",
                False,
                "failed",
                "To-Do 30.6 — Runtime Error in `generate_synthetic_lesion_dataset`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch30_todo_06", "To-Do 30.6", "generate_synthetic_lesion_dataset()", 10
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch30_todo_06",
                False,
                "failed",
                "To-Do 30.6 — Return a 2-Tuple `(X_tensor, y_tensor)`",
                f"[{desc}] Expected a 2-tuple of PyTorch tensors, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        X, y = got
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            errors.append(f"[{desc}] Both X and y must be `torch.Tensor`, got ({type(X).__name__}, {type(y).__name__}).")
            continue

        if X.dtype != torch.float32 or y.dtype != torch.float32:
            errors.append(f"[{desc}] Both tensors must have dtype `torch.float32`, got ({X.dtype}, {y.dtype}).")

        if tuple(X.shape) != (n_s, 1, sz, sz):
            errors.append(f"[{desc}] Expected `X.shape == {(n_s, 1, sz, sz)}`, got {tuple(X.shape)}.")
            continue
        if tuple(y.shape) != (n_s,):
            errors.append(f"[{desc}] Expected `y.shape == {(n_s,)}`, got {tuple(y.shape)}.")
            continue

        if not torch.isfinite(X).all() or not torch.isfinite(y).all():
            errors.append(f"[{desc}] Generated tensors contain NaN or Inf values.")
            continue

        unique_y = set(torch.unique(y).tolist())
        if unique_y != {0.0, 1.0}:
            errors.append(f"[{desc}] Labels must contain only binary classes {{0.0, 1.0}}, got {unique_y}.")
            continue

        pos_rate = float(y.mean().item())
        if not (0.45 <= pos_rate <= 0.55):
            errors.append(f"[{desc}] Classes should be balanced (~50/50), got positive rate {pos_rate:.3f}.")

        X_rep, y_rep = got_repeat
        if not (torch.equal(X, X_rep) and torch.equal(y, y_rep)):
            errors.append(f"[{desc}] Not deterministic! Two calls with `seed={seed}` produced different tensors.")

        X_dif, _ = got_diff
        if torch.equal(X, X_dif):
            errors.append(f"[{desc}] Different seeds (`{seed}` vs `{seed + 100}`) produced identical tensors.")

        pos_imgs = X[y == 1.0, 0]
        neg_imgs = X[y == 0.0, 0]
        pos_peak = float(pos_imgs.amax(dim=(1, 2)).mean().item())
        neg_peak = float(neg_imgs.amax(dim=(1, 2)).mean().item())
        if pos_peak <= neg_peak + 0.25:
            errors.append(
                f"[{desc}] Positive lesion patches should have a clearly brighter focal peak than healthy patches "
                f"(mean max intensity: pos={pos_peak:.3f} vs neg={neg_peak:.3f})."
            )

        # Verify lesions appear at varied spatial coordinates (translation equivariance!)
        peak_coords = set()
        for patch in pos_imgs.numpy():
            r, c = np.unravel_index(int(np.argmax(patch)), patch.shape)
            peak_coords.add((int(r), int(c)))
        if len(peak_coords) < 5:
            errors.append(
                f"[{desc}] Positive lesion peaks occurred at only {len(peak_coords)} unique coordinate(s) — "
                "randomize lesion center `(cy, cx)` across interior locations!"
            )

    if errors:
        return CheckResult(
            "ch30_todo_06",
            False,
            "failed",
            "To-Do 30.6 — Check Synthetic Medical Lesion Dataset Shapes, Seed & Signal",
            "Use `np.random.default_rng(seed)` to generate balanced `(X[n, 1, H, W], y[n])` float32 tensors with noisy background (`0`) vs. background + randomized interior Gaussian nodule (`1`).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch30_todo_06",
        True,
        "passed",
        "To-Do 30.6 Passed — Deterministic Synthetic Medical Lesion Generator Verified!",
        "Confirmed NCHW float32 shapes, balanced classes, strict seed reproducibility, strong focal lesion contrast, and randomized interior locations.",
        tip="Randomizing lesion coordinates across the patch tests true CNN translation equivariance—a flat linear classifier struggles while shared 3x3 filters find nodules anywhere!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_todo_06",
        chapter="ch30",
        title="To-Do 30.6: Synthetic Medical Lesion Patch Generator (generate_synthetic_lesion_dataset)",
        book_section="Section 30.3.5 (Synthetic Medical Lesion Dataset & CNN Training)",
        points=10,
        hints={
            1: "Create `rng = np.random.default_rng(seed)`, construct balanced binary labels (`n_samples // 2` zeros and remaining ones), shuffle them, and allocate `images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)`.",
            2: "Precompute a coordinate grid `yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')`. For each patch add background noise; when `lbl == 1`, sample random interior `(cy, cx)` in `[3, img_size - 3)` and add a 2D Gaussian nodule `amp * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sigma**2))`.",
            3: "def generate_synthetic_lesion_dataset(n_samples=160, img_size=16, seed=42):\n    rng = np.random.default_rng(seed)\n    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2), dtype=np.float32)\n    rng.shuffle(labels)\n    yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')\n    images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)\n    for i, lbl in enumerate(labels):\n        patch = rng.normal(rng.uniform(0.15, 0.30), 0.12, size=(img_size, img_size))\n        if lbl == 1.0:\n            cy, cx = rng.integers(3, img_size - 3), rng.integers(3, img_size - 3)\n            patch += rng.uniform(0.95, 1.25) * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2.0 * 1.5**2))\n        images[i, 0] = np.clip(patch, 0.0, 2.0)\n    return torch.from_numpy(images), torch.from_numpy(labels)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 30.1 (30 pts): audit_cnn_feature_map_shapes(model, input_shape)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch30_exam_01", "Exam Task 30.1", fn, "audit_cnn_feature_map_shapes", 30)
    if bad is not None:
        return bad

    class _MiniAuditNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1)
            self.head = nn.Linear(12, 2)

        def forward(self, x: Any) -> Any:
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.mean(dim=(2, 3))
            return self.head(x)

    m_cnn = _make_ref_medical_cnn(num_classes=1)
    m_mini = _MiniAuditNet()

    cases = [
        (
            m_cnn,
            (4, 1, 16, 16),
            [
                ("features.0", (4, 8, 16, 16)),
                ("features.3", (4, 16, 8, 8)),
                ("fc", (4, 1)),
            ],
            "MedicalPatchCNN on (4, 1, 16, 16)",
        ),
        (
            m_cnn,
            (2, 1, 24, 24),
            [
                ("features.0", (2, 8, 24, 24)),
                ("features.3", (2, 16, 12, 12)),
                ("fc", (2, 1)),
            ],
            "MedicalPatchCNN on (2, 1, 24, 24)",
        ),
        (
            m_mini,
            (3, 1, 16, 16),
            [
                ("conv1", (3, 6, 16, 16)),
                ("conv2", (3, 12, 8, 8)),
                ("head", (3, 2)),
            ],
            "Custom 2-Conv + Linear net on (3, 1, 16, 16)",
        ),
    ]
    errors = []

    for model, in_shape, expected, desc in cases:
        try:
            got = fn(model, in_shape)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch30_exam_01",
                False,
                "failed",
                "Exam Task 30.1 — Runtime Error in `audit_cnn_feature_map_shapes`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch30_exam_01", "Exam Task 30.1", "audit_cnn_feature_map_shapes(model, input_shape)", 30
            )

        if not isinstance(got, list):
            errors.append(f"[{desc}] Expected a list of `(module_name, shape_tuple)` pairs, got {type(got).__name__}.")
            continue

        # Check hook cleanup immediately!
        leftover_hooks = sum(len(m._forward_hooks) for m in model.modules())
        if leftover_hooks > 0:
            errors.append(
                f"[{desc}] Found {leftover_hooks} leftover forward hook(s) on `model`! "
                "Always call `handle.remove()` on every registered hook handle."
            )

        norm_got = []
        bad_entry = False
        for item in got:
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str) or not isinstance(item[1], tuple):
                errors.append(f"[{desc}] Each entry must be `(str_name, tuple_shape)`, got {item!r}.")
                bad_entry = True
                break
            norm_got.append((item[0], tuple(int(d) for d in item[1])))

        if not bad_entry and norm_got != expected:
            errors.append(f"[{desc}] Expected {expected}, got {norm_got}.")

    if errors:
        return CheckResult(
            "ch30_exam_01",
            False,
            "failed",
            "Exam Task 30.1 — Check Forward Hook Registration, Shape Capture & Cleanup",
            "Register forward hooks on every `nn.Conv2d` and `nn.Linear` via `model.named_modules()`, run `model(torch.zeros(input_shape))` inside `model.eval()` and `torch.no_grad()`, remove all hooks cleanly, and return `[(name, tuple(out.shape)), ...]`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch30_exam_01",
        True,
        "passed",
        "Exam Task 30.1 Passed (30/30 pts) — CNN Feature-Map Shape Auditor Verified!",
        "Captured exact Conv2d and Linear output shapes across multiple architectures and resolutions with zero leaked forward hooks.",
        points_earned=30,
        points_possible=30,
        tip="Always remove PyTorch hooks in a `try ... finally` block so failed forward passes never leak hooks into long training runs!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_exam_01",
        chapter="ch30",
        title="Exam 30.1: Forward-Hook Feature-Map Shape Auditor (audit_cnn_feature_map_shapes)",
        book_section="Chapter 30 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Iterate `for name, module in model.named_modules():`. Whenever `isinstance(module, (nn.Conv2d, nn.Linear))`, attach a forward hook capturing `(name, tuple(output.shape))` and save the hook handle in a list.",
            2: "Switch `model.eval()`, run `with torch.no_grad(): model(torch.zeros(input_shape))` inside a `try:` block, and remove every handle (`h.remove()`) inside `finally:`.",
            3: "def audit_cnn_feature_map_shapes(model, input_shape):\n    shapes, handles = [], []\n    def _make_hook(layer_name):\n        def _hook(_mod, _inp, out):\n            shapes.append((layer_name, tuple(out.shape)))\n        return _hook\n    for name, mod in model.named_modules():\n        if isinstance(mod, (nn.Conv2d, nn.Linear)):\n            handles.append(mod.register_forward_hook(_make_hook(name)))\n    try:\n        model.eval()\n        with torch.no_grad():\n            model(torch.zeros(input_shape))\n    finally:\n        for h in handles:\n            h.remove()\n    return shapes",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 30.2 (35 pts): train_medical_patch_cnn(n_samples=200, epochs=15, seed=42)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch30_exam_02", "Exam Task 30.2", fn, "train_medical_patch_cnn", 35)
    if bad is not None:
        return bad

    expected_keys = ["val_acc", "val_auc", "final_train_loss"]
    errors = []

    cases = [
        ({"n_samples": 200, "epochs": 15, "seed": 42}, "default (n_samples=200, epochs=15, seed=42)"),
        ({"n_samples": 160, "epochs": 15, "seed": 7}, "alternate seed (n_samples=160, epochs=15, seed=7)"),
    ]

    first_res = None
    for kwargs, desc in cases:
        try:
            got = fn(**kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch30_exam_02",
                False,
                "failed",
                "Exam Task 30.2 — Check Signature `train_medical_patch_cnn(n_samples=200, epochs=15, seed=42)`",
                f"Failed on {desc} with TypeError: {exc}",
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch30_exam_02",
                False,
                "failed",
                "Exam Task 30.2 — Runtime Error in `train_medical_patch_cnn`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch30_exam_02", "Exam Task 30.2", "train_medical_patch_cnn()", 35
            )

        key_err = _check_dict_keys("ch30_exam_02", "Exam Task 30.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        for k in expected_keys:
            if not _is_np_or_py_number(got[k]) or not math.isfinite(float(got[k])):
                errors.append(f"[{desc}] `{k}` must be a finite float, got {got[k]!r}.")

        if errors:
            continue

        val_acc = float(got["val_acc"])
        val_auc = float(got["val_auc"])
        final_loss = float(got["final_train_loss"])

        if not (0.0 <= val_acc <= 1.0):
            errors.append(f"[{desc}] `val_acc` must lie in [0, 1], got {val_acc}.")
        elif val_acc < 0.78:
            errors.append(f"[{desc}] `val_acc` was {val_acc:.4f} (expected >= 0.78). Check mini-batch training loop and `opt.zero_grad()`.")

        if not (0.0 <= val_auc <= 1.0):
            errors.append(f"[{desc}] `val_auc` must lie in [0, 1], got {val_auc}.")
        elif val_auc < 0.90:
            errors.append(
                f"[{desc}] `val_auc` was {val_auc:.4f} (expected >= 0.90). "
                "Use mini-batches (e.g. batch_size=32), `BCEWithLogitsLoss`, and `Adam(lr=0.01)`!"
            )

        if not (0.0 <= final_loss < 0.65):
            errors.append(
                f"[{desc}] `final_train_loss` was {final_loss:.5f} (expected in [0, 0.65), strictly below random guessing 0.69315)."
            )

        if kwargs["seed"] == 42:
            first_res = got

    # Reproducibility check for seed=42
    if not errors and first_res is not None:
        repeat_res = fn(n_samples=200, epochs=15, seed=42)
        for k in expected_keys:
            if not math.isclose(float(first_res[k]), float(repeat_res[k]), abs_tol=1e-4):
                errors.append(
                    f"[reproducibility] Calling twice with `seed=42` changed `{k}` ({first_res[k]} vs {repeat_res[k]}). "
                    "Call `torch.manual_seed(seed)` before instantiating `MedicalPatchCNN`!"
                )

    if errors:
        return CheckResult(
            "ch30_exam_02",
            False,
            "failed",
            "Exam Task 30.2 — Check MedicalPatchCNN Training Loop, AUC & Determinism",
            "Generate the synthetic lesion dataset, split stratified 75/25, seed PyTorch (`torch.manual_seed(seed)`), train `MedicalPatchCNN(num_classes=1)` with `BCEWithLogitsLoss` + `Adam(lr=0.01)`, and achieve `val_auc >= 0.90`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch30_exam_02",
        True,
        "passed",
        "Exam Task 30.2 Passed (35/35 pts) — End-to-End Medical Patch CNN Trainer Verified!",
        f"Achieved reproducible high discrimination on synthetic lesion patches (seed=42: val_acc={first_res['val_acc']}, val_auc={first_res['val_auc']}, loss={first_res['final_train_loss']}).",
        points_earned=35,
        points_possible=35,
        tip="Always pair `BCEWithLogitsLoss` with raw logits (no trailing Sigmoid inside the model) for numerical stability, and apply `torch.sigmoid` only during evaluation!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_exam_02",
        chapter="ch30",
        title="Exam 30.2: End-to-End Medical Patch CNN Trainer (train_medical_patch_cnn)",
        book_section="Chapter 30 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Generate `(X, y)` with `generate_synthetic_lesion_dataset(n_samples, img_size=16, seed=seed)` and split indices 75/25 using `train_test_split(..., test_size=0.25, stratify=y.numpy(), random_state=seed)`.",
            2: "Call `torch.manual_seed(seed)`, build `model = MedicalPatchCNN(num_classes=1)`, `loss_fn = nn.BCEWithLogitsLoss()`, `opt = torch.optim.Adam(model.parameters(), lr=0.01)`. Train in mini-batches (`batch_size=32`) for `epochs` epochs, then evaluate in `model.eval()` + `torch.no_grad()`.",
            3: "def train_medical_patch_cnn(n_samples=200, epochs=15, seed=42):\n    X, y = generate_synthetic_lesion_dataset(n_samples, 16, seed)\n    idx = np.arange(len(y))\n    tr, va = train_test_split(idx, test_size=0.25, stratify=y.numpy(), random_state=seed)\n    torch.manual_seed(seed)\n    model = MedicalPatchCNN(num_classes=1)\n    opt = torch.optim.Adam(model.parameters(), lr=0.01)\n    loss_fn = nn.BCEWithLogitsLoss()\n    for _ in range(epochs):\n        model.train()\n        perm = torch.randperm(len(tr))\n        ep_loss = 0.0\n        for s in range(0, len(tr), 32):\n            b = tr[perm[s:s+32].numpy()]\n            opt.zero_grad()\n            loss = loss_fn(model(X[b]).squeeze(-1), y[b])\n            loss.backward(); opt.step()\n            ep_loss += loss.item() * len(b)\n        final_loss = ep_loss / len(tr)\n    model.eval()\n    with torch.no_grad():\n        probs = torch.sigmoid(model(X[va]).squeeze(-1)).numpy()\n    preds = (probs >= 0.5).astype(np.float32)\n    return {'val_acc': round(float((preds == y[va].numpy()).mean()), 4), 'val_auc': round(float(roc_auc_score(y[va].numpy(), probs)), 4), 'final_train_loss': round(float(final_loss), 5)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 30.3 (35 pts): compute_saliency_attribution_map(model, x_single)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import torch

    bad = _check_callable("ch30_exam_03", "Exam Task 30.3", fn, "compute_saliency_attribution_map", 35)
    if bad is not None:
        return bad

    torch.manual_seed(42)
    model = _make_ref_medical_cnn(num_classes=1)
    X_ref, y_ref = _make_ref_lesion_dataset(n_samples=60, img_size=16, seed=42)
    pos_idx = int(np.where(y_ref.numpy() == 1.0)[0][0])

    x_16 = X_ref[pos_idx : pos_idx + 1].clone().detach()
    torch.manual_seed(99)
    x_20 = torch.randn(1, 1, 20, 20)

    expected_keys = ["saliency_map", "peak_coord", "pred_prob"]
    errors = []

    for x_orig, desc in [
        (x_16, "16x16 positive lesion patch"),
        (x_20, "20x20 variable-size patch"),
    ]:
        x_input = x_orig.clone().detach()
        assert not x_input.requires_grad

        try:
            got = fn(model, x_input)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch30_exam_03",
                False,
                "failed",
                "Exam Task 30.3 — Runtime Error in `compute_saliency_attribution_map`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch30_exam_03", "Exam Task 30.3", "compute_saliency_attribution_map(model, x_single)", 35
            )

        key_err = _check_dict_keys("ch30_exam_03", "Exam Task 30.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if x_input.requires_grad or x_input.grad is not None or not torch.equal(x_input, x_orig):
            errors.append(
                f"[{desc}] Input tensor `x_single` was mutated! Clone/detach it first: `x = x_single.clone().detach().requires_grad_(True)`."
            )

        # Compute reference saliency map
        model.eval()
        model.zero_grad()
        x_ref = x_orig.clone().detach().requires_grad_(True)
        logit = model(x_ref).squeeze()
        logit.backward()
        exp_sal = x_ref.grad.detach().abs().squeeze().cpu().numpy()
        exp_peak = tuple(int(v) for v in np.unravel_index(int(np.argmax(exp_sal)), exp_sal.shape))
        exp_prob = round(float(torch.sigmoid(logit.detach()).item()), 4)

        sal = got["saliency_map"]
        if not isinstance(sal, np.ndarray) or sal.shape != exp_sal.shape:
            errors.append(
                f"[{desc}] `saliency_map` must be a 2D NumPy array of shape {exp_sal.shape}, "
                f"got {type(sal).__name__} (shape={getattr(sal, 'shape', None)})."
            )
            continue

        if np.any(sal < 0):
            errors.append(f"[{desc}] `saliency_map` contains negative values — take `.abs()` of `x.grad`!")
        elif not np.allclose(sal, exp_sal, atol=1e-4):
            max_diff = float(np.max(np.abs(sal - exp_sal)))
            errors.append(
                f"[{desc}] `saliency_map` differs from `|d(logit)/dx|` (max diff={max_diff:.2e}). "
                "Backpropagate from the raw `logit` (not post-sigmoid probability) in `model.eval()`!"
            )

        peak = got["peak_coord"]
        if not isinstance(peak, tuple) or len(peak) != 2 or not (_is_int_like(peak[0]) and _is_int_like(peak[1])):
            errors.append(f"[{desc}] `peak_coord` must be a 2-tuple of ints `(row, col)`, got {peak!r}.")
        elif (int(peak[0]), int(peak[1])) != exp_peak:
            errors.append(f"[{desc}] `peak_coord`: expected {exp_peak}, got {peak!r}.")

        prob = got["pred_prob"]
        if not _is_np_or_py_number(prob) or not math.isclose(float(prob), exp_prob, abs_tol=1e-4):
            errors.append(f"[{desc}] `pred_prob`: expected {exp_prob}, got {prob!r}.")

    if errors:
        return CheckResult(
            "ch30_exam_03",
            False,
            "failed",
            "Exam Task 30.3 — Check Input-Gradient Saliency Attribution Map",
            "Clone `x_single` with `requires_grad_(True)` in `model.eval()`, backprop `logit = model(x).squeeze()`, take `x.grad.detach().abs().squeeze().cpu().numpy()`, locate `peak_coord`, and compute `pred_prob = round(sigmoid(logit), 4)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch30_exam_03",
        True,
        "passed",
        "Exam Task 30.3 Passed (35/35 pts) — Input-Gradient Saliency Map Verified!",
        "Computed exact `|d(logit)/dx|` pixel attribution maps, peak coordinates, and predicted probabilities across variable image sizes without mutating inputs.",
        points_earned=35,
        points_possible=35,
        tip="Backpropagating from the pre-sigmoid `logit` avoids sigmoid saturation gradients near 0 or 1, giving crisper clinical attribution maps!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch30_exam_03",
        chapter="ch30",
        title="Exam 30.3: Input-Gradient Saliency Attribution Map (compute_saliency_attribution_map)",
        book_section="Chapter 30 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Put `model.eval()` and `model.zero_grad()`. Clone `x_single` safely without mutating it: `x = x_single.clone().detach().requires_grad_(True)`.",
            2: "Compute `logit = model(x).squeeze()`, call `logit.backward()`, extract `saliency = x.grad.detach().abs().squeeze().cpu().numpy()`, find `peak_coord = tuple(int(v) for v in np.unravel_index(int(np.argmax(saliency)), saliency.shape))`, and compute `pred_prob = round(float(torch.sigmoid(logit.detach()).item()), 4)`.",
            3: "def compute_saliency_attribution_map(model, x_single):\n    model.eval()\n    model.zero_grad()\n    x = x_single.clone().detach().requires_grad_(True)\n    logit = model(x).squeeze()\n    logit.backward()\n    saliency = x.grad.detach().abs().squeeze().cpu().numpy()\n    peak = tuple(int(v) for v in np.unravel_index(int(np.argmax(saliency)), saliency.shape))\n    pred_prob = round(float(torch.sigmoid(logit.detach()).item()), 4)\n    return {'saliency_map': saliency, 'peak_coord': peak, 'pred_prob': pred_prob}",
        },
        verifier=_verify_exam_03,
    )
)
