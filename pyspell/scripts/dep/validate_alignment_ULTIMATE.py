"""validate_alignment_ULTIMATE.py

Visualization-heavy alignment QC modeled after the MATLAB helper
`viz_alignment_validation_deck(...)` added to batchRunCellReg_ULTIMATE.m.

Produces:
  1) One overview figure (multi-session):
       Validation_ReferenceVsAllSessions_Footprints.png
  2) Per-session validate_alignment-style 2x2 panels:
       Validation_Panel_Ref<ref>_vs_Sess<i>.png
  3) Per-session "shifted footprints" figure (explicit before/after):
       Validation_ShiftedFootprints_Ref<ref>_vs_Sess<i>.png

Overlays use: Red = reference, Green = session (yellow = good alignment)

You can use this as:
  - a module (call validate_alignment_deck)
  - a CLI on a .npz or .mat that contains the required arrays

Expected data fields (npz/mat):
  mean_images              : (N,H,W) or list/tuple of 2D arrays
  footprints_proj_raw      : (N,H,W) or list/tuple of 2D arrays
  footprints_proj_aligned  : (N,H,W) or list/tuple of 2D arrays (optional)
  alignment_translations   : (N,2|3) or (2|3,N) array (optional)
  scores                   : (N,) array/list or list of arrays (optional)
  session_names            : list of strings (optional)

Notes
-----
- If footprints_proj_aligned is omitted, the script will generate a
  *display-only* alignment using a best-guess transform based on
  alignment_translations and correlation with the reference.
- The best-guess transform tries common sign/order conventions and picks
  the variant with highest corr(ref, transformed_moving).

John Stout / Spellman Lab ecosystem — ported style from MATLAB deck.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# --- optional deps for transforms ---
try:
    from scipy.ndimage import rotate as _nd_rotate
    from scipy.ndimage import shift as _nd_shift
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    from scipy.io import loadmat as _loadmat
    _HAVE_SCIPY_IO = True
except Exception:
    _HAVE_SCIPY_IO = False

# Try to import pyspell.cellregpy
try:
    from pyspell.cellregpy import (
        compute_centroids,
        compute_data_distribution, 
        estimate_num_bins,
        compute_centroid_distances_model_custom,
        compute_spatial_correlations_model,
        compute_p_same,
        estimate_registration_accuracy,
        cluster_cells,
        choose_best_model,
        initial_registration_centroid_distances_custom,
        initial_registration_spatial_corr
    )
    _HAVE_PYSPELL = True
except ImportError:
    # Try adding parent directory
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parents[1] # pyspell/scripts -> pyspell -> root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from pyspell.cellregpy import (
            compute_centroids,
            compute_data_distribution, 
            estimate_num_bins,
            compute_centroid_distances_model_custom,
            compute_spatial_correlations_model,
            compute_p_same,
            estimate_registration_accuracy,
            cluster_cells,
            choose_best_model,
            initial_registration_centroid_distances_custom,
            initial_registration_spatial_corr
        )
        _HAVE_PYSPELL = True
    except ImportError:
        _HAVE_PYSPELL = False
        print("Warning: pyspell not found. Model computation will be skipped.")


ArrayLike2D = Union[np.ndarray]


# =============================
# Utility / normalization
# =============================

def _as_list_of_2d(x: Any, name: str) -> List[np.ndarray]:
    """Convert x to a list of 2D float arrays."""
    if x is None:
        raise ValueError(f"{name} is required")

    # Already list/tuple
    if isinstance(x, (list, tuple)):
        out = [np.asarray(a, dtype=float) for a in x]
    else:
        arr = np.asarray(x)
        if arr.ndim == 2:
            out = [arr.astype(float)]
        elif arr.ndim == 3:
            out = [arr[i].astype(float) for i in range(arr.shape[0])]
        else:
            raise ValueError(f"{name} must be 2D, 3D, or list/tuple of 2D arrays; got shape {arr.shape}")

    # sanity
    for i, a in enumerate(out):
        if a.ndim != 2:
            raise ValueError(f"{name}[{i}] is not 2D (shape={a.shape})")
    return out


def _clamp01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0
    x[x > 1] = 1
    return x


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 0:
        return np.zeros_like(x, dtype=float)
    return _clamp01(x / mx)


def make_rgb_overlay(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return RGB overlay where A=red, B=green."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    mx = max(float(np.max(A)) if A.size else 0.0, float(np.max(B)) if B.size else 0.0, 1e-12)
    Ar = _clamp01(A / mx)
    Bg = _clamp01(B / mx)
    rgb = np.stack([Ar, Bg, np.zeros_like(Ar)], axis=-1)
    return rgb


def corr2_nan(A: np.ndarray, B: np.ndarray) -> float:
    """Correlation of A and B ignoring non-finite pixels."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    mask = np.isfinite(A) & np.isfinite(B)
    if mask.sum() < 10:
        return float("nan")
    a = A[mask]
    b = B[mask]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den == 0:
        return float("nan")
    return float(np.sum(a * b) / den)


# =============================
# Transform handling
# =============================

@dataclass
class BestTransform:
    tx: float
    ty: float
    rot_deg: float
    swap_xy: bool
    signx: int
    signy: int
    signr: int
    rot_then_trans: bool


def _apply_rigid_transform(
    img: np.ndarray,
    tx: float,
    ty: float,
    rot_deg: float,
    *,
    rot_then_trans: bool,
    fill: float = 0.0,
) -> np.ndarray:
    """Apply rotation (crop) and translation to a 2D image."""
    img = np.asarray(img, dtype=float)
    out = img

    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for rigid transforms in this script")

    def do_rot(x: np.ndarray) -> np.ndarray:
        if rot_deg == 0:
            return x
        return _nd_rotate(x, rot_deg, reshape=False, order=1, mode="constant", cval=fill)

    def do_shift(x: np.ndarray) -> np.ndarray:
        if tx == 0 and ty == 0:
            return x
        # ndimage.shift takes (shift_rows, shift_cols) = (dy, dx)
        return _nd_shift(x, shift=(ty, tx), order=1, mode="constant", cval=fill)

    if rot_then_trans:
        out = do_rot(out)
        out = do_shift(out)
    else:
        out = do_shift(out)
        out = do_rot(out)

    return out


def apply_transform_best(
    moving: np.ndarray,
    ref: np.ndarray,
    dx: float,
    dy: float,
    rot_deg: float = 0.0,
) -> Tuple[np.ndarray, BestTransform, float]:
    """Try common sign/swap/order conventions and return best by corr2_nan."""
    swap_opts = [False, True]
    sign_opts = [-1, 1]
    rot_opts = [-1, 1]
    order_opts = [True, False]  # rot_then_trans

    best_score = -np.inf
    best_img = moving
    best_params = BestTransform(
        tx=0.0,
        ty=0.0,
        rot_deg=0.0,
        swap_xy=False,
        signx=1,
        signy=1,
        signr=1,
        rot_then_trans=True,
    )

    for sw in swap_opts:
        for sx in sign_opts:
            for sy in sign_opts:
                for sr in rot_opts:
                    for rot_then_trans in order_opts:
                        if sw:
                            tx = sx * dy
                            ty = sy * dx
                        else:
                            tx = sx * dx
                            ty = sy * dy
                        r = sr * rot_deg

                        cand = _apply_rigid_transform(
                            moving,
                            tx=tx,
                            ty=ty,
                            rot_deg=r,
                            rot_then_trans=rot_then_trans,
                            fill=0.0,
                        )
                        sc = corr2_nan(ref, cand)
                        if np.isnan(sc):
                            continue
                        if sc > best_score:
                            best_score = sc
                            best_img = cand
                            best_params = BestTransform(
                                tx=float(tx),
                                ty=float(ty),
                                rot_deg=float(r),
                                swap_xy=sw,
                                signx=sx,
                                signy=sy,
                                signr=sr,
                                rot_then_trans=rot_then_trans,
                            )

    if best_score == -np.inf:
        best_score = float("nan")

    return best_img, best_params, float(best_score)


def _get_transform_params(alignment_translations: Optional[np.ndarray], idx: int) -> Tuple[float, float, float]:
    """Extract (dx,dy,rotDeg) from an alignment_translations array."""
    if alignment_translations is None:
        return 0.0, 0.0, 0.0

    T = np.asarray(alignment_translations, dtype=float)
    dx = dy = rot = 0.0

    # Allow (2|3, N) or (N, 2|3)
    if T.ndim == 1:
        if T.size >= 2:
            dx, dy = float(T[0]), float(T[1])
        if T.size >= 3:
            rot = float(T[2])
        return dx, dy, rot

    if T.ndim != 2:
        return 0.0, 0.0, 0.0

    if T.shape[0] in (2, 3):
        if idx < T.shape[1]:
            dx = float(T[0, idx])
            dy = float(T[1, idx])
            if T.shape[0] == 3:
                rot = float(T[2, idx])
    elif T.shape[1] in (2, 3):
        if idx < T.shape[0]:
            dx = float(T[idx, 0])
            dy = float(T[idx, 1])
            if T.shape[1] == 3:
                rot = float(T[idx, 2])

    if not np.isfinite(dx):
        dx = 0.0
    if not np.isfinite(dy):
        dy = 0.0
    if not np.isfinite(rot):
        rot = 0.0

    return dx, dy, rot


def get_session_score(scores: Any, idx: int) -> float:
    """Extract a per-session scalar score from common container types."""
    if scores is None:
        return float("nan")

    try:
        if isinstance(scores, (list, tuple)):
            if idx >= len(scores):
                return float(scores[0]) if len(scores) else float("nan")
            v = scores[idx]
            if np.isscalar(v):
                return float(v)
            v = np.asarray(v).ravel()
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")

        arr = np.asarray(scores)
        if arr.ndim == 0:
            return float(arr)
        if idx < arr.size:
            return float(arr.ravel()[idx])
        return float(arr.ravel()[0])
    except Exception:
        return float("nan")


def session_names_to_labels(session_names: Optional[Sequence[str]], N: int) -> List[str]:
    if not session_names:
        return [f"Session {i+1}" for i in range(N)]

    labels: List[str] = []
    for nm in session_names:
        if nm is None:
            labels.append("(none)")
            continue
        nm = str(nm)
        # Use parent directory if path-like
        base = os.path.basename(nm.rstrip(os.sep))
        parent = os.path.basename(os.path.dirname(nm.rstrip(os.sep)))
        # Heuristic: prefer parent if base looks like a file
        if "." in base and parent:
            labels.append(parent)
        else:
            labels.append(base or nm)

    # pad/truncate to N
    if len(labels) < N:
        labels.extend([f"Session {i+1}" for i in range(len(labels), N)])
    return labels[:N]


def savefig_both(fig: plt.Figure, out_base: str, *, dpi: int = 200, also_pdf: bool = False) -> None:
    if out_base is None:
        return
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(out_base + ".pdf", bbox_inches="tight")


# =============================
# Plotting
# =============================

def validate_alignment_deck(
    mean_images: Any,
    footprints_proj_raw: Any,
    footprints_proj_aligned: Any = None,
    *,
    reference_session_index: int = 0,
    alignment_translations: Optional[np.ndarray] = None,
    scores: Any = None,
    out_dir: str = ".",
    session_names: Optional[Sequence[str]] = None,
    show: bool = False,
    also_pdf: bool = False,
) -> None:
    """Generate the full validation deck."""

    if not _HAVE_SCIPY:
        raise ImportError(
            "This script uses scipy.ndimage for transforms. Install scipy or adapt transforms to your stack."
        )

    mean_images_l = _as_list_of_2d(mean_images, "mean_images")
    fp_raw_l = _as_list_of_2d(footprints_proj_raw, "footprints_proj_raw")
    fp_aligned_l = _as_list_of_2d(footprints_proj_aligned, "footprints_proj_aligned") if footprints_proj_aligned is not None else None

    N = len(mean_images_l)
    if len(fp_raw_l) != N:
        raise ValueError(f"footprints_proj_raw has {len(fp_raw_l)} sessions but mean_images has {N}")
    if fp_aligned_l is not None and len(fp_aligned_l) != N:
        raise ValueError(f"footprints_proj_aligned has {len(fp_aligned_l)} sessions but mean_images has {N}")

    ref = int(reference_session_index)
    if ref < 0 or ref >= N:
        raise ValueError(f"reference_session_index must be in [0,{N-1}] (0-based). Got {ref}.")

    labels = session_names_to_labels(session_names, N)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------
    # Figure 1: overview
    # ---------------------
    order = [ref] + [i for i in range(N) if i != ref]
    fig1, axes = plt.subplots(nrows=N, ncols=3, figsize=(12, max(3, 2.2 * N)), constrained_layout=True)
    if N == 1:
        axes = np.array([axes])

    fig1.suptitle(
        f"Validation: reference vs all sessions (Footprints) | reference = {labels[ref]} (idx={ref+1})",
        fontweight="bold",
    )

    for r, s in enumerate(order):
        # Column 1: session alone (aligned if available else raw)
        ax = axes[r, 0]
        solo = fp_aligned_l[s] if fp_aligned_l is not None else fp_raw_l[s]
        ax.imshow(_normalize01(solo), cmap="gray")
        ax.axis("off")
        ax.set_title("REFERENCE\n" + labels[s] if s == ref else labels[s], fontsize=9)

        # Column 2: RAW overlay vs ref
        ax = axes[r, 1]
        ax.imshow(make_rgb_overlay(fp_raw_l[ref], fp_raw_l[s]))
        ax.axis("off")
        ax.set_title("RAW overlay (R=ref, G=this)", fontsize=9)

        # Column 3: ALIGNED overlay vs ref
        ax = axes[r, 2]
        if fp_aligned_l is not None:
            aligned = fp_aligned_l[s]
        else:
            dx, dy, rot = _get_transform_params(alignment_translations, s)
            aligned, _, _ = apply_transform_best(fp_raw_l[s], fp_raw_l[ref], dx, dy, rot)
        ax.imshow(make_rgb_overlay(fp_raw_l[ref] if fp_aligned_l is None else fp_aligned_l[ref], aligned))
        ax.axis("off")
        sc = get_session_score(scores, s)
        sc_str = "n/a" if not np.isfinite(sc) else f"{sc:.3f}"
        ax.set_title(f"ALIGNED overlay (score={sc_str})", fontsize=9)

    savefig_both(fig1, os.path.join(out_dir, "Validation_ReferenceVsAllSessions_Footprints"), also_pdf=also_pdf)
    if not show:
        plt.close(fig1)

    # ---------------------------------
    # Per-session: validate panels + shifted footprints
    # ---------------------------------
    for s in range(N):
        if s == ref:
            continue

        sc = get_session_score(scores, s)
        sc_str = "n/a" if not np.isfinite(sc) else f"{sc:.3f}"

        # Determine a transform for display (based on mean images)
        dx, dy, rot = _get_transform_params(alignment_translations, s)
        mean_s_aligned, best_tf, _ = apply_transform_best(mean_images_l[s], mean_images_l[ref], dx, dy, rot)

        # Mean images aligned overlay
        mean_ref = mean_images_l[ref]
        mean_s_raw = mean_images_l[s]

        # Footprints
        fp_ref_raw = fp_raw_l[ref]
        fp_s_raw = fp_raw_l[s]

        if fp_aligned_l is not None:
            fp_ref_aligned = fp_aligned_l[ref]
            fp_s_aligned = fp_aligned_l[s]
        else:
            # Apply the same best transform found from mean images
            fp_s_aligned = _apply_rigid_transform(
                fp_s_raw,
                tx=best_tf.tx,
                ty=best_tf.ty,
                rot_deg=best_tf.rot_deg,
                rot_then_trans=best_tf.rot_then_trans,
                fill=0.0,
            )
            fp_ref_aligned = fp_ref_raw

        # ---------------- Panel figure (2x2) ----------------
        fig2, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), constrained_layout=True)
        fig2.suptitle(
            f"Alignment Validation (Score: {sc_str}) | Ref={labels[ref]} (idx={ref+1}) vs {labels[s]} (idx={s+1})",
            fontweight="bold",
        )

        ax[0, 0].imshow(make_rgb_overlay(mean_ref, mean_s_raw))
        ax[0, 0].set_title("Mean Images Corrected (Pre-Align) (Red=Ref, Grn=Other)")
        ax[0, 0].axis("off")

        ax[0, 1].imshow(make_rgb_overlay(mean_ref, mean_s_aligned))
        ax[0, 1].set_title("Mean Images ALIGNED (Should be yellow)")
        ax[0, 1].axis("off")

        ax[1, 0].imshow(make_rgb_overlay(fp_ref_raw, fp_s_raw))
        ax[1, 0].set_title("Footprints RAW (Red=Ref, Grn=Other)")
        ax[1, 0].axis("off")

        ax[1, 1].imshow(make_rgb_overlay(fp_ref_aligned, fp_s_aligned))
        ax[1, 1].set_title("Footprints ALIGNED (Should be yellow)")
        ax[1, 1].axis("off")

        out_base = os.path.join(out_dir, f"Validation_Panel_Ref{ref+1}_vs_Sess{s+1}")
        savefig_both(fig2, out_base, also_pdf=also_pdf)
        if not show:
            plt.close(fig2)

        # ---------- Shifted footprints explicit figure ----------
        # "Translation-only" view using the same best transform but with rot=0.
        fp_s_shift_only = _apply_rigid_transform(
            fp_s_raw,
            tx=best_tf.tx,
            ty=best_tf.ty,
            rot_deg=0.0,
            rot_then_trans=best_tf.rot_then_trans,
            fill=0.0,
        )

        fig3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), constrained_layout=True)
        fig3.suptitle(
            f"Shifted footprints (explicit) | Ref={labels[ref]} (idx={ref+1}) vs {labels[s]} (idx={s+1})",
            fontweight="bold",
        )

        ax3[0, 0].imshow(_normalize01(fp_s_raw), cmap="gray")
        ax3[0, 0].set_title("Moving footprints (RAW)")
        ax3[0, 0].axis("off")

        ax3[0, 1].imshow(_normalize01(fp_s_shift_only), cmap="gray")
        ax3[0, 1].set_title("Moving footprints (SHIFT only)")
        ax3[0, 1].axis("off")

        ax3[1, 0].imshow(make_rgb_overlay(fp_ref_raw, fp_s_raw))
        ax3[1, 0].set_title("Overlay RAW (R=Ref, G=Moving)")
        ax3[1, 0].axis("off")

        ax3[1, 1].imshow(make_rgb_overlay(fp_ref_aligned, fp_s_aligned))
        ax3[1, 1].set_title("Overlay ALIGNED (R=Ref, G=Moving)")
        ax3[1, 1].axis("off")

        out_base3 = os.path.join(out_dir, f"Validation_ShiftedFootprints_Ref{ref+1}_vs_Sess{s+1}")
        savefig_both(fig3, out_base3, also_pdf=also_pdf)
        if not show:
            plt.close(fig3)


# =============================
# IO helpers for CLI
# =============================


def _maybe_squeeze_mat_cell(x: Any) -> Any:
    """Best-effort conversion of MATLAB cell arrays loaded via scipy.io.loadmat."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        # squeeze
        x = np.squeeze(x)
        return [np.asarray(e).squeeze() for e in x.tolist()]
    return x


def load_inputs(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = dict(np.load(path, allow_pickle=True))
        return data

    if ext == ".mat":
        if not _HAVE_SCIPY_IO:
            raise ImportError("scipy.io.loadmat is required to load .mat files")
        md = _loadmat(path, squeeze_me=True, struct_as_record=False)
        # Drop MATLAB meta keys
        md = {k: v for k, v in md.items() if not k.startswith("__")}
        # Convert common cell-arrays
        for k in list(md.keys()):
            md[k] = _maybe_squeeze_mat_cell(md[k])
        return md

    raise ValueError(f"Unsupported file type: {ext}. Use .npz or .mat")


def _pick_key(d: Dict[str, Any], candidates: Sequence[str], required: bool = True) -> Any:
    for c in candidates:
        if c in d:
            return d[c]
    if required:
        raise KeyError(f"Could not find any of keys {candidates} in input file")
    return None


# ============================================================================ #
#                     PART B: MODELING STEP VALIDATION                         #
# ============================================================================ #

def plot_x_y_displacements(
    neighbors_x_displacements: np.ndarray,
    neighbors_y_displacements: np.ndarray,
    microns_per_pixel: float,
    maximal_distance: float,
    number_of_bins: int,
    centers_of_bins: Any,
    out_dir: str,
    show: bool = False,
    also_pdf: bool = False
) -> None:
    """
    Plots the (x,y) distribution of cell-pair displacements.
    Exact replica of MATLAB plot_displacement (batchRunCellReg_ULTIMATE.m lines 812-877).
    """
    if neighbors_x_displacements is None or neighbors_y_displacements is None:
        print("Skipping plot_x_y_displacements (missing data)")
        return

    x_disp = np.asarray(neighbors_x_displacements).ravel()
    y_disp = np.asarray(neighbors_y_displacements).ravel()
    mask = np.isfinite(x_disp) & np.isfinite(y_disp)
    x_disp = x_disp[mask]
    y_disp = y_disp[mask]
    if len(x_disp) == 0:
        return

    # --- Build bin centers exactly as MATLAB ---
    # MATLAB: xout_temp_2=linspace(0,maximal_distance,number_of_bins+1);
    #         xout_2=xout_temp_2(2:2:end);
    #         centers_of_bins_xy{1}=[-flip(xout_2), xout_2];
    xout_temp_2 = np.linspace(0, maximal_distance, number_of_bins + 1)
    xout_2 = xout_temp_2[1::2]  # MATLAB 2:2:end → 0-based [1::2]
    xy_centers = np.concatenate([-np.flip(xout_2), xout_2])
    n_xy = len(xy_centers)

    # --- hist3 equivalent ---
    # MATLAB: hist3([x;y]', {xy_centers, xy_centers})
    # Build edges from centers for np.histogram2d
    if n_xy > 1:
        step = xy_centers[1] - xy_centers[0]
        edges = np.concatenate([xy_centers - step / 2, [xy_centers[-1] + step / 2]])
    else:
        edges = np.array([-maximal_distance, maximal_distance])

    H, _, _ = np.histogram2d(x_disp, y_disp, bins=[edges, edges])

    # MATLAB: flipud(fliplr(H))
    H = np.flipud(np.fliplr(H))

    # Log normalisation: imagesc(log(1+H)./max(max(log(1+H))))
    H_log = np.log1p(H)
    mx = float(np.max(H_log))
    if mx > 0:
        H_log /= mx

    # --- Figure (matches MATLAB axes positions) ---
    fig = plt.figure(figsize=(8, 7))
    size_x, size_y = 0.75, 0.75

    ax = fig.add_axes([0.12, 0.15, size_x, size_y])
    # imagesc-style: plot as image with bin-index axes
    im = ax.imshow(H_log, aspect='equal', cmap='jet',
                   interpolation='nearest', vmin=0, vmax=1)

    # Manual tick labels in microns (matching MATLAB)
    # MATLAB: y=round(linspace(1,number_of_bins,9))
    # Here the image is n_xy × n_xy so ticks span [0, n_xy-1]
    # Extract the dist-centers for tick labels
    if isinstance(centers_of_bins, (list, tuple)):
        cob0 = np.asarray(centers_of_bins[0])
    else:
        cob0 = np.asarray(centers_of_bins).ravel()
    max_cob = float(np.max(cob0)) if len(cob0) else maximal_distance

    tick_positions = np.round(np.linspace(0, n_xy - 1, 9)).astype(int)
    y_labels = np.round(np.linspace(microns_per_pixel * max_cob,
                                     -microns_per_pixel * max_cob, 9)).astype(int)
    x_labels = np.round(np.linspace(-microns_per_pixel * max_cob,
                                     microns_per_pixel * max_cob, 9)).astype(int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(y_labels, fontsize=14)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlabel('x displacement (µm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('y displacement (µm)', fontweight='bold', fontsize=14)

    # --- Circles at 4 µm and 8 µm (MATLAB index-space radii) ---
    # MATLAB: plot(n/2 + n/2*4/maxD/mpp*sin(t), n/2 + n/2*4/maxD/mpp*cos(t), ...)
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = n_xy / 2, n_xy / 2
    r1 = n_xy / 2 * 4 / maximal_distance / microns_per_pixel
    r2 = n_xy / 2 * 8 / maximal_distance / microns_per_pixel
    ax.plot(cx + r1 * np.sin(theta), cy + r1 * np.cos(theta),
            ':', color='white', linewidth=4)
    ax.plot(cx + r2 * np.sin(theta), cy + r2 * np.cos(theta),
            '--', color='white', linewidth=4)

    # --- Manual jet colorbar (matches MATLAB style) ---
    cax = fig.add_axes([0.855, 0.15, 0.02, size_y])
    cmap_jet = plt.cm.jet(np.linspace(0, 1, 64))
    for i in range(64):
        cax.fill_between([0, 1],
                         i / 64, (i + 1) / 64,
                         color=cmap_jet[i])
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([])
    cax.text(3.5, 0.5, 'Number of cell-pairs (log)',
             fontsize=14, fontweight='bold', rotation=90,
             ha='center', va='center', transform=cax.transAxes)
    cax.text(1.5, 0.0, '0', fontsize=14, fontweight='bold',
             ha='left', transform=cax.transAxes)
    cax.text(1.5, 1.0, 'Max', fontsize=14, fontweight='bold',
             ha='left', transform=cax.transAxes)

    savefig_both(fig, os.path.join(out_dir, "Stage 3 - (x,y) displacements"), also_pdf=also_pdf)
    if not show:
        plt.close(fig)


def plot_models(
    # Centroid args
    centroid_distances_model_parameters: np.ndarray,
    NN_centroid_distances: np.ndarray,
    NNN_centroid_distances: np.ndarray,
    centroid_distances_distribution: np.ndarray,
    centroid_distances_model_same_cells: np.ndarray,
    centroid_distances_model_different_cells: np.ndarray,
    centroid_distances_model_weighted_sum: np.ndarray,
    centroid_distance_intersection: float,
    centers_of_bins_dist: np.ndarray,
    # Spatial args (optional)
    spatial_correlations_model_parameters: Optional[np.ndarray] = None,
    NN_spatial_correlations: Optional[np.ndarray] = None,
    NNN_spatial_correlations: Optional[np.ndarray] = None,
    spatial_correlations_distribution: Optional[np.ndarray] = None,
    spatial_correlations_model_same_cells: Optional[np.ndarray] = None,
    spatial_correlations_model_different_cells: Optional[np.ndarray] = None,
    spatial_correlations_model_weighted_sum: Optional[np.ndarray] = None,
    spatial_correlation_intersection: Optional[float] = None,
    centers_of_bins_corr: Optional[np.ndarray] = None,
    # General
    microns_per_pixel: float = 1.0,
    maximal_distance: float = 10.0,
    out_dir: str = ".",
    show: bool = False,
    also_pdf: bool = False
) -> None:
    """
    Plots probabilistic models.
    Exact replica of MATLAB plot_model (batchRunCellReg_ULTIMATE.m lines 879-1004).
    """
    has_spatial = (spatial_correlations_model_parameters is not None)
    number_of_bins = len(centers_of_bins_dist)
    x_dist = microns_per_pixel * centers_of_bins_dist

    # Build histogram edges from centers
    if number_of_bins > 1:
        step_d = centers_of_bins_dist[1] - centers_of_bins_dist[0]
        edges_dist = np.concatenate([centers_of_bins_dist - step_d / 2,
                                     [centers_of_bins_dist[-1] + step_d / 2]])
    else:
        edges_dist = np.linspace(0, maximal_distance, 10)

    # MATLAB: [n1,~]=hist(NN_centroid_distances, centers_of_bins{1})
    n1_cd, _ = np.histogram(NN_centroid_distances, bins=edges_dist)
    n2_cd, _ = np.histogram(NNN_centroid_distances, bins=edges_dist)

    # Bar offset (MATLAB: 0.25*microns_per_pixel*maximal_distance/number_of_bins)
    bar_offset_d = 0.25 * microns_per_pixel * maximal_distance / number_of_bins
    bar_width_d = 2 * bar_offset_d  # barwidth=0.5 of bin in MATLAB

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    # ==================== subplot(2,2,1): Centroid Dist Histogram ====================
    ax = axes[0, 0]
    ax.bar(x_dist + bar_offset_d, n1_cd, width=bar_width_d,
           color='g', edgecolor='none', label='Nearest neighbors')
    ax.bar(x_dist - bar_offset_d, n2_cd, width=bar_width_d,
           color='r', edgecolor='none', label='Other neighbors')
    ax.set_xlim(0, microns_per_pixel * maximal_distance)
    xtk_d = np.arange(0, microns_per_pixel * maximal_distance + 1, 3)
    ax.set_xticks(xtk_d)
    ax.set_xticklabels(xtk_d.astype(int), fontsize=14)
    ax.set_xlabel('Centroids distance (µm)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)
    ax.tick_params(labelsize=14)

    # ==================== subplot(2,2,2): Spatial Corr Histogram ====================
    if has_spatial:
        # Filter negatives (MATLAB: NN_spatial_correlations(NN_spatial_correlations<0)=[])
        nn_sc = np.asarray(NN_spatial_correlations).ravel()
        nnn_sc = np.asarray(NNN_spatial_correlations).ravel()
        nn_sc = nn_sc[nn_sc >= 0]
        nnn_sc = nnn_sc[nnn_sc >= 0]

        x_corr = np.asarray(centers_of_bins_corr)
        n_bins_c = len(x_corr)
        if n_bins_c > 1:
            step_c = x_corr[1] - x_corr[0]
            edges_corr = np.concatenate([x_corr - step_c / 2, [x_corr[-1] + step_c / 2]])
        else:
            edges_corr = np.linspace(0, 1, 10)

        n1_sc, _ = np.histogram(nn_sc, bins=edges_corr)
        n2_sc, _ = np.histogram(nnn_sc, bins=edges_corr)

        bar_offset_c = 0.25 / n_bins_c
        bar_width_c = 2 * bar_offset_c

        ax = axes[0, 1]
        ax.bar(x_corr + bar_offset_c, n1_sc, width=bar_width_c,
               color='g', edgecolor='none', label='Nearest neighbors')
        ax.bar(x_corr - bar_offset_c, n2_sc, width=bar_width_c,
               color='r', edgecolor='none', label='Other neighbors')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper left', frameon=False)
        xtk_c = np.linspace(0, 1, 6)
        ax.set_xticks(xtk_c)
        ax.set_xticklabels([f'{v:.1f}' for v in xtk_c], fontsize=14)
        ax.set_xlabel('Spatial correlation', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)
        ax.tick_params(labelsize=14)
    else:
        axes[0, 1].axis('off')

    # ==================== subplot(2,2,3): Centroid Model Fit =======================
    ax_m = axes[1, 0]
    p_same = float(centroid_distances_model_parameters[0])

    # Blue bar (MATLAB: bar(..., 'FaceColor','b', 'EdgeColor','none', 'barwidth',1))
    if number_of_bins > 1:
        bw_full = x_dist[1] - x_dist[0]
    else:
        bw_full = 1.0
    ax_m.bar(x_dist, centroid_distances_distribution, width=bw_full,
             color='b', edgecolor='none')

    # Model curves
    ax_m.plot(x_dist, p_same * centroid_distances_model_same_cells,
              '--', color='g', linewidth=3, label='Same cell model')
    ax_m.plot(x_dist, (1 - p_same) * centroid_distances_model_different_cells,
              '--', color='r', linewidth=3, label='Different cells model')
    ax_m.plot(x_dist, centroid_distances_model_weighted_sum,
              '-', color='k', linewidth=3, label='Overall model')

    # Intersection line + percentage annotations
    if centroid_distance_intersection is not None and np.isfinite(centroid_distance_intersection):
        ci = float(centroid_distance_intersection)
        ymax_d = float(np.max(centroid_distances_distribution)) if len(centroid_distances_distribution) else 1
        ax_m.plot([ci, ci], [0, ymax_d], '--', color='k', linewidth=2)

        norm_same = centroid_distances_model_same_cells / (np.sum(centroid_distances_model_same_cells) + 1e-12)
        norm_diff = centroid_distances_model_different_cells / (np.sum(centroid_distances_model_different_cells) + 1e-12)
        same_gt = float(np.sum(norm_same[x_dist > ci]))
        diff_gt = float(np.sum(norm_diff[x_dist > ci]))
        ax_m.text(ci + 1, 0.9 * ymax_d, f'{round(100 * same_gt)}%',
                  fontsize=14, fontweight='bold', ha='center', color='g')
        ax_m.text(ci - 1, 0.9 * ymax_d, f'{round(100 * (1 - same_gt))}%',
                  fontsize=14, fontweight='bold', ha='center', color='g')
        ax_m.text(ci + 1, 0.8 * ymax_d, f'{round(100 * diff_gt)}%',
                  fontsize=14, fontweight='bold', ha='center', color='r')
        ax_m.text(ci - 1, 0.8 * ymax_d, f'{round(100 * (1 - diff_gt))}%',
                  fontsize=14, fontweight='bold', ha='center', color='r')

    ax_m.set_xlim(0, microns_per_pixel * maximal_distance)
    ax_m.set_xlabel('Centroids distance (µm)', fontweight='bold', fontsize=14)
    ax_m.set_ylabel('Probability density', fontweight='bold', fontsize=14)
    ax_m.set_xticks(xtk_d)
    ax_m.set_xticklabels(xtk_d.astype(int), fontsize=14)
    ax_m.tick_params(labelsize=14)

    # ==================== subplot(2,2,4): Spatial Model Fit ========================
    if has_spatial:
        ax_ms = axes[1, 1]
        p_same_s = float(spatial_correlations_model_parameters[0])

        if n_bins_c > 1:
            bw_full_c = x_corr[1] - x_corr[0]
        else:
            bw_full_c = 0.05
        ax_ms.bar(x_corr, spatial_correlations_distribution, width=bw_full_c,
                  color='b', edgecolor='none')

        ax_ms.plot(x_corr, p_same_s * spatial_correlations_model_same_cells,
                   '--', color='g', linewidth=3)
        ax_ms.plot(x_corr, (1 - p_same_s) * spatial_correlations_model_different_cells,
                   '--', color='r', linewidth=3)
        ax_ms.plot(x_corr, spatial_correlations_model_weighted_sum,
                   '-', color='k', linewidth=3)

        if spatial_correlation_intersection is not None and np.isfinite(spatial_correlation_intersection):
            sci = float(spatial_correlation_intersection)
            ymax_s = float(np.max(spatial_correlations_distribution)) if len(spatial_correlations_distribution) else 1
            ax_ms.plot([sci, sci], [0, ymax_s], '--', color='k', linewidth=2)

            norm_same_s = spatial_correlations_model_same_cells / (np.sum(spatial_correlations_model_same_cells) + 1e-12)
            norm_diff_s = spatial_correlations_model_different_cells / (np.sum(spatial_correlations_model_different_cells) + 1e-12)
            same_gt_s = float(np.sum(norm_same_s[x_corr > sci]))
            diff_gt_s = float(np.sum(norm_diff_s[x_corr > sci]))
            ax_ms.text(sci + 0.1, 0.9 * ymax_s, f'{round(100 * same_gt_s)}%',
                       fontsize=14, fontweight='bold', ha='center', color='g')
            ax_ms.text(sci - 0.1, 0.9 * ymax_s, f'{round(100 * (1 - same_gt_s))}%',
                       fontsize=14, fontweight='bold', ha='center', color='g')
            ax_ms.text(sci + 0.1, 0.8 * ymax_s, f'{round(100 * diff_gt_s)}%',
                       fontsize=14, fontweight='bold', ha='center', color='r')
            ax_ms.text(sci - 0.1, 0.8 * ymax_s, f'{round(100 * (1 - diff_gt_s))}%',
                       fontsize=14, fontweight='bold', ha='center', color='r')

        ax_ms.set_xlim(0, 1)
        ax_ms.set_xlabel('Spatial correlation', fontweight='bold', fontsize=14)
        ax_ms.set_ylabel('Probability density', fontweight='bold', fontsize=14)
        xtk_c = np.linspace(0, 1, 6)
        ax_ms.set_xticks(xtk_c)
        ax_ms.set_xticklabels([f'{v:.1f}' for v in xtk_c], fontsize=14)
        ax_ms.tick_params(labelsize=14)
        ax_ms.legend(['Observed data', 'Same cell model', 'Different cells model', 'Overall model'],
                     loc='upper left', frameon=False)
    else:
        axes[1, 1].axis('off')

    savefig_both(fig, os.path.join(out_dir, "Stage 3 - model"), also_pdf=also_pdf)
    if not show:
        plt.close(fig)


def plot_cell_scores(
    cell_scores: np.ndarray,
    cell_scores_exclusive: np.ndarray,
    cell_scores_positive: np.ndarray,
    cell_scores_negative: np.ndarray,
    p_same_registered_pairs: Union[list, np.ndarray],
    out_dir: str,
    show: bool = False,
    also_pdf: bool = False
) -> None:
    """
    Plots score distributions.
    Exact replica of MATLAB plot_scores (batchRunCellReg_ULTIMATE.m lines 1116-1308).
    """
    # MATLAB: xout_temp=linspace(0,1,41); xout=xout_temp(2:2:end);
    xout_temp = np.linspace(0, 1, 41)
    xout = xout_temp[1::2]  # 20 bin centers

    # Histogram edges from centers
    if len(xout) > 1:
        step = xout[1] - xout[0]
        edges = np.concatenate([xout - step / 2, [xout[-1] + step / 2]])
    else:
        edges = np.linspace(0, 1, 22)

    number_of_clusters = len(cell_scores) if cell_scores is not None else 0
    size_x, size_y = 0.65, 0.65

    def _score_panel(fig, pos, inset_pos, data, xlabel_text, show_title=False):
        """Helper: main bar + CDF inset, matching MATLAB absolute axes positions."""
        ax = fig.add_axes(pos)
        if data is None or len(data) == 0:
            return
        n1, _ = np.histogram(data, bins=edges)
        total = float(np.sum(n1))
        if total > 0:
            n1 = n1 / total
        ax.bar(xout, n1, width=1.0 * step, color='steelblue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        xtk = np.linspace(0, 1, 6)
        ax.set_xticks(xtk)
        ax.set_xticklabels([f'{v:.1f}' for v in xtk], fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel_text, fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=14)
        if show_title:
            ax.text(-0.25, 1.2, f'{number_of_clusters} registered cells',
                    fontsize=16, fontweight='bold', ha='center',
                    transform=ax.transAxes)

        # CDF inset (MATLAB: plot(flip(xout), cumsum(flip(n1))))
        ax_in = fig.add_axes(inset_pos)
        ax_in.plot(np.flip(xout), np.cumsum(np.flip(n1)), linewidth=2)
        ax_in.set_ylim(0, 1)
        ax_in.invert_xaxis()
        xtk3 = np.linspace(0, 1, 3)
        ax_in.set_xticks(xtk3)
        ax_in.set_xticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
        ax_in.set_yticks(xtk3)
        ax_in.set_yticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
        ax_in.set_xlabel('Score', fontsize=14, fontweight='bold')
        ax_in.set_ylabel('Cum. fraction', fontsize=14, fontweight='bold')

    fig = plt.figure(figsize=(12, 10))

    # MATLAB axes positions (from plot_scores):
    # True Positive:  axes('position',[0.6  0.58 size_x/2 size_y/2])  → Top Right
    # True Negative:  axes('position',[0.12 0.58 size_x/2 size_y/2])  → Top Left
    # Overall:        axes('position',[0.6  0.1  size_x/2 size_y/2])  → Bottom Right
    # Exclusivity:    axes('position',[0.12 0.1  size_x/2 size_y/2])  → Bottom Left

    sx2, sy2 = size_x / 2, size_y / 2

    # True Negative (Top Left)
    _score_panel(fig,
                 [0.12, 0.58, sx2, sy2],
                 [0.2, 0.73, sx2 / 3, sy2 / 3],
                 cell_scores_negative, 'True negative scores')

    # True Positive (Top Right) — with cluster count title
    _score_panel(fig,
                 [0.6, 0.58, sx2, sy2],
                 [0.68, 0.73, sx2 / 3, sy2 / 3],
                 cell_scores_positive, 'True positive scores',
                 show_title=True)

    # Exclusivity (Bottom Left)
    _score_panel(fig,
                 [0.12, 0.1, sx2, sy2],
                 [0.2, 0.25, sx2 / 3, sy2 / 3],
                 cell_scores_exclusive, 'Exclusivity cell scores')

    # Overall (Bottom Right)
    _score_panel(fig,
                 [0.6, 0.1, sx2, sy2],
                 [0.68, 0.25, sx2 / 3, sy2 / 3],
                 cell_scores, 'Overall cell scores')

    savefig_both(fig, os.path.join(out_dir, "Stage 5 - cell scores"), also_pdf=also_pdf)
    if not show:
        plt.close(fig)

    # ---- P_same pairs plot (separate figure) ----
    # MATLAB extracts upper triangle only: for k=1:N, for m=k+1:N, ...
    if p_same_registered_pairs is not None:
        p_pairs = []
        if isinstance(p_same_registered_pairs, (list, tuple)):
            for mat in p_same_registered_pairs:
                if mat is not None:
                    mat_arr = np.asarray(mat)
                    if mat_arr.ndim == 2:
                        # Upper triangle
                        rows_m, cols_m = mat_arr.shape
                        for k in range(rows_m):
                            for m in range(k + 1, cols_m):
                                v = mat_arr[k, m]
                                if np.isfinite(v):
                                    p_pairs.append(v)
                    else:
                        v = mat_arr.ravel()
                        p_pairs.extend(v[np.isfinite(v)])
        elif isinstance(p_same_registered_pairs, np.ndarray):
            v = p_same_registered_pairs.ravel()
            p_pairs = list(v[np.isfinite(v)])

        if len(p_pairs) > 0:
            p_pairs = np.array(p_pairs)
            fig2 = plt.figure(figsize=(6, 5))

            ax2 = fig2.add_axes([0.15, 0.15, 0.75, 0.75])
            n1_p, _ = np.histogram(p_pairs, bins=edges)
            total_p = float(np.sum(n1_p))
            if total_p > 0:
                n1_p = n1_p / total_p
            ax2.bar(xout, n1_p, width=step, color='steelblue')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            xtk = np.linspace(0, 1, 6)
            ax2.set_xticks(xtk)
            ax2.set_xticklabels([f'{v:.1f}' for v in xtk], fontsize=14, fontweight='bold')
            ax2.set_xlabel('Registered pairs P$_{same}$', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Probability', fontsize=14, fontweight='bold')
            ax2.tick_params(labelsize=14)

            # CDF inset
            ax2_in = fig2.add_axes([0.3, 0.5, 0.3, 0.3])
            ax2_in.plot(np.flip(xout), np.cumsum(np.flip(n1_p)), linewidth=2)
            ax2_in.set_ylim(0, 1)
            ax2_in.invert_xaxis()
            xtk3 = np.linspace(0, 1, 3)
            ax2_in.set_xticks(xtk3)
            ax2_in.set_xticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
            ax2_in.set_yticks(xtk3)
            ax2_in.set_yticklabels([f'{v:.1f}' for v in xtk3], fontsize=14, fontweight='bold')
            ax2_in.set_xlabel('P$_{same}$', fontsize=14, fontweight='bold')
            ax2_in.set_ylabel('Cum. fraction', fontsize=14, fontweight='bold')

            savefig_both(fig2, os.path.join(out_dir, "Stage 5 - Registered pairs P_same"), also_pdf=also_pdf)
            if not show:
                plt.close(fig2)


def plot_all_registered_projections(
    spatial_footprints,  # list of arrays or cell array
    cell_to_index_map,
    out_dir: str,
    show: bool = False,
    also_pdf: bool = False,
    stage_label: str = "Stage 5"
) -> None:
    """
    Plots projections of all cells.
    Exact replica of MATLAB plot_all_registered_proj
    (batchRunCellReg_ULTIMATE.m lines 1006-1113).
    Green = cells detected in all sessions.
    """
    if spatial_footprints is None or cell_to_index_map is None:
        return

    # Convert to list of 3D arrays
    if isinstance(spatial_footprints, np.ndarray) and spatial_footprints.dtype == object:
        fp_list = [np.asarray(e) for e in spatial_footprints.tolist()]
    elif isinstance(spatial_footprints, (list, tuple)):
        fp_list = [np.asarray(e) for e in spatial_footprints]
    else:
        arr = np.asarray(spatial_footprints)
        if arr.dtype == object:
            fp_list = [np.asarray(e) for e in arr.tolist()]
        else:
            print("Warning: Unknown spatial_footprints format")
            return

    map_arr = np.asarray(cell_to_index_map)  # (num_clusters, num_sessions)
    num_sessions = len(fp_list)
    if map_arr.shape[1] != num_sessions:
        print(f"Warning: Map sessions ({map_arr.shape[1]}) != footprint sessions ({num_sessions})")
        return

    # MATLAB: cells_in_all_days = find(sum(cell_to_index_map'>0) == number_of_sessions)
    present_counts = np.sum(map_arr > 0, axis=1)
    idx_all = np.where(present_counts == num_sessions)[0]

    # MATLAB pixel_weight_threshold = 0.5
    pixel_weight_threshold = 0.5

    projections = []
    print("Calculating spatial footprints projections:")
    for s in range(num_sessions):
        fps = fp_list[s]  # (n_cells, h, w)
        if fps.ndim != 3:
            print(f"Warning: Session {s} footprints not 3D")
            continue

        c_idxs = map_arr[:, s].astype(int)

        # Handle 1-based vs 0-based indices
        valid_mask = c_idxs >= 0
        if np.any(valid_mask):
            mx = c_idxs[valid_mask].max()
            if mx >= fps.shape[0]:
                c_idxs = c_idxs - 1
        c_idxs[c_idxs < 0] = -1

        h, w = fps.shape[1], fps.shape[2]

        # ---- Per-cell pixel_weight_threshold + normalization (MATLAB lines 1041-1048) ----
        # For each cell: zero out pixels < 0.5*max, then normalize to [0,1]
        n_cells = fps.shape[0]
        normalized_fps = np.zeros_like(fps, dtype=float)
        for k in range(n_cells):
            fp = fps[k].astype(float)
            mx_fp = float(np.max(fp))
            if mx_fp > 0:
                fp[fp < pixel_weight_threshold * mx_fp] = 0
                normalized_fps[k] = fp / mx_fp

        # ---- Build RGB projection (MATLAB lines 1051-1058) ----
        # MATLAB assignment order:
        #   R = sum(other)
        #   G = sum(other) + sum(all)  [overwrites initial G = sum(all)]
        #   B = sum(other)
        # Cells "in all": green; others: white/gray

        idxs_all = c_idxs[idx_all]
        idxs_all = idxs_all[idxs_all >= 0]

        # "other" = clusters present in this session but NOT in all sessions
        logical_not_all = present_counts < num_sessions
        idx_other_clusters = np.where(np.logical_and(c_idxs > -1, logical_not_all))[0]
        # Wait — we need to use cluster indices. Let me re-derive properly.
        # MATLAB: other_cells{n} = find(cell_to_index_map(:,n)'>0 & logical_1)
        #   where logical_1 = sum(cell_to_index_map'>0) < num_sessions
        # In other words: clusters that ARE in this session but NOT in all sessions.
        other_cluster_mask = (c_idxs >= 0) & (present_counts < num_sessions)
        idxs_other = c_idxs[other_cluster_mask]
        idxs_other = idxs_other[idxs_other >= 0]

        img_rgb = np.zeros((h, w, 3), dtype=float)
        sum_all = np.zeros((h, w), dtype=float)
        sum_other = np.zeros((h, w), dtype=float)

        if len(idxs_all) > 0:
            sum_all = np.sum(normalized_fps[idxs_all], axis=0)
        if len(idxs_other) > 0:
            sum_other = np.sum(normalized_fps[idxs_other], axis=0)

        img_rgb[..., 0] = sum_other                 # R
        img_rgb[..., 1] = sum_other + sum_all        # G
        img_rgb[..., 2] = sum_other                  # B
        img_rgb = _clamp01(img_rgb)
        projections.append(img_rgb)

    # ---- Grid plot (MATLAB lines 1061-1093) ----
    subx = 4
    suby = int(np.ceil(num_sessions / subx))
    if num_sessions <= 4:
        # MATLAB: figure with (1, number_of_sessions) layout
        fig, axes = plt.subplots(1, num_sessions, figsize=(4 * num_sessions, 5))
        if num_sessions == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(suby, subx, figsize=(4 * subx, 4 * suby))

    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(len(axes_flat)):
        if i < num_sessions:
            axes_flat[i].imshow(projections[i])
            axes_flat[i].set_xticks([])
            axes_flat[i].set_yticks([])
            axes_flat[i].set_title(f'Session {i + 1}', fontsize=14, fontweight='bold')
            if i == 0:
                h_img = projections[i].shape[0]
                w_img = projections[i].shape[1]
                axes_flat[i].text(0.01 * w_img, 0.02 * h_img, 'Detected in',
                                  fontsize=14, color='g', fontweight='bold')
                axes_flat[i].text(0.01 * w_img, 0.06 * h_img, 'all sessions',
                                  fontsize=14, color='g', fontweight='bold')
        else:
            axes_flat[i].axis('off')

    if "4" in stage_label:
        fname = f"{stage_label} - projections - initial registration"
    else:
        fname = f"{stage_label} - projections - final registration"

    savefig_both(fig, os.path.join(out_dir, fname), also_pdf=also_pdf)
    if not show:
        plt.close(fig)


def plot_pairwise_session_overlap(
    spatial_footprints,   # list of arrays (n_cells, h, w) per session
    cell_to_index_map,    # (n_clusters, n_sessions) — 1-indexed
    out_dir: str,
    show: bool = False,
    also_pdf: bool = False,
) -> None:
    """
    Pairwise session overlap matrix.

    Creates an N×N grid of subplots. For each pair (i, j):
      - Red channel  = cells present in session i (row)
      - Green channel = cells present in session j (col)
      - Yellow = overlap (cells registered across both sessions)
    
    Diagonal panels show ALL cells for that session in white.
    
    A summary count annotation shows how many cells are shared.
    """
    if spatial_footprints is None or cell_to_index_map is None:
        return

    # Convert to list of 3D arrays
    if isinstance(spatial_footprints, np.ndarray) and spatial_footprints.dtype == object:
        fp_list = [np.asarray(e) for e in spatial_footprints.tolist()]
    elif isinstance(spatial_footprints, (list, tuple)):
        fp_list = [np.asarray(e) for e in spatial_footprints]
    else:
        arr = np.asarray(spatial_footprints)
        if arr.dtype == object:
            fp_list = [np.asarray(e) for e in arr.tolist()]
        else:
            print("Warning: Unknown spatial_footprints format for pairwise plot")
            return

    map_arr = np.asarray(cell_to_index_map)  # (n_clusters, n_sessions)
    n_sessions = len(fp_list)
    if map_arr.shape[1] != n_sessions:
        print(f"Warning: Map sessions ({map_arr.shape[1]}) != footprint sessions ({n_sessions})")
        return

    pixel_weight_threshold = 0.5

    # Pre-normalize all footprints once
    norm_fps = []
    for s in range(n_sessions):
        fps = fp_list[s].astype(float)
        out = np.zeros_like(fps)
        for k in range(fps.shape[0]):
            mx = float(np.max(fps[k]))
            if mx > 0:
                fp = fps[k].copy()
                fp[fp < pixel_weight_threshold * mx] = 0
                out[k] = fp / mx
        norm_fps.append(out)

    # Convert 1-based cell_to_index_map to 0-based:
    #   0 in map = "not present" → -1
    #   positive values → subtract 1 for 0-based indexing
    def _get_0based(s):
        c_idxs = map_arr[:, s].astype(int).copy()
        absent = c_idxs <= 0
        c_idxs -= 1          # shift everything: 1→0, 2→1, etc.
        c_idxs[absent] = -1  # mark absent clusters
        return c_idxs

    sess_idxs = [_get_0based(s) for s in range(n_sessions)]

    h, w = norm_fps[0].shape[1], norm_fps[0].shape[2]

    # Build the grid
    fig, axes = plt.subplots(n_sessions, n_sessions,
                             figsize=(4 * n_sessions, 4 * n_sessions))
    fig.suptitle('Pairwise Session Cell Overlap', fontsize=18, fontweight='bold', y=0.98)

    for i in range(n_sessions):
        for j in range(n_sessions):
            ax = axes[i][j] if n_sessions > 1 else axes

            if i == j:
                # Diagonal: show all cells for this session in white
                all_cells = sess_idxs[i][sess_idxs[i] >= 0]
                proj = np.zeros((h, w), dtype=float)
                if len(all_cells) > 0:
                    proj = np.sum(norm_fps[i][all_cells], axis=0)
                    proj = np.clip(proj, 0, 1)
                img = np.stack([proj, proj, proj], axis=-1)
                ax.imshow(img)
                n_total = len(all_cells)
                ax.set_title(f'Session {i+1}\n({n_total} cells)',
                             fontsize=12, fontweight='bold')
            else:
                # Off-diagonal: session i (red) vs session j (green)
                # Find clusters present in BOTH sessions i and j
                both_mask = (sess_idxs[i] >= 0) & (sess_idxs[j] >= 0)
                only_i_mask = (sess_idxs[i] >= 0) & (sess_idxs[j] < 0)
                only_j_mask = (sess_idxs[i] < 0) & (sess_idxs[j] >= 0)

                # Build projection for each category
                proj_both_i = np.zeros((h, w), dtype=float)
                proj_both_j = np.zeros((h, w), dtype=float)
                proj_only_i = np.zeros((h, w), dtype=float)
                proj_only_j = np.zeros((h, w), dtype=float)

                # Shared cells — use footprints from both sessions
                shared_i_cells = sess_idxs[i][both_mask]
                shared_j_cells = sess_idxs[j][both_mask]
                if len(shared_i_cells) > 0:
                    proj_both_i = np.sum(norm_fps[i][shared_i_cells], axis=0)
                if len(shared_j_cells) > 0:
                    proj_both_j = np.sum(norm_fps[j][shared_j_cells], axis=0)

                # Unique to session i
                only_i_cells = sess_idxs[i][only_i_mask]
                if len(only_i_cells) > 0:
                    proj_only_i = np.sum(norm_fps[i][only_i_cells], axis=0)

                # Unique to session j
                only_j_cells = sess_idxs[j][only_j_mask]
                if len(only_j_cells) > 0:
                    proj_only_j = np.sum(norm_fps[j][only_j_cells], axis=0)

                # Compose RGB:
                #   Shared cells → yellow (both R and G)
                #   Only session i → red
                #   Only session j → green
                shared_proj = np.clip(proj_both_i + proj_both_j, 0, 1)
                r_ch = np.clip(proj_only_i + shared_proj, 0, 1)
                g_ch = np.clip(proj_only_j + shared_proj, 0, 1)
                b_ch = np.zeros((h, w), dtype=float)

                img = np.stack([r_ch, g_ch, b_ch], axis=-1)
                ax.imshow(img)

                n_shared = int(both_mask.sum())
                n_only_i = int(only_i_mask.sum())
                n_only_j = int(only_j_mask.sum())
                ax.set_title(f'S{i+1} vs S{j+1}\n'
                             f'{n_shared} shared',
                             fontsize=11, fontweight='bold')
                # Legend text in corner
                ax.text(3, h - 8, f'S{i+1} only: {n_only_i}',
                        fontsize=9, color='red', fontweight='bold',
                        va='bottom')
                ax.text(3, h - 22, f'S{j+1} only: {n_only_j}',
                        fontsize=9, color='lime', fontweight='bold',
                        va='bottom')

            ax.set_xticks([])
            ax.set_yticks([])

            # Row/col labels on edges
            if j == 0:
                ax.set_ylabel(f'Session {i+1}', fontsize=13, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig_both(fig, os.path.join(out_dir, "Stage 5 - pairwise session overlap"), also_pdf=also_pdf)
    if not show:
        plt.close(fig)


def plot_session_projections(
    footprints_projections: list,
    out_dir: str,
    show: bool = False,
    also_pdf: bool = False
) -> None:
    """
    Plots grayscale projections of all sessions (Stage 1).
    Exact replica of MATLAB plot_session_projections
    (batchRunCellReg_ULTIMATE.m lines 770-809).
    """
    if footprints_projections is None or len(footprints_projections) == 0:
        return

    num_sessions = len(footprints_projections)
    subx = 4
    suby = int(np.ceil(num_sessions / subx))

    if num_sessions <= 4:
        fig, axes = plt.subplots(1, num_sessions, figsize=(4 * num_sessions, 5))
        if num_sessions == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(suby, subx, figsize=(4 * subx, 4 * suby))

    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(len(axes_flat)):
        if i < num_sessions:
            proj = np.asarray(footprints_projections[i], dtype=float)
            axes_flat[i].imshow(proj, cmap='gray', vmin=0, vmax=2)
            axes_flat[i].set_xticks([])
            axes_flat[i].set_yticks([])
            axes_flat[i].set_title(f'Session {i + 1}', fontsize=14, fontweight='bold')
        else:
            axes_flat[i].axis('off')

    savefig_both(fig, os.path.join(out_dir, "Stage 1 - spatial footprints projections"), also_pdf=also_pdf)
    if not show:
        plt.close(fig)


def plot_init_registration(
    cell_to_index_map: np.ndarray,
    number_of_bins: int,
    spatial_footprints,
    initial_registration_type: str,
    registered_cells: np.ndarray,
    non_registered_cells: np.ndarray,
    microns_per_pixel: float = 1.0,
    maximal_distance: float = 10.0,
    out_dir: str = ".",
    show: bool = False,
    also_pdf: bool = False
) -> None:
    """
    Plots initial registration results (Stage 4).
    Exact replica of MATLAB plot_init_registration
    (batchRunCellReg_ULTIMATE.m lines 1310-1378).
    """
    fig = plt.figure(figsize=(8, 6))

    if initial_registration_type.lower().startswith('spatial'):
        # Spatial correlation histogram
        xout = np.linspace(0, 1, number_of_bins)
        step = xout[1] - xout[0] if len(xout) > 1 else 0.05
        edges = np.concatenate([xout - step / 2, [xout[-1] + step / 2]])

        n1, _ = np.histogram(registered_cells, bins=edges)
        n2, _ = np.histogram(non_registered_cells, bins=edges)

        bar_offset = 0.25 / number_of_bins
        bar_width = 2 * bar_offset

        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        ax.bar(xout + bar_offset, n1, width=bar_width,
               color='g', edgecolor='none', label='Same Cell')
        ax.bar(xout - bar_offset, n2, width=bar_width,
               color='r', edgecolor='none', label='Different Cells')
        ax.set_xlim(0, 1)
        xtk = np.linspace(0, 1, 6)
        ax.set_xticks(xtk)
        ax.set_xticklabels([f'{v:.1f}' for v in xtk], fontsize=14)
        ax.set_xlabel('Spatial correlation', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper left', frameon=False)
    else:
        # Centroid distance histogram
        xout = np.linspace(0, maximal_distance, number_of_bins)
        step = xout[1] - xout[0] if len(xout) > 1 else 1.0
        edges = np.concatenate([xout - step / 2, [xout[-1] + step / 2]])

        n1, _ = np.histogram(registered_cells, bins=edges)
        n2, _ = np.histogram(non_registered_cells, bins=edges)

        bar_offset = 0.25 * microns_per_pixel * maximal_distance / number_of_bins
        bar_width = 2 * bar_offset

        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        ax.bar(microns_per_pixel * xout + bar_offset, n1, width=bar_width,
               color='g', edgecolor='none', label='Same Cell')
        ax.bar(microns_per_pixel * xout - bar_offset, n2, width=bar_width,
               color='r', edgecolor='none', label='Different Cells')
        ax.set_xlim(0, microns_per_pixel * maximal_distance)
        xtk = np.arange(0, microns_per_pixel * maximal_distance + 1, 3)
        ax.set_xticks(xtk)
        ax.set_xticklabels(xtk.astype(int), fontsize=14)
        ax.set_xlabel('Centroids distance (µm)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Number of cell-pairs', fontweight='bold', fontsize=14)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper left', frameon=False)

    savefig_both(fig, os.path.join(out_dir, "Stage 4 - same versus different cells"), also_pdf=also_pdf)
    if not show:
        plt.close(fig)

    # Also plot initial projections
    plot_all_registered_projections(
        spatial_footprints, cell_to_index_map, out_dir,
        show=show, also_pdf=also_pdf, stage_label="Stage 4"
    )

        
# ============================================================================ #
#                     PART B/C Deck Function                                   #
# ============================================================================ #

def compute_models_from_footprints(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes probabilistic models from footprints if model data is missing.
    Updates d in-place and returns it.
    """
    if not _HAVE_PYSPELL:
        return d
    
    # Check if we need to compute
    if 'centroid_distances_model_parameters' in d:
        return d 
        
    print("\n--- Computing probabilistic models from footprints (PySpell) ---")
    
    # Get footprints
    fps = d.get('spatial_footprints')
    if fps is None:
        print("Cannot compute models: missing 'spatial_footprints'")
        return d
        
    # Convert to list of arrays
    if isinstance(fps, np.ndarray) and fps.dtype == object:
        fps_list = [np.asarray(e) for e in fps.tolist()]
    elif isinstance(fps, (list, tuple)):
        fps_list = [np.asarray(e) for e in fps]
    elif isinstance(fps, np.ndarray) and fps.ndim == 3:
         # Single session? Or stack?
         # Assume list of sessions
         # But usually fps is list of (N, H, W)
         print("Warning: spatial_footprints is a 3D array, assuming it's a list error. Skipping.")
         return d
    else:
        # Try best effort
        try:
             fps_list = [np.asarray(e) for e in fps]
        except:
             return d
             
    # 1. Centroids
    microns_per_pixel = float(d.get('microns_per_pixel', 2.0))
    centroids = d.get('centroid_locations_corrected')
    
    if centroids is None:
        centroids = compute_centroids(fps_list, microns_per_pixel)
    elif isinstance(centroids, np.ndarray) and centroids.dtype == object:
         centroids = [np.asarray(e) for e in centroids.tolist()] 
         
    # 2. Data Distribution
    max_dist = float(d.get('maximal_distance', 12.0))
    max_dist_px = max_dist / microns_per_pixel
    
    data_dist = compute_data_distribution(fps_list, centroids, max_dist_px)
    d.update(data_dist) # Merge neighbors_x_displacements etc.
    
    # 3. Models
    n_bins, centers = estimate_num_bins(fps_list, max_dist_px)
    d['centers_of_bins'] = centers # Tuple
    
    # Centroid Model
    (p_same_grad_cent, same_model_cent, diff_model_cent, mix_model_cent, 
     int_cent, best_str_cent, mse_cent) = compute_centroid_distances_model_custom(
         data_dist['neighbors_centroid_distances'], n_bins, centers
    )
    
    d['centroid_distances_model_parameters'] = [0.5] # Placeholder or parse from string? 
    # Actually plot_models uses [0] as p_same.
    # best_str_cent has "p=..." 
    # Let's extract p from best_str manually or just pass it differently?
    # validation script expects d['centroid_distances_model_parameters'] to be an array where [0] is p_same.
    
    # Parse p from string "p=0.xxx, ..."
    try:
        p_val = float(best_str_cent.split(',')[0].split('=')[1])
    except:
        p_val = 0.5
        
    d['centroid_distances_model_parameters'] = np.array([p_val])
    d['centroid_distances_model_same_cells'] = same_model_cent
    d['centroid_distances_model_different_cells'] = diff_model_cent
    d['centroid_distances_model_weighted_sum'] = mix_model_cent
    d['centroid_distance_intersection'] = int_cent
    
    # Compute distribution for plotting (normalized density to match model)
    # Matches cellregpy.py / MATLAB logic
    counts, _ = np.histogram(data_dist['neighbors_centroid_distances'], 
                             bins=np.r_[centers[0], centers[0][-1] + (centers[0][1]-centers[0][0])])
    counts = counts.astype(float)
    if counts.sum() > 0:
        dist = counts / (counts.sum() + 1e-12)
        # Scaling matching compute_centroid_distances_model_custom
        dist = dist / (dist.max() + 1e-12)
        dist = dist * n_bins / (centers[0][-1] - centers[0][0] + 1e-12)
        d['centroid_distances_distribution'] = dist
    else:
        d['centroid_distances_distribution'] = counts
    
    # Spatial Model
    (p_same_grad_corr, same_model_corr, diff_model_corr, mix_model_corr,
     int_corr, best_str_corr, mse_corr) = compute_spatial_correlations_model(
         data_dist['neighbors_spatial_correlations'], n_bins, centers
     )
     
    try:
        p_val_corr = float(best_str_corr.split(',')[0].split('=')[1])
    except:
        p_val_corr = 0.5
        
    d['spatial_correlations_model_parameters'] = np.array([p_val_corr])
    d['spatial_correlations_model_same_cells'] = same_model_corr
    d['spatial_correlations_model_different_cells'] = diff_model_corr
    d['spatial_correlations_model_weighted_sum'] = mix_model_corr
    d['spatial_correlation_intersection'] = int_corr
    
    # Normalize spatial
    counts_s, _ = np.histogram(data_dist['neighbors_spatial_correlations'],
                               bins=np.r_[centers[1], centers[1][-1] + (centers[1][1]-centers[1][0])])
    counts_s = counts_s.astype(float)
    if counts_s.sum() > 0:
        dist_s = counts_s / (counts_s.sum() + 1e-12)
        dist_s = dist_s / (dist_s.max() + 1e-12)
        dist_s = dist_s * n_bins / (centers[1][-1] - centers[1][0] + 1e-12)
        d['spatial_correlations_distribution'] = dist_s
    else:
        d['spatial_correlations_distribution'] = counts_s
                                                          
    # 4. P_same and Clustering (Scores)
    p_same_cent, p_same_corr = compute_p_same(
        data_dist['all_to_all_centroid_distances'],
        data_dist['all_to_all_spatial_correlations'],
        centers,
        p_same_grad_cent,
        p_same_grad_corr
    )
    
    # Choose model
    model_type = d.get('model_type', 'auto')
    best_model = choose_best_model(mse_cent, mse_corr, 
                                   centroid_intersection=int_cent, 
                                   corr_intersection=int_corr)
    
    if model_type == 'auto':
        model_used = best_model
    else:
        model_used = model_type
        
    if model_used == 'Centroid distance':
        all_to_all_p_same = p_same_cent
        thresh = int_cent if np.isfinite(int_cent) else max_dist_px
    else:
        all_to_all_p_same = p_same_corr
        thresh = int_corr if np.isfinite(int_corr) else 0.5

    # Initial registration (Part D) - skipped for plotting, but needed for clustering seed?
    # cluster_cells takes cell_to_index_map.
    # We can use the one from input if available!!!!
    # If 'optimal_cell_to_index_map' is in d (from CellReg.mat), we use it to score.
    
    cmap = d.get('optimal_cell_to_index_map')
    if cmap is not None:
        # Just score the existing map
        p_vec, p_diff_vec, scores_vec = estimate_registration_accuracy(
            np.asarray(cmap), all_to_all_p_same, data_dist['all_to_all_indexes']
        )
        
        # Populate scores
        d['cell_scores'] = p_vec # List or array
        d['p_same_registered_pairs'] = p_vec
        
        # We need exclusive/pos/neg.
        # These are usually derived from ground truth or simulations?
        # In batchRunCellReg, cell_scores are P(same) for registered pairs.
        # "True Positive" / "True Negative" scores require ground truth, which we don't have.
        # BUT batchRunCellReg calculates them based on the model overlap?
        # Actually, `cell_scores_positive` etc are specific variables.
        # In batchRunCellReg:
        # cell_scores = p_same_vec
        # cell_scores_positive = p_same_vec (registered)
        # cell_scores_negative = (1-p_same) for NON-registered?
        # cell_scores_exclusive = ...?
        
        # Let's just create 'cell_scores' (Overall) for the plot.
        # The other subplots in `plot_cell_scores` might remain empty if we don't calculate them.
        
        d['cell_scores'] = p_vec
        d['cell_scores_positive'] = p_vec # Registered => Assumed Positive
        
        # To get negatives, we'd need non-registered pairs.
        # We can scan non-registered pairs and get their p_same?
        # This is getting complicated to fully replicate without running full clustering logic.
        # But `estimate_registration_accuracy` returns `p_same_vec` (registered).
        
    return d

def validate_modeling_deck(
    d: Dict[str, Any],
    out_dir: str,
    show: bool = False,
    also_pdf: bool = False
) -> None:
    """runs all the modeling validation plots if keys exist"""
    
    # Try to compute missing models
    try:
        d = compute_models_from_footprints(d)
    except Exception as e:
        print(f"Model computation failed: {e}")
        # traceback.print_exc()
    
    print("Running Modeling Validation Plots...")

    # Parse centers_of_bins early — needed by displacements and models
    cob = d.get('centers_of_bins')
    cob_dist = None
    cob_corr = None
    if cob is not None:
        if len(cob) >= 1: cob_dist = np.asarray(cob[0])
        if len(cob) >= 2: cob_corr = np.asarray(cob[1])
    
    # 1. Displacements

    if 'neighbors_x_displacements' in d and 'neighbors_y_displacements' in d:
        try:
             # Extract scalars
             mpp = float(d.get('microns_per_pixel', 2.0))
             # If max_dist in d, use it
             max_dist = float(d.get('maximal_distance', 12.0)) # Default 12
             n_bins = int(str(d.get('number_of_bins', 40)))  # Default 40
             
             plot_x_y_displacements(
                 d['neighbors_x_displacements'], d['neighbors_y_displacements'],
                 mpp, max_dist, n_bins,
                 cob if cob is not None else [np.linspace(0, max_dist, n_bins)],
                 out_dir, show, also_pdf
             )
        except Exception as e:
            print(f"Error plotting displacements: {e}")
            
    # 2. Models
    # Required keys for models
    req_model = ['centroid_distances_model_parameters', 'NN_centroid_distances', 'NNN_centroid_distances',
                 'centroid_distances_distribution', 'centroid_distances_model_same_cells',
                 'centroid_distances_model_different_cells', 'centroid_distances_model_weighted_sum',
                 'centers_of_bins'] 
    if all(k in d for k in req_model) and cob_dist is not None:
        try:
            plot_models(
                d['centroid_distances_model_parameters'],
                d['NN_centroid_distances'], d['NNN_centroid_distances'],
                d['centroid_distances_distribution'],
                d['centroid_distances_model_same_cells'], d['centroid_distances_model_different_cells'],
                d['centroid_distances_model_weighted_sum'],
                d.get('centroid_distance_intersection', 0),
                cob_dist,
                # Spatial
                d.get('spatial_correlations_model_parameters'),
                d.get('NN_spatial_correlations'), d.get('NNN_spatial_correlations'),
                d.get('spatial_correlations_distribution'),
                d.get('spatial_correlations_model_same_cells'), d.get('spatial_correlations_model_different_cells'),
                d.get('spatial_correlations_model_weighted_sum'),
                d.get('spatial_correlation_intersection'),
                cob_corr,
                # Opts
                float(d.get('microns_per_pixel', 2.0)),
                float(d.get('maximal_distance', 12.0)),
                out_dir, show, also_pdf
            )
        except Exception as e:
             print(f"Error plotting models: {e}")
    else:
        print("Skipping plot_models (missing keys)")

    # 3. Scores
    req_scores = ['cell_scores', 'cell_scores_exclusive', 'cell_scores_positive', 'cell_scores_negative']
    if all(k in d for k in req_scores):
        try:
             plot_cell_scores(
                 d['cell_scores'], d['cell_scores_exclusive'], d['cell_scores_positive'], d['cell_scores_negative'],
                 d.get('p_same_registered_pairs'),
                 out_dir, show, also_pdf
             )
        except Exception as e:
            print(f"Error plotting scores: {e}")

    # 4. Projections
    if 'spatial_footprints' in d and 'optimal_cell_to_index_map' in d: # Use optimal map
         try:
             plot_all_registered_projections(
                 d['spatial_footprints'], d['optimal_cell_to_index_map'],
                 out_dir, show, also_pdf
             )
         except Exception as e:
             print(f"Error plotting final projections: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Alignment validation deck (ULTIMATE)")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to .npz or .mat with mean_images / footprints")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output directory for figures")
    ap.add_argument("--ref", dest="ref", type=int, default=1, help="Reference session index (1-based by default)")
    ap.add_argument("--zero-based", action="store_true", help="Interpret --ref as 0-based")
    ap.add_argument("--show", action="store_true", help="Display figures interactively")
    ap.add_argument("--pdf", action="store_true", help="Also save PDF copies")

    args = ap.parse_args()

    d = load_inputs(args.in_path)

    # Part A keys
    mean_images = _pick_key(d, ["mean_images", "meanImages", "mean_imgs", "meanImgs"], required=False)
    fp_raw = _pick_key(d, ["footprints_proj_raw", "footprints_raw", "footprintsProjRaw", "fp_raw"], required=False)
    fp_aligned = _pick_key(d, ["footprints_proj_aligned", "footprints_aligned", "footprintsProjAligned", "fp_aligned"], required=False)

    alignment_translations = _pick_key(d, ["alignment_translations", "translations", "alignmentTranslations", "align_trans"], required=False)
    scores = _pick_key(d, ["scores", "maximal_cross_correlation", "max_cross_corr", "max_corr"], required=False)
    session_names = _pick_key(d, ["session_names", "sessionNames", "sessions"], required=False)

    ref = args.ref
    if not args.zero_based:
        ref = ref - 1

    # Run Part A if data exists
    if mean_images is not None and fp_raw is not None:
        try:
            validate_alignment_deck(
                mean_images,
                fp_raw,
                fp_aligned,
                reference_session_index=ref,
                alignment_translations=alignment_translations,
                scores=scores,
                out_dir=args.out_dir,
                session_names=session_names,
                show=args.show,
                also_pdf=args.pdf,
            )
        except Exception as e:
            print(f"Error running alignment deck: {e}")
    else:
        print("Skipping Alignment Deck (missing mean_images or footprints_proj_raw)")

    # Run Part B
    validate_modeling_deck(d, args.out_dir, args.show, args.pdf)

if __name__ == "__main__":
    folder_path = r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_superalignment\L612_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN";  
    main()

