# Diagnostic script: Visualize preprocessing steps and their effect on deconvolution
# Use: %matplotlib qt in IPython/Jupyter for interactive plots

import sys
from pathlib import Path

# Get the pyspell directory - works in PyCharm, Spyder, IPython, and command line
try:
    # When running as a script
    script_dir = Path(__file__).resolve().parent  # .../pyspell/scripts
    pyspell_dir = script_dir.parent               # .../pyspell
except NameError:
    # When running in IPython/Jupyter interactive mode
    pyspell_dir = Path.cwd()
sys.path.insert(0, str(pyspell_dir))

import numpy as np
import matplotlib.pyplot as plt
import deconvolution as dc
from s2pfuns import (
    postProcess,
    sgolay_detrend, percentile_detrend, highpass_detrend, als_detrend, correct_spike_drift_model
)
from scipy.stats import linregress


# ----------------------------
# Matplotlib interactive mode
# ----------------------------
try:
    ip = get_ipython()  # noqa
    if ip is not None:
        ip.run_line_magic("matplotlib", "qt")
except Exception:
    pass

# =========================================================================
# Load data
# =========================================================================
self = postProcess(
    s2ppath=r"X:\John\Subjects - GCaMP Recordings\L612_F_RightPFC_L6Chrimson_PFCgcamp8f_Panrec\SDswitch_day9_FOV1_optoRec\SDswitch_day9_FOV1_optoRec_img\suite2p\plane0"
)

# Options: 'sgolay', 'percentile', 'highpass', 'als', 'combined'
detrend_method = 'percentile'  # <-- CHANGE THIS TO TEST DIFFERENT METHODS
fr = 7.5

# Drift correction knobs
DRIFT_MODEL = "auto"           # "auto", "exp", "linear"
DRIFT_R2_THRESH = 0.10         # apply only if >= 0.10
DRIFT_CLIP = (0.2, 5.0)        # more aggressive than (0.3,3.0)
DRIFT_TOPQ = 0.6               # raise to 0.7-0.8 if floor effects dominate

# =========================================================================
# Loop neurons
# =========================================================================
for neuron_idx in range(self.C.shape[0]):

    ar_model = 2

    # neuropil correct
    Fc = self.F[neuron_idx, :] - self.Fneu[neuron_idx, :]

    # APPLY SELECTED DETRENDING METHOD
    if detrend_method == 'sgolay':
        f_processed, trend = sgolay_detrend(Fc, fr=fr, window_size=1001, add_median_back=True)
        method_label = "Sgolay (win=1001)"

    elif detrend_method == 'percentile':
        # First apply sgolay, then percentile to catch remaining wiggle
        f_temp, _ = sgolay_detrend(Fc, fr=fr, window_size=1001, add_median_back=True)
        f_processed, trend = percentile_detrend(f_temp, fr=fr, window_sec=30, percentile=8, add_median_back=True)
        method_label = "Sgolay + Percentile (30s, 8%)"

    elif detrend_method == 'highpass':
        f_processed, trend = highpass_detrend(Fc, fr=fr, cutoff_hz=0.005, add_median_back=True)
        method_label = "High-pass (0.005 Hz)"

    elif detrend_method == 'als':
        f_processed, trend = als_detrend(Fc, lam=1e8, p=0.005, add_median_back=True)
        method_label = "ALS (lam=1e8, p=0.005)"

    elif detrend_method == 'combined':
        f_temp, _ = sgolay_detrend(Fc, fr=fr, window_size=1001, add_median_back=True)
        f_processed, trend = percentile_detrend(f_temp, fr=fr, window_sec=20, percentile=5, add_median_back=True)
        method_label = "Combined (sgolay + percentile 20s, 5%)"

    else:
        f_processed, trend = sgolay_detrend(Fc, fr=fr, window_size=1001, add_median_back=True)
        method_label = "Sgolay (default)"

    # =============================================================================
    # TWO-PASS OPTIMIZATION TESTING
    # Compare: (1) baseline, (2) optimize_g, (3) refined sn, (4) s_min threshold
    # =============================================================================

    # --- RUN 1: Baseline with optimize_g=0 ---
    print("=== RUN 1: Baseline (optimize_g=0) ===")
    c1_run, bl1, c1_init, g1, sn1, sp1, lam1 = dc.constrained_foopsi(
        f_processed, p=ar_model, method_deconvolution='oasis', bas_nonneg=True,
        noise_range=[0.25, .5], noise_method='logmexp', sn=None,
        lags=5, fudge_factor=1.0, solvers=None, verbosity=True, s_min=None,
        optimize_g=0
    )
    num_spikes_run1 = int(np.sum(sp1 > 0))
    print(f"  Spikes detected: {num_spikes_run1}, sn={sn1:.4f}, g={g1}")

    # --- RUN 2: Optimize g using 10% of detected spikes ---
    print("\n=== RUN 2: Optimize g (using 10% of Run 1 spikes) ===")
    optimized_g = max(1, int(num_spikes_run1 * 0.1))  # at least 1
    c2_run, bl2, c2_init, g2, sn2, sp2, lam2 = dc.constrained_foopsi(
        f_processed, p=ar_model, method_deconvolution='oasis', bas_nonneg=True,
        noise_range=[0.25, .5], noise_method='logmexp', sn=None,
        lags=5, fudge_factor=1.0, solvers=None, verbosity=True, s_min=None,
        optimize_g=optimized_g
    )
    num_spikes_run2 = int(np.sum(sp2 > 0))
    print(f"  Spikes detected: {num_spikes_run2}, sn={sn2:.4f}, g={g2}")

    # --- Refine sn from residuals of Run 2 ---
    print("\n=== Refining noise estimate from residuals ===")

    c2_rescaled = c2_run * (np.std(f_processed) / (np.std(c2_run) + 1e-10)) + np.mean(f_processed) - np.mean(c2_run)
    residuals = f_processed - c2_rescaled

    residuals = residuals[np.abs(residuals) < 3 * np.std(residuals)]
    if len(residuals) > 10:
        mad0 = np.median(np.abs(residuals - np.median(residuals))) / 0.6745
        if mad0 > 0:
            residuals = residuals[np.abs(residuals - np.median(residuals)) < 3 * mad0]

    if len(residuals) > 5:
        sn_refined = np.median(np.abs(residuals - np.median(residuals))) / 0.6745
    else:
        sn_refined = sn2
    print(f"  sn from Run 2: {sn2:.4f}")
    print(f"  sn refined (MAD): {sn_refined:.4f}")

    # --- Compute s_min threshold from spike distribution ---
    spike_amplitudes = sp2[sp2 > 0]
    if len(spike_amplitudes) > 5:
        s_min_threshold = float(np.percentile(spike_amplitudes, 10))
        print(f"  s_min threshold (10th pct): {s_min_threshold:.4f}")
    else:
        s_min_threshold = 0.0
        print("  Not enough spikes, using s_min=0")

    # --- RUN 3: Final run with refined sn, s_min, and optimized g ---
    print("\n=== RUN 3: Final (refined sn + s_min + optimized g) ===")
    c3_run, bl3, c3_init, g3, sn3, sp3, lam3 = dc.constrained_foopsi(
        f_processed, p=ar_model, method_deconvolution='oasis', bas_nonneg=True,
        noise_range=[0.25, .5], noise_method='logmexp', sn=sn_refined,
        lags=5, fudge_factor=1.0, solvers=None, verbosity=True, s_min=s_min_threshold,
        optimize_g=optimized_g, g=g2
    )
    num_spikes_run3 = int(np.sum(sp3 > 0))
    print(f"  Spikes detected: {num_spikes_run3}")

    # --- Helper to scale C for plotting ---
    def scale_c(c_trace, f_ref):
        return c_trace * (np.std(f_ref) / (np.std(c_trace) + 1e-10)) + np.mean(f_ref) - np.mean(c_trace)

    # --- Normalize spike train by MAD (residual in same scale as f_processed) ---
    c3_rescaled = c3_run * (np.std(f_processed) / (np.std(c3_run) + 1e-10)) + np.mean(f_processed) - np.mean(c3_run)
    resid = f_processed - c3_rescaled
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    sp3_norm = sp3 / mad

    # =========================================================================
    # DRIFT CORRECTION (linear vs exponential)
    # =========================================================================
    print("\n=== Spike Drift Correction (linear vs exp) ===")
    sp_final, drift_info = correct_spike_drift_model(
        sp3_norm,
        r2_thresh=DRIFT_R2_THRESH,
        model=DRIFT_MODEL,
        clip=DRIFT_CLIP,
        t_ref_mode="median",
        fit_top_quantile=DRIFT_TOPQ,
        robust=True
    )

    print(
        f"  chosen={drift_info.get('model')}, "
        f"R²_lin={drift_info.get('r2_lin', 0):.3f}, "
        f"R²_exp={drift_info.get('r2_exp', 0):.3f}, "
        f"R²_used={drift_info.get('r2_used', 0):.3f}"
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Spike counts across runs")
    print("=" * 60)
    print(f"  Run 1 (baseline):           {num_spikes_run1} spikes")
    print(f"  Run 2 (optimize_g):         {num_spikes_run2} spikes")
    print(f"  Run 3 (sn + s_min + g):     {num_spikes_run3} spikes")
    print(f"  Final (drift-corrected):    {int(np.sum(sp_final > 0))} spikes")

    # =========================================================================
    # PLOT 1: 2x2 Deconvolution Comparison
    # =========================================================================
    # Color palette (seaborn-inspired)
    C_BLUE = '#4C72B0'
    C_ORANGE = '#DD8452'
    C_GREEN = '#55A868'
    C_PURPLE = '#8172B3'
    C_GRAY = '#777777'

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    frames = np.arange(len(f_processed))

    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Run 1: Baseline
    ax = axes[0, 0]
    ax.plot(frames, f_processed, color=C_GRAY, alpha=0.5, linewidth=0.5, label='F (processed)')
    ax.plot(frames, scale_c(c1_run, f_processed), color=C_BLUE, alpha=0.9, linewidth=1, label='C (fitted)')
    ax_twin = ax.twinx()
    ax_twin.fill_between(frames, 0, sp1, color=C_BLUE, alpha=0.25, label='Spikes')
    ax_twin.set_ylabel('Spikes', color=C_BLUE, fontsize=10)
    ax_twin.tick_params(axis='y', colors=C_BLUE)
    ax.set_title(f'Run 1: Baseline — {num_spikes_run1} spikes', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fluorescence', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Run 2: Optimize g
    ax = axes[0, 1]
    ax.plot(frames, f_processed, color=C_GRAY, alpha=0.5, linewidth=0.5, label='F (processed)')
    ax.plot(frames, scale_c(c2_run, f_processed), color=C_ORANGE, alpha=0.9, linewidth=1, label='C (fitted)')
    ax_twin = ax.twinx()
    ax_twin.fill_between(frames, 0, sp2, color=C_ORANGE, alpha=0.25)
    ax_twin.set_ylabel('Spikes', color=C_ORANGE, fontsize=10)
    ax_twin.tick_params(axis='y', colors=C_ORANGE)
    ax.set_title(f'Run 2: Optimize g={optimized_g} — {num_spikes_run2} spikes', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fluorescence', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Run 3: Final OASIS
    ax = axes[1, 0]
    ax.plot(frames, f_processed, color=C_GRAY, alpha=0.5, linewidth=0.5, label='F (processed)')
    ax.plot(frames, scale_c(c3_run, f_processed), color=C_GREEN, alpha=0.9, linewidth=1, label='C (fitted)')
    ax_twin = ax.twinx()
    ax_twin.fill_between(frames, 0, sp3, color=C_GREEN, alpha=0.25)
    ax_twin.set_ylabel('Spikes', color=C_GREEN, fontsize=10)
    ax_twin.tick_params(axis='y', colors=C_GREEN)
    ax.set_title(f'Run 3: Refined (sn={sn_refined:.3f}) — {num_spikes_run3} spikes', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fluorescence', fontsize=10)
    ax.set_xlabel('Frame', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Final: Drift-corrected
    ax = axes[1, 1]
    ax.plot(frames, f_processed, color=C_GRAY, alpha=0.5, linewidth=0.5, label='F (processed)')
    ax.plot(frames, scale_c(c3_run, f_processed), color=C_PURPLE, alpha=0.9, linewidth=1, label='C (fitted)')
    ax_twin = ax.twinx()
    ax_twin.fill_between(frames, 0, sp3_norm, color=C_GRAY, alpha=0.2, label='Before corr')
    ax_twin.fill_between(frames, 0, sp_final, color=C_PURPLE, alpha=0.35, label='After corr')
    ax_twin.set_ylabel('Spikes (MAD norm)', color=C_PURPLE, fontsize=10)
    ax_twin.tick_params(axis='y', colors=C_PURPLE)
    ax_twin.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(f'Final: Drift-Corrected — {int(np.sum(sp_final > 0))} spikes', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fluorescence', fontsize=10)
    ax.set_xlabel('Frame', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    fig.suptitle(f'Neuron {neuron_idx}: {method_label}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # =========================================================================
    # PLOT 2: Drift Correction Diagnostics
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))

    spike_times = np.flatnonzero(sp3_norm > 0)
    spike_amps_before = sp3_norm[spike_times]
    spike_amps_after = sp_final[spike_times]
    eps = 1e-12

    for ax_row in axes2:
        for ax in ax_row:
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Filter to top-quantile spikes (same as function uses for threshold decision)
    q_thresh = np.quantile(spike_amps_before, DRIFT_TOPQ) if len(spike_amps_before) > 0 else 0
    top_mask = spike_amps_before >= q_thresh
    top_times = spike_times[top_mask]
    top_amps_before = spike_amps_before[top_mask]
    top_amps_after = spike_amps_after[top_mask]

    # Use R² from drift_info (matches threshold decision)
    r2_lin = drift_info.get('r2_lin', 0)
    r2_exp = drift_info.get('r2_exp', 0)

    # Panel 1: Linear fit (before) - fit on TOP spikes only
    ax = axes2[0, 0]
    # Show all spikes faded, top spikes highlighted
    ax.scatter(spike_times, spike_amps_before, c=C_GRAY, alpha=0.3, s=15, edgecolors='none', label='All spikes')
    ax.scatter(top_times, top_amps_before, c=C_BLUE, alpha=0.7, s=25, edgecolors='none', label=f'Top {int(DRIFT_TOPQ*100)}%')
    if top_times.size >= 3:
        sl, it, r, _, _ = linregress(top_times.astype(float), top_amps_before.astype(float))
        ax.plot(top_times, it + sl * top_times, color='#C44E52', linestyle='--', linewidth=2, 
                label=f'Linear fit (R²={r2_lin:.3f})')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.set_title(f'Before: Amplitude vs Time (R²={r2_lin:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Spike Amplitude', fontsize=10)

    # Panel 2: Exponential fit (before) - fit on TOP spikes only
    ax = axes2[0, 1]
    ax.scatter(spike_times, np.log(spike_amps_before + eps), c=C_GRAY, alpha=0.3, s=15, edgecolors='none', label='All spikes')
    ax.scatter(top_times, np.log(top_amps_before + eps), c=C_ORANGE, alpha=0.7, s=25, edgecolors='none', label=f'Top {int(DRIFT_TOPQ*100)}%')
    if top_times.size >= 3:
        sl, it, r, _, _ = linregress(top_times.astype(float), np.log(top_amps_before + eps).astype(float))
        ax.plot(top_times, it + sl * top_times, color='#C44E52', linestyle='--', linewidth=2, 
                label=f'Exp fit (R²={r2_exp:.3f})')
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.set_title(f'Before: log(Amp) vs Time (R²={r2_exp:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('log(Spike Amplitude)', fontsize=10)

    # Panel 3: Correction factor
    ax = axes2[1, 0]
    corr_all = drift_info.get("corr_all", np.ones_like(sp3_norm))
    model_used = drift_info.get('model', 'none')
    ax.plot(corr_all, color=C_GREEN, linewidth=1.5, label='Correction factor')
    ax.axhline(1.0, color=C_GRAY, linestyle='--', linewidth=1, alpha=0.7, label='No correction')
    if model_used != 'none':
        ax.fill_between(range(len(corr_all)), 1.0, corr_all, alpha=0.2, color=C_GREEN)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    model_label = f'{model_used} fit → 1/{model_used}' if model_used != 'none' else 'No correction applied'
    ax.set_title(f'Correction Factor ({model_label})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Multiplier', fontsize=10)

    # Panel 4: After correction (or indicate no change)
    ax = axes2[1, 1]
    if model_used == 'none':
        # No correction - show same data with note
        ax.scatter(spike_times, spike_amps_before, c=C_GRAY, alpha=0.6, s=20, edgecolors='none', label='Spikes (unchanged)')
        ax.text(0.5, 0.5, 'NO CORRECTION\n(R² < threshold)', 
                transform=ax.transAxes, fontsize=14, va='center', ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.set_title('After: No Correction Applied', fontsize=11, fontweight='bold')
    else:
        # Show all spikes faded, top spikes highlighted
        ax.scatter(spike_times, spike_amps_after, c=C_GRAY, alpha=0.3, s=15, edgecolors='none', label='All spikes')
        ax.scatter(top_times, top_amps_after, c=C_PURPLE, alpha=0.7, s=25, edgecolors='none', label=f'Top {int(DRIFT_TOPQ*100)}%')
        if top_times.size >= 3:
            sl, it, r, _, _ = linregress(top_times.astype(float), top_amps_after.astype(float))
            r2_after = r**2
            ax.plot(top_times, it + sl * top_times, color='#C44E52', linestyle='--', linewidth=2, 
                    label=f'Fit (R²={r2_after:.3f})')
        else:
            r2_after = 0
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_title(f'After: Amplitude vs Time (R²={r2_after:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=10)
    ax.set_ylabel('Spike Amplitude', fontsize=10)

    fig2.suptitle(f'Neuron {neuron_idx}: Drift Correction Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)

    # =========================================================================
    # Interactive: Wait for keypress to continue
    # =========================================================================
    print(f"\n[Neuron {neuron_idx}] Explore the figures. Press 'n' or 'space' to continue, 'q' to quit.")
    print(f'Neuron_idx={neuron_idx}, iscell={self.iscell[neuron_idx]}')

    state = {'quit': False}

    def on_key(event):
        if event.key in ['n', ' ', 'enter']:
            plt.close(fig)
            plt.close(fig2)
        elif event.key == 'q':
            state['quit'] = True
            plt.close(fig)
            plt.close(fig2)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)  # Block until figure is closed

    if state['quit']:
        print("Exiting loop early by user request.")
        break
