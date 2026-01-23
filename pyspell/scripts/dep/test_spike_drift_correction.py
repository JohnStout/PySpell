# Test correct_spike_drift with full deconvolution pipeline on synthetic bleached calcium data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import convolve1d

from s2pfuns import _process_single_cell_foopsi, correct_spike_drift, sgolay_detrend

# === GENERATE SYNTHETIC CALCIUM DATA WITH BLEACHING ===
np.random.seed(42)
n_frames = 10000
fr = 7.5  # 7.5 Hz frame rate

# Time axis
t = np.arange(n_frames) / fr

# Generate spike times (random Poisson-ish, ~100 spikes for better stats)
n_spikes = 100
spike_times = np.sort(np.random.choice(n_frames, size=n_spikes, replace=False))

# Calcium kernel (GCaMP6f-like: fast rise, ~0.7s decay)
tau = 0.7  # decay time constant in seconds
kernel_len = int(5 * tau * fr)
kernel = np.exp(-np.arange(kernel_len) / (tau * fr))
kernel = kernel / np.max(kernel)  # Normalize

# Ground truth spikes with varied magnitude (realistic)
sp_true = np.zeros(n_frames)
sp_true[spike_times] = np.random.uniform(0.5, 1.5, size=n_spikes)  # Variable spike sizes

# Convolve to get calcium transients
c_true = convolve1d(sp_true, kernel, mode='constant')

# === ADD PHOTOBLEACHING ===
# 30% decay over recording 
bleach_decay = 0.3
bleach_factor = 1 - bleach_decay * (np.arange(n_frames) / n_frames)

# Apply bleaching - signal amplitude much larger relative to noise
baseline = 500  # Higher baseline like real data
signal_amplitude = 200  # Strong transients
f_clean = baseline * bleach_factor + c_true * signal_amplitude * bleach_factor

# Add realistic noise (smaller relative to signal)
noise_level = 10  # ~2% of baseline
noise = np.random.randn(n_frames) * noise_level
F = f_clean + noise

# Create neuropil trace (correlated with bleaching)
Fneu = baseline * 0.3 * bleach_factor + np.random.randn(n_frames) * 5

print("=== Synthetic Data Generated ===")
print(f"Frames: {n_frames}, Frame rate: {fr} Hz")
print(f"True spikes: {n_spikes}")
print(f"Bleaching: {bleach_decay*100:.0f}% decay")
print(f"Signal amplitude: {signal_amplitude}, Noise: {noise_level}")

# === RUN DECONVOLUTION WITHOUT SPIKE DRIFT CORRECTION ===
# Run with spike drift correction disabled to get raw spikes
result_no_correct = _process_single_cell_foopsi(
    0, F, Fneu, fr=fr, 
    sgolay_filt=True, neuropil_coeff=1.0, 
    fudge_factor=1.0, noise_range=[0.25, 0.5], 
    estimate_sn=True,
    correct_spk_drift=False  # Disable correction to get raw spikes
)

sp_before = result_no_correct['sp'].copy()

# === APPLY SPIKE DRIFT CORRECTION MANUALLY ===
# Use the raw F trace (before neuropil subtraction for bleaching estimation)
F_raw = F - Fneu
sp_after = correct_spike_drift(sp_before, F_raw=F_raw, fr=fr, reference_method='start')

# === RUN FULL PIPELINE (with correction built-in) ===
result_with_correct = _process_single_cell_foopsi(
    0, F, Fneu, fr=fr, 
    sgolay_filt=True, neuropil_coeff=1.0, 
    fudge_factor=1.0, noise_range=[0.25, 0.5], 
    estimate_sn=True,
    correct_spk_drift=True  # Enable correction
)

print(f"\nDeconvolution success: {result_no_correct['success']}")
print(f"Spikes detected (before correction): {np.sum(sp_before > 0.1 * np.max(sp_before + 1e-10))}")
print(f"Spikes detected (after correction): {np.sum(sp_after > 0.1 * np.max(sp_after + 1e-10))}")

# === ANALYZE DRIFT ===
threshold = 0.1 * np.max(sp_before + 1e-10)
spike_idx_before = np.where(sp_before > threshold)[0]
spike_mag_before = sp_before[spike_idx_before]

spike_idx_after = np.where(sp_after > threshold)[0]
spike_mag_after = sp_after[spike_idx_after]

if len(spike_idx_before) >= 10:
    slope_before, int_before, r_before, _, _ = linregress(spike_idx_before, spike_mag_before)
    slope_after, int_after, r_after, _, _ = linregress(spike_idx_after, spike_mag_after)
    
    print(f"\n=== DRIFT ANALYSIS ===")
    print(f"Before: slope = {slope_before:.6f}, R² = {r_before**2:.4f}")
    print(f"After:  slope = {slope_after:.6f}, R² = {r_after**2:.4f}")
    if abs(slope_before) > 1e-10:
        print(f"Slope reduction: {100*(1 - abs(slope_after/slope_before)):.1f}%")
else:
    print("Not enough spikes for drift analysis")
    slope_before = slope_after = int_before = int_after = r_before = r_after = 0

# === PLOT RESULTS ===
fig, axes = plt.subplots(5, 1, figsize=(16, 14))

# Panel 1: Synthetic bleached fluorescence trace
axes[0].plot(t, F, 'k', alpha=0.7, linewidth=0.5)
axes[0].plot(t, baseline * bleach_factor, 'r--', linewidth=1, label='Baseline decay')
axes[0].set_ylabel('F')
axes[0].set_title(f'Synthetic Bleached Calcium Trace ({bleach_decay*100:.0f}% decay, {n_spikes} true spikes)')
axes[0].legend()

# Panel 2: Deconvolved C trace overlay (with corrected version)
c_trace = result_no_correct['c']
f_raw,_ = sgolay_detrend(F-Fneu,fr=fr)

# Get corrected C using correct_spike_drift
F_raw_for_correction = F - Fneu
c_corrected = correct_spike_drift(c_trace, F_raw=F_raw_for_correction, fr=fr)

axes[1].plot(t, f_raw, 'k', alpha=0.4, linewidth=0.5, label='Detrended F-Fneu')
if np.max(np.abs(c_trace)) > 0:
    # Original C (scaled)
    c_scaled = c_trace * (np.std(f_raw) / np.std(c_trace + 1e-10))
    c_scaled = c_scaled + np.mean(f_raw) - np.mean(c_scaled)
    axes[1].plot(t, c_scaled, 'b', alpha=0.6, linewidth=0.8, label='C original (scaled)')
    
    # Corrected C (scaled)
    c_corr_scaled = c_corrected * (np.std(f_raw) / np.std(c_corrected + 1e-10))
    c_corr_scaled = c_corr_scaled + np.mean(f_raw) - np.mean(c_corr_scaled)
    axes[1].plot(t, c_corr_scaled, 'r', alpha=0.8, linewidth=0.8, label='C corrected (scaled)')
    
axes[1].set_ylabel('F / C')
axes[1].set_title('Deconvolved C Trace: Original vs Corrected')
axes[1].legend()

# Panel 3: Spike trains before and after correction
axes[2].fill_between(t, 0, sp_before, color='r', alpha=0.5, label='Before correction')
axes[2].fill_between(t, 0, -sp_after, color='b', alpha=0.5, label='After correction (inverted)')
axes[2].axhline(0, color='k', linewidth=0.5)
axes[2].set_ylabel('Spike Mag')
axes[2].set_title('Spike Trains: Before (top) vs After Correction (bottom, inverted)')
axes[2].legend()

# Panel 4: Spike magnitudes vs time with regression
if len(spike_idx_before) >= 10:
    axes[3].scatter(spike_idx_before, spike_mag_before, c='r', alpha=0.6, s=50, 
                    label=f'Before (slope={slope_before:.2e}, R²={r_before**2:.3f})')
    axes[3].scatter(spike_idx_after, spike_mag_after, c='b', alpha=0.6, s=50,
                    label=f'After (slope={slope_after:.2e}, R²={r_after**2:.3f})')
    axes[3].plot(spike_idx_before, int_before + slope_before * spike_idx_before, 'r--', linewidth=2)
    axes[3].plot(spike_idx_after, int_after + slope_after * spike_idx_after, 'b--', linewidth=2)
axes[3].axhline(np.median(spike_mag_before), color='gray', linestyle=':', label='Target (median)')
axes[3].set_xlabel('Frame')
axes[3].set_ylabel('Spike Mag')
axes[3].set_title('Spike Magnitudes vs Time: Before vs After Correction')
axes[3].legend()

# Panel 5: Histogram of spike magnitudes
axes[4].hist(spike_mag_before, bins=20, alpha=0.5, color='r', label='Before correction', density=True)
axes[4].hist(spike_mag_after, bins=20, alpha=0.5, color='b', label='After correction', density=True)
axes[4].axvline(np.median(spike_mag_before), color='r', linestyle='--', linewidth=2)
axes[4].axvline(np.median(spike_mag_after), color='b', linestyle='--', linewidth=2)
axes[4].set_xlabel('Spike Magnitude')
axes[4].set_ylabel('Density')
axes[4].set_title('Distribution of Spike Magnitudes')
axes[4].legend()

plt.tight_layout()
plt.suptitle('correct_spike_drift Verification on Synthetic Bleached Data', y=1.01, fontsize=14)
plt.show()
