# Test correct_spike_drift with synthetic bleached calcium data
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === GENERATE SYNTHETIC CALCIUM DATA ===
np.random.seed(42)
n_frames = 10000
fr = 7.5  # 7.5 Hz frame rate

# Time axis
t = np.arange(n_frames) / fr

# Generate spike times (random Poisson-ish)
spike_times = np.sort(np.random.choice(n_frames, size=50, replace=False))

# Calcium kernel (GCaMP-like decay)
tau = 1.0  # 1 second decay
kernel_len = int(5 * tau * fr)
kernel = np.exp(-np.arange(kernel_len) / (tau * fr))

# Ground truth spikes (uniform magnitude)
sp_true = np.zeros(n_frames)
sp_true[spike_times] = 1.0

# Convolve to get calcium trace
c_true = np.convolve(sp_true, kernel, mode='full')[:n_frames]

# === ADD PHOTOBLEACHING ===
# 30% decay over recording
bleach_factor = 1 - 0.3 * (np.arange(n_frames) / n_frames)

# Apply bleaching to both signal and baseline
baseline = 100
f_bleached = baseline * bleach_factor + c_true * 50 * bleach_factor

# Add noise
noise = np.random.randn(n_frames) * 5
f_noisy = f_bleached + noise

# === SIMULATE WHAT DECONVOLUTION MIGHT OUTPUT ===
# The "sp" from deconvolution would have magnitudes affected by bleaching
sp_bleached = sp_true.copy()
sp_bleached[spike_times] = 1.0 * bleach_factor[spike_times]  # Spikes decay with bleaching

# === APPLY correct_spike_drift ===
from s2pfuns import correct_spike_drift
%matplotlib qt

sp_corrected = correct_spike_drift(sp_bleached, min_spikes=10)

# === PLOT RESULTS ===
fig, axes = plt.subplots(4, 1, figsize=(14, 10))

# Panel 1: Bleached fluorescence trace
axes[0].plot(t, f_noisy, 'k', alpha=0.7, linewidth=0.5)
axes[0].set_ylabel('F')
axes[0].set_title('Synthetic Bleached Calcium Trace (30% decay)')

# Panel 2: Original vs bleached spikes
axes[1].stem(t[spike_times], sp_true[spike_times], linefmt='g-', markerfmt='go', basefmt='k-', label='True (uniform)')
axes[1].stem(t[spike_times], sp_bleached[spike_times], linefmt='r-', markerfmt='ro', basefmt='k-', label='Bleached')
axes[1].set_ylabel('Spike Mag')
axes[1].set_title('Spike Magnitudes: True vs Bleached')
axes[1].legend()

# Panel 3: Bleached vs corrected
axes[2].stem(t[spike_times], sp_bleached[spike_times], linefmt='r-', markerfmt='ro', basefmt='k-', label='Bleached')
axes[2].stem(t[spike_times], sp_corrected[spike_times], linefmt='b-', markerfmt='bo', basefmt='k-', label='Corrected')
axes[2].set_ylabel('Spike Mag')
axes[2].set_title('Spike Magnitudes: Bleached vs Corrected')
axes[2].legend()

# Panel 4: Regression lines before/after
spike_idx = np.where(sp_bleached > 0.1 * np.max(sp_bleached))[0]
slope_before, int_before, r_before, _, _ = linregress(spike_idx, sp_bleached[spike_idx])
slope_after, int_after, r_after, _, _ = linregress(spike_idx, sp_corrected[spike_idx])

axes[3].scatter(spike_idx, sp_bleached[spike_idx], c='r', alpha=0.6, label=f'Before (slope={slope_before:.4f}, R²={r_before**2:.3f})')
axes[3].scatter(spike_idx, sp_corrected[spike_idx], c='b', alpha=0.6, label=f'After (slope={slope_after:.4f}, R²={r_after**2:.3f})')
axes[3].plot(spike_idx, int_before + slope_before * spike_idx, 'r--', linewidth=2)
axes[3].plot(spike_idx, int_after + slope_after * spike_idx, 'b--', linewidth=2)
axes[3].set_xlabel('Frame')
axes[3].set_ylabel('Spike Mag')
axes[3].set_title('Linear Regression: Before vs After Correction')
axes[3].legend()

plt.tight_layout()
plt.show()

print(f"\n=== RESULTS ===")
print(f"Before correction: slope = {slope_before:.6f}, R² = {r_before**2:.4f}")
print(f"After correction:  slope = {slope_after:.6f}, R² = {r_after**2:.4f}")
print(f"Slope reduction: {100*(1 - abs(slope_after/slope_before)):.1f}%")