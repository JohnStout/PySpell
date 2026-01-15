# After implementation, run in Python interactive console:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Add pyspell folder to path

from s2pfuns import postProcess

# path
self = postProcess(s2ppath=r"X:\John\Subjects - GCaMP Recordings\L612_F_RightPFC_L6Chrimson_PFCgcamp8f_Panrec\SDswitch_day9_FOV1_optoRec\SDswitch_day9_FOV1_optoRec_img\suite2p\plane0")

# Test new adaptive method
C_adaptive, S_adaptive, metrics = self.cleanup_raw_traces_adaptive(verbose=2)

# Compare with existing method
C_old, S_old = self.cleanup_raw_traces()

# Visual comparison for cell 0
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(self.F[0] - self.Fneu[0], 'k', linewidth=0.5)
axes[0].set_title('Original F - Fneu')
axes[1].plot(C_old[0], 'b', label='Old'); axes[1].plot(C_adaptive[0], 'r', label='Adaptive')
axes[1].legend(); axes[1].set_title('Denoised C')
axes[2].plot(S_old[0], 'b', label='Old'); axes[2].plot(S_adaptive[0], 'r', label='Adaptive')
axes[2].legend(); axes[2].set_title('Spikes S')
plt.tight_layout(); plt.show()