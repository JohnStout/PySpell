# This module are functions that interface with thorlabs outputs and includes a RAM efficient version of converting data
#
# RAM friendly mechanisms:
#   -> memory mapping
#   -> chunk saving, then cleaning memory
#   -> Multi-file save (splits file into separate files) - NOT SUPPORTED
#
# Essentially, we map a file to disk via memory mapping, then write chunks of data to memory.
# The chunks of data take up memory and so we continuously delete what we previously used to conserve memory.
#
#
# 07/20s - 08/5: JS and VC added thorfuns function to convert camera data
#
# John Stout

# packages
import numpy as np
import os
import xmltodict
import matplotlib.pyplot as plt
import time
import psutil
import tifffile as tf
import h5py
import scipy.stats as stat
import scipy.io as sio
#import dask.array as da
#from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing
import rawpy
from pathlib import Path
import sys
import cv2
import glob
from tqdm import tqdm
from natsort import natsorted

# load imgfuns
#try:
#    import imgfuns
#except:
#    #os.chdir(os.path.join(os.getcwd(),'code'))
#    sys.path.append(str(Path(__file__).resolve().parent))
#    import imgfuns
# TODO: RECHECK AND VALIDATE ALL FUNCTIONS IN CONVERT
# TODO: CHECK 4D, Validate suite2p and max-proj
# TODO: Validate with single plane data
# TODO: Implement NWB as an option?

# Minimal ram usage
# the matlab version has a thresholding mechanism to detect potential LED artifacts
class RawToTif():
    '''
    This code writes your .raw file to .tif using various mechanisms.

    Writing suite2p style provides options to chunk write your data, performing very fast.

    This code is rather expensive on memory if you attempt to write one large file and so we must
    write separate files!

    10/17/2024: @JS discovered major issue with using np.memmap to index out the image files. Was 
                    generating incorrect max projection images as compared to both matlab and Fiji (watching the whole video)
                    - The updated method slices out the image planes
                    - - - This is really for a second reason: some LED artifacts are present in only one plane. This needs to be handled plane by plane.
    
    
    12/14/2024: @JS changed & bit operator to "and" logical operator.
    12/14/2024: Addition/fixing of suite2p method
    12/14/2024: Apparently, on our machine, parallel processing may actually increw time for numpy conversion. Now file is converted in __init__
    2/9/2025: @JS updated I/O operations on __init__ to improve loading
                - Updated the method to convert, changing loading/converting mechanisms based on whether the data are multiplane or single plane
                - Updated the method to convert to handle LED artifacts when opto stim occurs during same plane acquisition
    '''

    def __init__(self, filepath: str):
        '''
        Loads and converts .raw file while skipping the flyback frame. Provides options to process data.

            Args:
                >>> filepath: path to .raw file

        '''
        print("Please use method .convert(method='max_proj') rather than 'suite2p' and '4D'")
        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()

        # search for .raw file
        if '.raw' in os.path.split(filepath)[-1]:
            rootpath = os.path.split(filepath)[0]
        else:
            # define root path
            rootpath = filepath

            # discover your .raw imaging file
            filepath = [i for i in os.listdir(rootpath) if '.raw' in i and 'Image' in i]
            assert len(filepath)==1, "The code does not currently support multiple saved .raw files"
            print("Discovered", filepath[0])
            filepath = os.path.join(rootpath,filepath[0]) # save the result

        # get metadata
        root_contents = os.listdir(rootpath)
        metadata_file = [i for i in root_contents if '.xml' in i][0]
        metadata_path = os.path.join(rootpath,metadata_file)
        file = xmltodict.parse(open(metadata_path,"r").read()) # .xml file

        # define frame rate based on metadata
        fr = float(file['ThorImageExperiment']['LSM']['@frameRate'])

        # get dimensions of recorded data
        x=int(file['ThorImageExperiment']['LSM']['@pixelX'])
        y=int(file['ThorImageExperiment']['LSM']['@pixelY'])
        t=int(file['ThorImageExperiment']['Timelapse']['@timepoints']) # this is how the thorlabs code works
        z=int(file['ThorImageExperiment']['ZStage']['@steps']) # check this variable
        dims=(z,t,y,x)

        # data
        # Read the .raw file - FAST METHOD using np.fromfile
        print("Reading image data...")
        read_start = time.process_time()
        
        # Read entire file at once (MUCH faster than chunk-by-chunk)
        raw_data = np.fromfile(filepath, dtype='int16')
        
        # Calculate expected frame count
        pixels_per_frame = y * x
        total_raw_frames = len(raw_data) // pixels_per_frame
        
        # Reshape to (n_frames, y, x)
        raw_data = raw_data[:total_raw_frames * pixels_per_frame]  # trim any partial frame
        all_frames = raw_data.reshape(total_raw_frames, y, x)
        
        print(f"  Read {total_raw_frames} raw frames in {time.process_time() - read_start:.2f}s")
        
        # Remove flyback frames (every 4th frame for multiplane)
        if z > 1:
            # For multiplane: frames are [p0, p1, p2, flyback, p0, p1, p2, flyback, ...]
            # Keep frames where (index % 4) != 3
            keep_mask = np.arange(total_raw_frames) % 4 != 3
            planes = np.ascontiguousarray(all_frames[keep_mask])  # FAST: keep as numpy array
            total_frames = total_raw_frames // 4
            assert len(planes) == total_frames * 3, f"Frame count mismatch: {len(planes)} vs expected {total_frames * 3}"
        else:
            # Single plane: no flyback to remove - just use the array directly
            planes = all_frames  # No copy needed, just reassign reference
            total_frames = total_raw_frames
        
        print(f"  Kept {len(planes)} imaging frames (removed {total_raw_frames - len(planes)} flyback frames)")
        del raw_data  # free memory (all_frames may be aliased to planes)

        '''
        # instead of pulling all of that into memory, lets write it immediately, then call the mapped data
        offset=0; vector_list = []; counter = 0; offset_list = []
        try:
            for ti in range(t):
                for zi in range(z):     
                    vector_list.append(np.memmap(filepath, dtype='int16', offset=offset, mode='r', shape=(x,y)))
                    offset_list.append(offset) 
                    offset+=int(x*y*16/8) # bytes (16bit/8)
                    counter+=1                      
                # skip the flyback frame
                offset+=int(x*y*16/8)
        except:
            print("Aborting loop at:",str(ti),"/",str(t))          
        '''

        # store this for later
        self.planes = planes
        self.dims = (z,total_frames,y,x)
        self.fr = fr
        self.filepath = filepath
        self.rootpath = rootpath
        self.root_contents = root_contents
        self.metadata = file
        self.imgmode = 'multiplane' if z > 1 else 'single plane'
        #self.idx_offset_np = offset_list # this is really important for indexing from the np.memmap .raw file

        print("rootpath:",self.rootpath)

    def test_led_artifacts(
        self,
        outlier_thresh: float = 3.0,
        neg_z_exclude: float = 3.0,      # NEW: exclude negative-z warmup frames from CLEAN pool
        sigma_hp: float = 12.0,
        template_n: int = 30,
        bg_keep_frac: float = 0.70,      # darkest fraction used as "background"
        do_rowwise_bg: bool = False,
        do_rowwise_template_scale: bool = False,
        save_fig: bool = True,
    ):
        """
        Diagnostic for LED artifact detection & correction comparisons.

        Compares:
        1) Highpass (preview)
        2) Interpolation (preview; loses info)
        3) Template subtraction (scaled; preserves events)
        4) Background subtraction (scalar; preserves events if artifact is mostly offset)
            (+ optional row-wise version for banding)

        IMPORTANT CHANGE:
        - LED events are detected as POSITIVE outliers only: z > outlier_thresh
        - Negative outliers (laser power-up) are excluded from *clean/background/template* pools:
                z < -neg_z_exclude
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        import scipy.stats as stats
        from matplotlib.gridspec import GridSpec

        print("=" * 70)
        print("LED ARTIFACT DETECTION & CORRECTION METHOD COMPARISON")
        print("=" * 70)

        n_frames = len(self.planes)

        # --------------------------
        # Helpers
        # --------------------------
        def highpass(img: np.ndarray, sigma):
            if sigma is None or sigma <= 0:
                return img.copy()
            return img - gaussian_filter(img, sigma=float(sigma))

        def background_mask(ref_img: np.ndarray, keep_frac: float = 0.70):
            thr = np.quantile(ref_img, keep_frac)
            return ref_img <= thr

        def bg_level(img: np.ndarray, mask: np.ndarray):
            return float(np.median(img[mask]))

        def correct_bg_offset(frame: np.ndarray, ref: np.ndarray, keep_frac: float = 0.70):
            m = background_mask(ref, keep_frac=keep_frac)
            d = bg_level(frame, m) - bg_level(ref, m)
            return frame.astype(np.float32) - d, d

        def correct_bg_rowwise(frame: np.ndarray, ref: np.ndarray, keep_frac: float = 0.70):
            f = frame.astype(np.float32)
            r = ref.astype(np.float32)
            thr = np.quantile(r, keep_frac)
            m = r <= thr
            row_off = np.zeros((f.shape[0],), dtype=np.float32)
            for rr in range(f.shape[0]):
                mr = m[rr, :]
                if np.any(mr):
                    row_off[rr] = np.median(f[rr, mr]) - np.median(r[rr, mr])
            return f - row_off[:, None], row_off

        def estimate_scale_global(frame: np.ndarray, template: np.ndarray, mask: np.ndarray, eps: float = 1e-6):
            f = frame[mask].ravel().astype(np.float32)
            t = template[mask].ravel().astype(np.float32)
            return float((f @ t) / ((t @ t) + eps))

        def estimate_scale_rowwise(frame: np.ndarray, template: np.ndarray, mask: np.ndarray, eps: float = 1e-6):
            f = frame.astype(np.float32)
            t = template.astype(np.float32)
            a = np.zeros((f.shape[0],), dtype=np.float32)
            for rr in range(f.shape[0]):
                mr = mask[rr, :]
                if not np.any(mr):
                    continue
                fr = f[rr, mr]
                tr = t[rr, mr]
                a[rr] = float((fr @ tr) / ((tr @ tr) + eps))
            return a

        def subtract_scaled_template(frame: np.ndarray, template: np.ndarray, mask: np.ndarray, rowwise: bool = False):
            f = frame.astype(np.float32)
            t = template.astype(np.float32)
            if not rowwise:
                a = estimate_scale_global(f, t, mask)
                return f - a * t, a
            else:
                a_y = estimate_scale_rowwise(f, t, mask)
                return f - (a_y[:, None] * t), a_y

        def show_img(ax, img, title, cmap="gray", vmin=None, vmax=None):
            if vmin is None or vmax is None:
                vmin, vmax = np.percentile(img, [1, 99])
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        # ==========================
        # STEP 1: Detect LED events (POSITIVE ONLY)
        # ==========================
        print("\n[1/4] Detecting LED events via mean pixel intensity...")

        subsample = max(1, n_frames // 3000)
        sample_indices = np.arange(0, n_frames, subsample)
        mean_pixels = np.array([np.mean(self.planes[i]) for i in sample_indices], dtype=np.float32)
        time_sec = sample_indices / float(self.fr)

        # SIGNED z-score
        z = stats.zscore(mean_pixels, nan_policy="omit")

        # Positive LED spikes only
        led_mask = z > outlier_thresh

        # Negative “laser power-up / warmup” outliers (exclude from clean pool)
        neg_mask = z < -neg_z_exclude

        # Clean pool excludes BOTH led spikes and warmup negatives
        clean_mask = ~(led_mask | neg_mask)

        outlier_sample_idx = np.where(led_mask)[0]
        neg_sample_idx = np.where(neg_mask)[0]
        clean_sample_idx = np.where(clean_mask)[0]

        outlier_frames = sample_indices[outlier_sample_idx]
        clean_frames = sample_indices[clean_sample_idx]

        print(f"  Total frames: {n_frames} (subsampled to {len(sample_indices)})")
        print(f"  Detected {len(outlier_frames)} LED frames (z > {outlier_thresh})")
        print(f"  Excluding {len(neg_sample_idx)} warmup/negative frames (z < -{neg_z_exclude}) from clean pool")
        print(f"  Clean frames available (for bg/template): {len(clean_frames)}")

        if len(outlier_frames) == 0:
            print("\n  No positive LED artifacts detected in sampled frames.")
            return {
                "n_outliers": 0,
                "outlier_frames": np.array([]),
                "clean_frames": clean_frames,
                "mean_pixels": mean_pixels,
                "z_scores_signed": z,
                "time_sec": time_sec,
                "neg_excluded_frames": sample_indices[neg_sample_idx],
            }

        if len(clean_frames) == 0:
            print("\n  WARNING: No clean frames after exclusions. Lower neg_z_exclude or outlier_thresh.")
            return {
                "n_outliers": int(len(outlier_frames)),
                "outlier_frames": outlier_frames,
                "clean_frames": clean_frames,
                "mean_pixels": mean_pixels,
                "z_scores_signed": z,
                "time_sec": time_sec,
                "neg_excluded_frames": sample_indices[neg_sample_idx],
            }

        # ==========================
        # STEP 2: Select example frames
        # ==========================
        print("\n[2/4] Selecting example frames for comparison...")

        # pick baseline as middle clean (stable by construction)
        baseline_idx = int(clean_frames[len(clean_frames) // 2])

        # pick contaminated example as strongest LED event (not first)
        contam_idx = int(outlier_frames[np.argmax(z[outlier_sample_idx])])

        # neighbors for interpolation/reference: nearest clean before/after contam
        frames_before = clean_frames[clean_frames < contam_idx]
        frames_after = clean_frames[clean_frames > contam_idx]
        interp_before_idx = int(frames_before[-1]) if len(frames_before) > 0 else max(0, contam_idx - 1)
        interp_after_idx = int(frames_after[0]) if len(frames_after) > 0 else min(n_frames - 1, contam_idx + 1)

        print(f"  Baseline frame: {baseline_idx}")
        print(f"  Contaminated (example) frame: {contam_idx}")
        print(f"  Neighbor clean frames: {interp_before_idx}, {interp_after_idx}")

        # ==========================
        # STEP 3: Apply correction methods
        # ==========================
        print("\n[3/4] Testing correction methods...")

        baseline_orig = self.planes[baseline_idx].astype(np.float32)
        contam_orig = self.planes[contam_idx].astype(np.float32)
        before_frame = self.planes[interp_before_idx].astype(np.float32)
        after_frame = self.planes[interp_after_idx].astype(np.float32)

        # Clean reference for background subtraction (and mask definition)
        ref_clean = (before_frame + after_frame) / 2.0

        # Method 1: Highpass (preview)
        baseline_hp = highpass(baseline_orig, sigma_hp)
        contam_hp = highpass(contam_orig, sigma_hp)

        # Method 2: Interpolation (preview)
        contam_interp = ref_clean

        # Method 3: Template subtraction (scaled)
        # Choose top-N LED frames by z (strongest events), and match clean frames in time
        led_order = np.argsort(z[outlier_sample_idx])[::-1]
        sel_led_frames = outlier_frames[led_order[: min(template_n, len(outlier_frames))]].astype(int)

        # time-matched clean frames: nearest clean to each LED frame (prevents warmup/drift bias)
        clean_arr = clean_frames.astype(int)
        matched_clean = []
        for lf in sel_led_frames:
            j = int(np.argmin(np.abs(clean_arr - lf)))
            matched_clean.append(clean_arr[j])
        matched_clean = np.array(matched_clean, dtype=int)

        template_contam = np.median(
            np.stack([self.planes[i].astype(np.float32) for i in sel_led_frames], axis=0),
            axis=0
        )
        template_clean = np.median(
            np.stack([self.planes[i].astype(np.float32) for i in matched_clean], axis=0),
            axis=0
        )
        led_template = (template_contam - template_clean).astype(np.float32)

        bg_mask = background_mask(ref_clean, keep_frac=bg_keep_frac)

        contam_template_sub, a_contam = subtract_scaled_template(
            contam_orig, led_template, bg_mask, rowwise=do_rowwise_template_scale
        )
        baseline_template_sub, a_base = subtract_scaled_template(
            baseline_orig, led_template, bg_mask, rowwise=do_rowwise_template_scale
        )

        # Method 4: Basic background subtraction (scalar + optional row-wise)
        contam_bg_sub, d_bg = correct_bg_offset(contam_orig, ref_clean, keep_frac=bg_keep_frac)
        baseline_bg_sub, d_bg_base = correct_bg_offset(baseline_orig, ref_clean, keep_frac=bg_keep_frac)

        if do_rowwise_bg:
            contam_bg_row, d_bg_row = correct_bg_rowwise(contam_orig, ref_clean, keep_frac=bg_keep_frac)
            baseline_bg_row, d_bg_row_base = correct_bg_rowwise(baseline_orig, ref_clean, keep_frac=bg_keep_frac)
        else:
            contam_bg_row, d_bg_row = None, None
            baseline_bg_row, d_bg_row_base = None, None

        print(f"  ✓ Highpass (σ={sigma_hp})")
        print("  ✓ Interpolation (preview)")
        print(f"  ✓ Scaled template subtraction (n={len(sel_led_frames)}, rowwise_scale={do_rowwise_template_scale})")
        print(f"  ✓ Background subtraction (keep_frac={bg_keep_frac}, rowwise={do_rowwise_bg})")

        # ==========================
        # STEP 4: Plot
        # ==========================
        print("\n[4/4] Generating comparison figure...")

        n_method_cols = 6 if do_rowwise_bg else 5  # orig + HP + interp + template + bg + (rowwise bg)
        total_cols = max(4, n_method_cols)

        fig = plt.figure(figsize=(4 * total_cols, 12))
        gs = GridSpec(4, total_cols, figure=fig, height_ratios=[1.2, 1, 1, 1], hspace=0.3, wspace=0.2)

        # Row 1: trace
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_sec, mean_pixels, "b-", linewidth=0.5, alpha=0.7, label="Mean pixel")

        # LED events (positive)
        ax1.scatter(time_sec[outlier_sample_idx], mean_pixels[outlier_sample_idx],
                    c="red", s=30, zorder=5, label=f"LED events (z>{outlier_thresh})")

        # warmup negative excluded frames (for visibility only)
        if len(neg_sample_idx) > 0:
            ax1.scatter(time_sec[neg_sample_idx], mean_pixels[neg_sample_idx],
                        c="orange", s=18, zorder=4, label=f"Warmup excluded (z<-{neg_z_exclude})")

        ax1.axvline(baseline_idx / float(self.fr), color="green", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_idx})")
        ax1.axvline(contam_idx / float(self.fr), color="red", linestyle="--", alpha=0.7, label=f"Contaminated ({contam_idx})")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mean intensity")
        ax1.set_title(f"Mean Pixel Trace: {len(outlier_frames)} LED events detected (sampled)")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Row 2: baseline/contam/diff/template in first 4 cols
        ax2a = fig.add_subplot(gs[1, 0])
        show_img(ax2a, baseline_orig, f"BASELINE (frame {baseline_idx})\nOriginal")

        ax2b = fig.add_subplot(gs[1, 1])
        show_img(ax2b, contam_orig, f"CONTAMINATED (frame {contam_idx})\nOriginal")

        ax2c = fig.add_subplot(gs[1, 2])
        show_img(ax2c, contam_orig - baseline_orig, "DIFFERENCE\n(Contam - Baseline)")

        ax2d = fig.add_subplot(gs[1, 3])
        show_img(ax2d, led_template, f"LED TEMPLATE\n(median contam - median clean)\n(n={len(sel_led_frames)})")

        # Row 3: contaminated methods
        col = 0
        ax3a = fig.add_subplot(gs[2, col]); col += 1
        show_img(ax3a, contam_orig, "CONTAMINATED\nOriginal")

        ax3b = fig.add_subplot(gs[2, col]); col += 1
        show_img(ax3b, contam_hp, f"HIGHPASS (preview)\nσ={sigma_hp}")

        ax3c = fig.add_subplot(gs[2, col]); col += 1
        show_img(ax3c, contam_interp, "INTERPOLATE (preview)\n(avg clean neighbors)")

        ax3d = fig.add_subplot(gs[2, col]); col += 1
        a_txt = f"{a_contam:.3f}" if np.isscalar(a_contam) else f"rowwise ({a_contam.min():.2f}..{a_contam.max():.2f})"
        show_img(ax3d, contam_template_sub, f"TEMPLATE SUB (scaled)\nscale={a_txt}")

        ax3e = fig.add_subplot(gs[2, col]); col += 1
        show_img(ax3e, contam_bg_sub, f"BG SUB (scalar)\nΔbg={d_bg:.2f}")

        if do_rowwise_bg:
            ax3f = fig.add_subplot(gs[2, col]); col += 1
            show_img(ax3f, contam_bg_row, "BG SUB (row-wise)\n(Δ per row)")

        # Row 4: baseline methods
        col = 0
        ax4a = fig.add_subplot(gs[3, col]); col += 1
        show_img(ax4a, baseline_orig, "BASELINE\nOriginal")

        ax4b = fig.add_subplot(gs[3, col]); col += 1
        show_img(ax4b, baseline_hp, f"BASELINE HP (preview)\nσ={sigma_hp}")

        ax4c = fig.add_subplot(gs[3, col]); col += 1
        b_before = max(0, baseline_idx - 5)
        b_after = min(n_frames - 1, baseline_idx + 5)
        baseline_interp = (self.planes[b_before].astype(np.float32) + self.planes[b_after].astype(np.float32)) / 2.0
        show_img(ax4c, baseline_interp, "BASELINE Interp (preview)\n(avg ±5 frames)")

        ax4d = fig.add_subplot(gs[3, col]); col += 1
        a_txt_b = f"{a_base:.3f}" if np.isscalar(a_base) else f"rowwise ({a_base.min():.2f}..{a_base.max():.2f})"
        show_img(ax4d, baseline_template_sub, f"BASELINE Template Sub\nscale={a_txt_b}")

        ax4e = fig.add_subplot(gs[3, col]); col += 1
        show_img(ax4e, baseline_bg_sub, f"BASELINE BG Sub\nΔbg={d_bg_base:.2f}")

        if do_rowwise_bg:
            ax4f = fig.add_subplot(gs[3, col]); col += 1
            show_img(ax4f, baseline_bg_row, "BASELINE BG Sub (row-wise)")

        plt.tight_layout()

        outpath = os.path.join(self.rootpath, "ledArtifactTest.png")
        if save_fig:
            plt.savefig(outpath, dpi=150, bbox_inches="tight")
            print(f"  Figure saved to: {outpath}")
        plt.show()

        return {
            "n_outliers": int(len(outlier_frames)),
            "outlier_frames": outlier_frames,
            "neg_excluded_frames": sample_indices[neg_sample_idx],
            "clean_frames": clean_frames,
            "mean_pixels": mean_pixels,
            "z_scores_signed": z,
            "time_sec": time_sec,
            "baseline_idx": baseline_idx,
            "contam_idx": contam_idx,
            "interp_neighbors": (interp_before_idx, interp_after_idx),
            "led_template": led_template,
            "bg_keep_frac": bg_keep_frac,
            "bg_delta_contam": d_bg,
            "template_scale_contam": a_contam,
            "do_rowwise_bg": do_rowwise_bg,
            "do_rowwise_template_scale": do_rowwise_template_scale,
        }

    def convert(self, method: str = 'max_proj', chunker: int = 1000, led_artifacts: str = 'y', memmap_write: bool = False, wipe_and_replace: bool = False, preview_upsample: bool = False, test_upsample: bool = False, upsample_small: bool = False):

        '''
        Method to convert data
        
        Args:
            >>> method: method on how to format your data
                    '4D': Preserves your z-dimension and saves your file as a 4D array (z,t,y,x)
                    'suite2p': preserves your z-dimension but saves your file as a 3D array (t,y,x) as such:
                                frame0 = time0_plane0_channel0
                                frame1 = time0_plane1_channel0
                                frame2 = time0_plane2_channel0
                            Assuming a 3 plane video (code is agnostic to number of planes)
                    'max_proj': maximum projection taken over the z-plane to generate a 3D file (t,y,x)
            
            >>> chunker: how many images to save at once
            >>> led_artifacts: 'y' or 'n' - whether to perform LED artifact correction
            >>> led_method: method for LED artifact correction (only used when led_artifacts='y')
                    'interpolate': Replace contaminated frames with interpolated neighbors [RECOMMENDED]
                    'highpass': Spatial highpass filtering to remove uniform LED illumination
                    'template': Subtract average LED pattern from contaminated frames

            >>> memmap_write: False. This can be removed. The imwrite method is better and the result is still memory mappable.

            IMPORTANT** led_artifacts is only functional for max_proj

            This code uses parallel processing to handle the large imaging dataset

        Written by John Stout

        # --- EDITS --- #
        # 10/9/2024: updated the max_proj method and defaulted the convert method to max_proj
                        - Included an option for artifact conversion
                        - Included an option for the user to control the scale of saving with "chunker"
        # 10/15/2024: Updated mechanism to perform computations in parallel using copilot
        # 12/18/2024: Finished updating the run_parallel mechanism
        # 1/10/2025: Fixed issue with shape. Must have edited the code in dec to handle numpy rather than list and didnt fix .shape attribute
        # 2/9/2025: Includes capacities to convert 1 plane imaging with LED saturation events being interpolated.

        '''
        print("This code does not support multi-channel recordings")

        print("Starting at",str(psutil.virtual_memory()[2]),"<%> RAM utility")
        code_start = time.process_time()        

        # get dimensions
        z,t,y,x = self.dims

        # chunky writing variables
        total_count = t*z; # get total count of timepoints and amount of samples to chunk data by
        count_range = list(range(total_count)) # define the range over which to sample data

        # temporary solution to prevent use of other methods before making sure they follow the updated procedures set by 'max_proj' and the __init__
        assert method != '4D', "method=='4D' has not been validated. Please set method='max_proj' or method='suite2p' "
        
        # create a memory mappable file, with vectorized data
        if '4D' in method and self.imgmode == 'multiplane':

            code_start = time.process_time()   
            print("method: 4D detected. Your file will be saved with dimensions (z,t,y,x):",z,t,y,x)
            print("Please wait while memory mapped file is created...")
            self.fname = fname_new(self.rootpath,'img_mmap_4D.tif')
            #self.fname = os.path.join(self.rootpath,'img_mmap_4D.tif')
            im = tf.memmap(
                self.fname,
                shape=(z,t,y,x),
                dtype=np.uint16,
                imagej=True
                #append=True
            )
            print(time.process_time() - code_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # Chunking by time, into a z-plane
            for zi in range(z):
                time_range = list(range(t)); #chunker = 500; 

                # array of ca data in plane zi
                np_mem_list = []
                for idxi in self.idx_offset_np[zi::z]:
                    np_mem_list.append(np.memmap(self.filepath, dtype='int16', offset=idxi, mode='r', shape=(x,y)))

                # chunk write
                for timei in time_range[::chunker]:
                    im[zi,timei:timei+chunker,:,:] = np_mem_list[timei:timei+chunker] 
                    im.flush()
                    del im; im=tf.memmap(self.fname)
                    print("Run time for:",str(timei),"/",str(t), time.process_time() - code_start)
                    print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")   

                del np_mem_list

        # this is the suite2p method for 4D data
        elif 'suite2p' in method and self.imgmode == 'multiplane':

            # planes is already a numpy array from __init__
            print(f"Suite2p mode - planes shape: {self.planes.shape}, dtype: {self.planes.dtype}")
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")  

            # TODO: replace this with boolean
            if led_artifacts.lower() == 'y':
                print("Running image interpolation for LED artifacts...")
                ledArtifacts = dict(); meanData = dict(); meanXYzData = dict()
                for zi in range(z):
                    print("Working to correct artifacts in plane",zi)

                    # identify candidate artifact events
                    mean_pixels  = np.mean(np.mean(self.planes[zi::3],axis=1),axis=1) # get a pixel average over time
                    meanXYz      = np.abs(stat.zscore(mean_pixels,axis=0)) # zscore the averaged pixels
                    ledArtifact  = np.asarray(np.where(meanXYz > 7)).flatten()
                    ledArtifacts['Axis'+str(zi)] =  ledArtifact
                    meanData['Axis'+str(zi)]     =  mean_pixels
                    meanXYzData['Axis'+str(zi)]  =  meanXYz

                    # check by introducing artifacts and then interpolating them
                    # self.planes[9999] = np.full((512, 512), 1000)
                    # self.planes[10000] = np.full((512, 512), 1000)
                    # self.planes[10001] = np.full((512, 512), 1000)
                    # after you run the code below, plot
                    # plt.plot(meanXYz)

                    # interpolate missing data
                    for imgi in ledArtifact:
                        if imgi > 1 and imgi < len(meanXYz):
                            print("Interpolating artifact at index:",imgi)

                            # get data surrounding artifact
                            img_temp = np.moveaxis(self.planes[zi::3][imgi-1:imgi+2], 0, -1)
                            img_interp = interp_img(img=img_temp)

                            # reshape result
                            img_interp = np.moveaxis(img_interp,-1,0)

                            # replace data
                            self.planes[zi+imgi*3] = img_interp[1]

                # save array
                print("Saving ledArtifact data...")
                ledMat = {"ledArtifact": ledArtifacts,
                        "meanXY": meanData,
                        "meanXYz": meanXYzData,
                        "info": "ledArtifact is an index of artifacts. meanXY is the pixel average. meanXYz is |zscore(meanXY)|."}
                artFile = os.path.join(self.rootpath,'ledArtifactDataInterp.mat')
                sio.savemat(artFile, ledMat)

            # quicker write
            self.fname = os.path.join(self.rootpath, 'imgPlaneZ.tif')

            # convert to numpy then save
            print(f'Writing imgPlaneZ.tif to: {self.fname}')
            tf.imwrite(self.fname, self.planes, dtype=self.planes.dtype, bigtiff=True)
        
        # here is the max projection method that the lab prefers
        elif 'max_proj' in method and self.imgmode == 'multiplane':

            print("method: max_proj detected. Your file will be saved with dimensions (t,y,x):",t,y,x)
            print("Please wait while memory mappable file is created...")
            
            # might save time by replacing this with a more efficient approach
            if wipe_and_replace == True:
                imgFound = len([i for i in os.listdir(self.rootpath) if 'img.tif' in i])
                if imgFound > 0:
                    print("Wiping img.tif file to replace it.")
                    os.remove(os.path.join(self.rootpath,'img.tif'))
            self.fname = fname_new(self.rootpath,'img.tif')

            # lets chunk it!
            # beautiful thing about python is that if the loop exceeds the samples, python will grab the remaining samples, despite you requesting more than what exists!
            time_loop = list(range(t)); time_chunker = time_loop[0::chunker]
            assert time_loop[-1]+chunker > t, "You will not write all samples! Looping mechanism exceeds the total count of samples! FIX ME!"

            # Initialize timing
            process_start = time.process_time()

            # planes is already a numpy array from __init__
            print(f"Max_proj mode - planes shape: {self.planes.shape}, dtype: {self.planes.dtype}")

            assert self.planes.shape[0] % z == 0, f'array is improperly divided into frames (not divisible by z={z})'

            # separate planes using numpy slicing (fast, instant)
            separated_planes = np.zeros((z, int(self.planes.shape[0]/z), self.planes.shape[1], self.planes.shape[2]), dtype=self.planes.dtype)    
            print(f'Separating planes into Z: {separated_planes.shape[0]}, t: {separated_planes.shape[1]}, y: {separated_planes.shape[2]}, x: {separated_planes.shape[3]}')            
            for zi in range(z):
                separated_planes[zi, :, :, :] = self.planes[zi::z]
            print("Time to separate planes:", time.process_time() - process_start, "sec")

            # artifact detection
            if led_artifacts.lower() == 'y':
                print("Running image interpolation for LED artifacts...")
                ledArtifacts = dict(); meanData = dict(); meanXYzData = dict()
                for zi in range(z):
                    print("Working to correct artifacts in plane",zi)

                    # identify candidate artifact events
                    mean_pixels  = np.mean(np.mean(separated_planes[zi],axis=1),axis=1) # get a pixel average over time
                    meanXYz      = np.abs(stat.zscore(mean_pixels,axis=0)) # zscore the averaged pixels
                    ledArtifact  = np.asarray(np.where(meanXYz > 7)).flatten()
                    ledArtifacts['Axis'+str(zi)] =  ledArtifact
                    meanData['Axis'+str(zi)]     =  mean_pixels
                    meanXYzData['Axis'+str(zi)]  =  meanXYz

                    # interpolate missing data
                    for imgi in ledArtifact:
                        if imgi > 1 and imgi < len(meanXYz):
                            print("Interpolating artifact at index:",imgi)

                            # get data surrounding artifact
                            img_temp = np.moveaxis(separated_planes[zi][imgi-1:imgi+2], 0, -1)
                            img_interp = interp_img(img=img_temp)

                            # reshape result
                            img_interp = np.moveaxis(img_interp,-1,0)

                            # replace data
                            separated_planes[zi][imgi] = img_interp[1]

                            # fact check - these are blank arrays as expected
                            # plt.imshow(img_interp[2]-max_proj[imgi+1])
                            # plt.imshow(img_interp[0]-max_proj[imgi-1])

                # save array
                print("Saving ledArtifact data...")
                ledMat = {"ledArtifact": ledArtifacts,
                        "meanXY": meanData,
                        "meanXYz": meanXYzData,
                        "info": "ledArtifact is an index of artifacts. meanXY is the pixel average. meanXYz is |zscore(meanXY)|."}
                artFile = os.path.join(self.rootpath,'ledArtifactDataInterp.mat')
                sio.savemat(artFile, ledMat)

            # max projection
            print("Calculating max projection. This may take a moment...")
            process_start = time.process_time()        
            max_proj = np.max(separated_planes,axis=0) # rewriting same array to help memory
            max_proj = max_proj.astype('int16')
            del separated_planes # clean up memory
            print("Time to calculate max projection:",time.process_time() - process_start)
            print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

            # ensure that we have the appropriate bit precision
            if max_proj.dtype != 'int16':
                max_proj = max_proj.astype('int16')

            # save the max-projection image
            print("Writing imaging data to:", self.fname)

            # mechanisms to write files to disk
            if memmap_write is True:
                # memory map write
                im = tf.memmap(
                    self.fname,
                    shape=(t,y,x),
                    dtype=np.uint16,
                    imagej=True
                    #append=True
                )
                print("Time (sec):",time.process_time() - code_start)
                print("Update:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

                # This is an alternative method to write iteratively and is less prone to crashing
                for framesi in time_chunker:
                    im[framesi:framesi+chunker,:,:] = max_proj[framesi:framesi+chunker,:,:]
                    im.flush() # write to disk
                    del im; im=tf.memmap(self.fname) # clean up memory
                    print("Run time for",str(framesi),"/",str(t),"::: Time (s):",time.process_time() - code_start, "::: Memory:",str(psutil.virtual_memory()[2]),"<%> RAM utility")

                    # validated
                    #np_array = np.array(temp)
                    #fig, ax = plt.subplots(nrows=1,ncols=4)
                    #ax[0].imshow(np_array[0,0,:,:])
                    #ax[0].set_title("Axis1")
                    #ax[1].imshow(np_array[1,0,:,:])
                    #ax[1].set_title("Axis2")
                    #ax[2].imshow(np_array[2,0,:,:])
                    #ax[2].set_title("Axis3")
                    #ax[3].imshow(max_proj[0,:,:])
                    #ax[3].set_title("Max_proj")
            else:
                # quicker write
                tf.imwrite(self.fname, max_proj, dtype=max_proj.dtype, bigtiff=True)
        
        elif self.imgmode == 'single plane':
            print("Single plane data detected. Saving as 3D array (t,y,x)")
            self.fname = fname_new(self.rootpath,'img.tif')
            
            # planes is already a numpy array from __init__
            print(f"Single plane data shape: {self.planes.shape}, dtype: {self.planes.dtype}")
            
            # if you have a single plane image and you ran opto, it means the light may have contaminated your image
            # Supports multiple correction methods based on led_method parameter
            if led_artifacts.lower() == 'y':
                from scipy.ndimage import gaussian_filter
                import scipy.stats as stats
                
                process_start = time.process_time()
                original_dtype = self.planes.dtype
                n_frames = self.planes.shape[0]
                y_dim, x_dim = self.planes.shape[1], self.planes.shape[2]
                
                # ---- Step 1: Detect contaminated frames ----
                print("\n[1/3] Detecting LED-contaminated frames...")
                subsample = max(1, n_frames // 3000)
                sample_indices = np.arange(0, n_frames, subsample)
                mean_pixels = np.array([np.mean(self.planes[i]) for i in sample_indices])
                
                z_scores = stats.zscore(mean_pixels)  # SIGNED z-scores (not absolute) to detect positive outliers only
                outlier_thresh = 3.0
                outlier_mask = z_scores > outlier_thresh  # POSITIVE outliers only (LED artifacts are bright)
                outlier_sample_idx = np.where(outlier_mask)[0]
                clean_sample_idx = np.where(~outlier_mask)[0]
                
                outlier_frames = sample_indices[outlier_sample_idx]
                clean_frames = sample_indices[clean_sample_idx]
                
                print(f"  Detected {len(outlier_frames)} contaminated frames (z > {outlier_thresh})")
                print(f"  Clean frames: {len(clean_frames)}")
                
                # Store original data for QC
                qc_sample_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
                qc_original_frames = {idx: self.planes[idx].copy() for idx in qc_sample_frames}
                mean_pixels_before = mean_pixels.copy()
                
                correction_info = "No LED artifacts detected"
                if len(outlier_frames) == 0:
                    print("  No LED artifacts detected. Skipping correction.")
                else:
                    # ---- Step 2: Apply correction method ----
                    print(f"\n[2/3] Applying interpolation correction...")
                    
                    # INTERPOLATION: Replace contaminated frames with average of neighbors
                    n_corrected = 0
                    for contam_idx in outlier_frames:
                        contam_idx = int(contam_idx)
                        
                        # Find nearest clean frames before and after
                        before_candidates = clean_frames[clean_frames < contam_idx]
                        after_candidates = clean_frames[clean_frames > contam_idx]
                        
                        if len(before_candidates) > 0 and len(after_candidates) > 0:
                            before_idx = int(before_candidates[-1])
                            after_idx = int(after_candidates[0])
                            # Interpolate as average
                            self.planes[contam_idx] = ((self.planes[before_idx].astype(np.float32) + 
                                                        self.planes[after_idx].astype(np.float32)) / 2).astype(original_dtype)
                            n_corrected += 1
                        elif len(before_candidates) > 0:
                            self.planes[contam_idx] = self.planes[int(before_candidates[-1])]
                            n_corrected += 1
                        elif len(after_candidates) > 0:
                            self.planes[contam_idx] = self.planes[int(after_candidates[0])]
                            n_corrected += 1
                        
                        if n_corrected % 20 == 0:
                            print(f"    Corrected {n_corrected}/{len(outlier_frames)} frames...")
                    
                    print(f"  Interpolated {n_corrected} contaminated frames")
                    correction_info = f"Temporal interpolation: replaced {n_corrected} frames with neighbor averages"

                print(f"\n[3/3] LED artifact correction complete ({time.process_time() - process_start:.1f}s)")
                
                # ---- QC Visualization ----
                print("Generating QC visualization...")
                
                # Compute mean pixel after correction
                mean_pixels_after = np.array([np.mean(self.planes[i]) for i in sample_indices])
                time_axis = sample_indices / self.fr

                # Save metadata
                artMat = {
                    "led_method": "interpolate",
                    "n_contaminated_frames": len(outlier_frames),
                    "contaminated_frame_indices": outlier_frames,
                    "outlier_thresh": outlier_thresh,
                    "mean_pixels_before": mean_pixels_before,
                    "mean_pixels_after": mean_pixels_after,
                    "time_axis_sec": time_axis,
                    "info": correction_info
                }
                artFile = os.path.join(self.rootpath, 'ledArtifactCorrection_interpolate.mat')
                sio.savemat(artFile, artMat)
                print(f"Metadata saved to: {artFile}")

            # save img
            # Upsample small images to 512x512 for proper downstream processing (suite2p)
            target_size = 512
            img_height, img_width = self.planes.shape[1], self.planes.shape[2]
            if upsample_small and (img_height < target_size or img_width < target_size):
                import cv2
                from scipy.ndimage import gaussian_filter
                
                n_frames = self.planes.shape[0]
                sigma = 0.7
                dtype_info = np.iinfo(self.planes.dtype)
                
                # --- OPTIONAL PREVIEW ---
                if preview_upsample:
                    import matplotlib.pyplot as plt
                    print(f"\nSmall image detected ({img_height}x{img_width}). Previewing upsampling on sample frames...")
                    sample_idx = [0, n_frames // 2, n_frames - 1]
                    
                    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                    for col, idx in enumerate(sample_idx):
                        original_frame = self.planes[idx]
                        upsampled_frame = cv2.resize(original_frame.astype(np.float32), 
                                                      (target_size, target_size), 
                                                      interpolation=cv2.INTER_CUBIC)
                        upsampled_frame = gaussian_filter(upsampled_frame, sigma=sigma)
                        upsampled_frame = np.clip(upsampled_frame, dtype_info.min, dtype_info.max)
                        
                        axes[0, col].imshow(original_frame, cmap='gray')
                        axes[0, col].set_title(f'Original frame {idx} ({img_height}x{img_width})')
                        axes[0, col].axis('off')
                        
                        axes[1, col].imshow(upsampled_frame, cmap='gray')
                        axes[1, col].set_title(f'Upsampled frame {idx} ({target_size}x{target_size})')
                        axes[1, col].axis('off')
                    
                    plt.suptitle('Upsampling Preview')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.rootpath, 'upsampling_preview.png'), dpi=150)
                    plt.show()
                    
                    proceed = input(f"\nProceed with upsampling all {n_frames} frames? (y/n): ").strip().lower()
                    if proceed != 'y':
                        print("Upsampling skipped. Saving original resolution.")
                        tf.imwrite(self.fname, self.planes, dtype=self.planes.dtype, bigtiff=True)
                        process_end = time.process_time()
                        print(f"Total time spent converting: {(process_end - code_start)/60:.2f} minutes")
                        return
                
                # --- FULL UPSAMPLING (always runs unless preview aborted) ---
                if test_upsample:
                    frames_to_write = min(1000, n_frames)
                    out_path = os.path.join(self.rootpath, 'img_upsample_test.tif')
                    print(f"\nTEST MODE: Upsampling {frames_to_write} frames to {out_path}")
                else:
                    frames_to_write = n_frames
                    out_path = self.fname
                    print(f"\nUpsampling {n_frames} frames from {img_height}x{img_width} to {target_size}x{target_size}...")
                
                upsample_start = time.process_time()
                
                if test_upsample:
                    # Test mode: small enough to fit in memory, use proven tf.imwrite
                    test_array = np.zeros((frames_to_write, target_size, target_size), dtype=self.planes.dtype)
                    for i in range(frames_to_write):
                        frame = cv2.resize(self.planes[i].astype(np.float32), 
                                           (target_size, target_size), 
                                           interpolation=cv2.INTER_CUBIC)
                        frame = gaussian_filter(frame, sigma=sigma)
                        frame = np.clip(frame, dtype_info.min, dtype_info.max)
                        test_array[i] = frame.astype(self.planes.dtype)
                        if (i + 1) % 100 == 0:
                            print(f"  Upsampled {i+1}/{frames_to_write} frames...")
                    
                    tf.imwrite(out_path, test_array, bigtiff=True)
                    print(f"\nTest file saved. Open in Fiji to verify: {out_path}")
                    print("Re-run with test_upsample=False to process all frames.")
                    process_end = time.process_time()
                    print(f"Total time spent converting: {(process_end - code_start)/60:.2f} minutes")
                    return
                else:
                    # Full mode: write in chunks to avoid 163 GB in RAM
                    chunk_size = 1000
                    progress_interval = max(1, frames_to_write // 10)
                    for chunk_start in range(0, frames_to_write, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, frames_to_write)
                        chunk_array = np.zeros((chunk_end - chunk_start, target_size, target_size), dtype=self.planes.dtype)
                        
                        for j, i in enumerate(range(chunk_start, chunk_end)):
                            frame = cv2.resize(self.planes[i].astype(np.float32), 
                                               (target_size, target_size), 
                                               interpolation=cv2.INTER_CUBIC)
                            frame = gaussian_filter(frame, sigma=sigma)
                            frame = np.clip(frame, dtype_info.min, dtype_info.max)
                            chunk_array[j] = frame.astype(self.planes.dtype)
                        
                        # First chunk creates the file, subsequent chunks append
                        if chunk_start == 0:
                            tf.imwrite(out_path, chunk_array, dtype=self.planes.dtype, bigtiff=True)
                        else:
                            tf.imwrite(out_path, chunk_array, dtype=self.planes.dtype, bigtiff=True, append=True)
                        
                        if (chunk_end) % progress_interval == 0 or chunk_end == frames_to_write:
                            elapsed = (time.process_time() - upsample_start) / 60
                            print(f"  Upsampled {chunk_end}/{frames_to_write} frames... ({elapsed:.1f} min)")
                
                print(f"Upsampling complete: saved {target_size}x{target_size} to {out_path} ({(time.process_time() - upsample_start)/60:.1f} min)")
            else:
                tf.imwrite(self.fname, self.planes, dtype=self.planes.dtype, bigtiff=True)
        
        process_end = time.process_time()
        print(f"Total time spent converting: {(process_end - code_start)/60:.2f} minutes")

    def split_file():
        '''
        This function will be called internally to split the .raw file into multiple separate .raw files which then repopulate fname to then write out the data as needed
        '''
        pass

# interpolates frames
def interp_img(img):
    '''
    Translated from Matlab to Python thanks to CoPilot :)

    Args:
        >>> img: Y x X x Z shaped data with Z representing 3 images, the middle one needing interpolation.
                    if your data is of shape ZYX, just run np.moveaxis(array,0,-1)

    Outputs:
        >>> img_interp: interpolated image data (2nd image) based on the first and last image provided
        
    NOTE: if your array differs from the expected 512 x 512 XY pixel dimensions, you may have to fact check that the output is correct!

    '''
    from scipy.interpolate import RegularGridInterpolator

    # Extract the first and last slices
    A1 = img[:, :, 0]
    A3 = img[:, :, 2]
    
    # Create a new 3D matrix to store the interpolated data
    A_interp = np.zeros((img.shape[0],img.shape[1], img.shape[2]))
    
    # Assign the first and last slices to the new matrix
    A_interp[:, :, 0] = A1
    A_interp[:, :, 2] = A3
    
    # Create the grid for interpolation
    X, Y = np.arange(img.shape[0]), np.arange(img.shape[1])
    Z = np.array([0, 2])
    Xq, Yq, Zq = np.meshgrid(X, Y, np.arange(3), indexing='ij')
    
    # Perform the interpolation
    points = (X, Y, Z)
    interpolator = RegularGridInterpolator(points, img[:, :, [0, 2]], method='linear')
    img_interp = interpolator((Xq, Yq, Zq))

    # return 16 bit precision
    img_interp = img_interp.astype(np.int16)

    return img_interp

def fname_new(rootpath,fname):
    '''
    This code searches for existing fnames and updates the naming convention as to prevent overwrite
    
    Args:
        >>> rootpath: folder that you want your data saved to
        >>> fname: file name to save your data as
    '''
    root_contents = os.listdir(rootpath)
    next = False
    while next is False:
        if fname in root_contents:
            fullpath = os.path.join(rootpath,fname.split('.tif')[0]+'_new.tif')
            next = True
        else:
            fullpath = os.path.join(rootpath,fname)
            next = True

    return fullpath

# function to delete a .tif file
def remTif(fname):
    '''
    remTif: removes/deletes an img.tif file

    Args:
        >>> fname: path/to/your/img.tif
    
    '''
    # Delete the file if it exists
    if os.path.exists(fname):
        os.remove(fname)
        print(f"{fname} deleted")
    else:
        print(f"{fname} does not exist")

# converting behavioral data
# TODO: Build a GUI that allows to user to fix behavioral/recording issues
# for example, the user may have the wrong order of recording buttons or maybe didnt stop thorsync while xploring data generating piezo motor errors that misalign data
def importThorsync(bpath, imgdata_present=True):
    '''
    importThorSync
        Equivalent to the MATLAB version. Written by John Stout
        Additions:
            Handles times when your imaging and behavioral data are misaligned by using the piezo monitor

    Args:
        >>> bpath: path to behavioral data, including the .h5 extension
        >>> imgdata_present: boolean to indicate if imaging data is present. If True, the code will check for misalignment of behavioral and imaging data

    John Stout merged written code with copilot  

    UPDATES
    - 4/29/25: Updated list comprehension line when searching for lick index
    '''
    # [bData,frameData,trialData]=importThorsync(fileName, subsamp, saveData)

    import h5py
    import os
    import numpy as np

    # Default parameters
    def check_and_set_defaults(subsamp=None, saveData=None):
        if subsamp is None:
            subsamp = [1, 1]
        if saveData is None:
            saveData = True
        return subsamp, saveData
    subsamp, saveData = check_and_set_defaults()

    # behavioral path
    bpath = os.path.abspath(bpath)
    ext = os.path.splitext(bpath)[1]

    # Search for .h5 file extension - copilot
    if not ext.endswith('.h5'):
        print("Searching for .h5 file")
        dirFiles = os.listdir(bpath)  # directory contents
        fnames = [f for f in dirFiles if f.endswith('.h5')]  # file names in directory
        fileName = os.path.join(bpath, fnames[0])
        print(f"Discovered and loading: {fnames[0]}")
    else:
        fileName = bpath

    # Reading behavioral data
    print("Reading behavioral data from:",fileName)
    dataIn = h5py.File(fileName,'r')
    bData  = dict()
    for i in dataIn['DI'].keys():
        bData[i] = dataIn['DI'][i][:]
        if np.max(bData[i]) > 0:
            bData[i] = np.ravel(bData[i]/np.max(bData[i]))

    # another source of error are cases where the user stopped the img recording then started
    # viewing the data and causing the piezo to work again without stopping thorsync
    # np.where(np.logical_and(piezo_norm < 0.3, bData['FrameOut'] > 0.9)==True)
    
    # manually search for those events and fill in the index to set to zero
    #index = 5345472
    #for key in bData.keys():
    #    bData[key][index:-1] = 0.0

    # Index of frame times for behavior
    if imgdata_present:
        frameTimes = np.where(np.diff(bData['FrameOut'],axis=0)==1)[0]

        # piezo cycles
        piezo = dataIn['AI']['PiezoMonitor'][:]
        piezo_norm = np.ravel(piezo/np.max(piezo)) # scale to 1 and convert to 1D

        # when recordings start, the piezo kicks on
        idx_offrec = np.where(piezo_norm < 0.3)[0]

        # make sure that the first and last value of idx_offrec are outside of frameIdx
        good_rec = ( (idx_offrec[0] < frameTimes[0]) & (idx_offrec[-1] > frameTimes[-1]) )
        #assert good_rec==True, "Your behavioral data are misaligned with your imaging data. Use this session to modify the code"
        if good_rec == False:

            # if the first frame index is less than the first flatlined piezo, that means the experimenter started recording their
            # imaging data before starting thorsync. This happens because the piezo turns on when you hit start on the img software. So there
            # was never a flatlined piezo
            img_too_soon = frameTimes[0] < idx_offrec[0] # you started recording img too soon

            # the experimenter stopped thorsync before the img rec. The last index of frameIdx > last index of piezo.
            img_too_late = frameTimes[-1] > idx_offrec[-1] # you stopped recording img too late

            # if img_too_late
            if img_too_late:
                print("Experimenter turned off 1) ThorSync then 2) ThorImg. Trim end of imaging data.")
                save_tag = "_trimImgEnd"
            elif img_too_soon:
                print("Experimenter turned on 1) ThorImg then 2) ThorSync. Trim the start of imaging data.")
                save_tag = "_trimImgStart"

    # Extract velocity data from treadmill rotations
    if 'RotaryA' in bData and 'RotaryB' in bData:
        bData['RotaryA'][bData['RotaryA'] == 4] = 1
        bData['RotaryB'][bData['RotaryB'] == 128] = 1
        position = []
        counter = 0
        for i in range(len(bData['RotaryA']) - 1):
            aState = bData['RotaryA'][i]
            aNextState = bData['RotaryA'][i + 1]
            bNextState = bData['RotaryB'][i + 1]
            if aState != aNextState and aNextState == 1:
                if bNextState != aNextState:
                    counter += 1
                elif bNextState == aNextState:
                    counter -= 1
            position.append(counter)
        position.append(counter)
        position = np.array(position) * -1 * (38/250)  # flip direction and convert to cm
        bData['Velocity'] = np.diff(np.convolve(position, np.ones(100)/100, mode='same')) * 1000  # convert to cm/sec assuming 1kHz sample rate

    # Fit the other behavioral variables based on frameTimes
    if imgdata_present:
        frameData = {k: v[frameTimes] for k, v in bData.items()}

    # Get trial data
    trialStartTimes = np.where(np.diff(bData['trialOut']) == 1)[0]
    trialEndTimes = np.where(np.diff(bData['trialOut']) == -1)[0]

    if trialEndTimes[0] < trialStartTimes[0]:
        trialStartTimes = np.insert(trialStartTimes, 0, 0)
    if len(trialStartTimes) > len(trialEndTimes):
        trialEndTimes = np.append(trialEndTimes, len(bData['trialOut']))

    trialData = {'trial': [], 'trialLR': [], 'irrelLR': [], 'setID': [], 'lickNumL': [], 'lickNumR': [], 'trialCorrect': [], 'resDir': [], 'opto': [], 'info': []}
    trialData['info'] = 'Behavioral data from recording session. Note that in python version, zero-indexing is applied. MATLAB saved data add +1 to any indices.'
    for x in range(len(trialStartTimes)):
        trialData['trial'].append(x)
        trialData['trialLR'].append(1 if np.max(bData['trialLROut'][trialStartTimes[x]:trialEndTimes[x]]) == 1 else -1)
        trialData['irrelLR'].append(1 if np.max(bData['irrelLROut'][trialStartTimes[x]:trialEndTimes[x]]) == 1 else -1)
        trialData['setID'].append(1 if np.max(bData['setIDOut'][trialStartTimes[x]:trialEndTimes[x]]) == 1 else -1)
        trialData['lickNumL'].append(len(np.where(np.diff(bData['lickingLOut'][trialStartTimes[x]:trialEndTimes[x]]) == 1)[0]))
        trialData['lickNumR'].append(len(np.where(np.diff(bData['lickingROut'][trialStartTimes[x]:trialEndTimes[x]]) == 1)[0]))
        trialData['trialCorrect'].append(np.max(bData['rewardOut'][trialStartTimes[x]:trialEndTimes[x]]))
        trialData['resDir'].append(trialData['trialLR'][x] if trialData['trialCorrect'][x] == 1 else trialData['trialLR'][x] * -1)
        
        if x < len(trialStartTimes)-1:
            trialData['opto'].append(1 if np.max(bData['LEDStim'][trialEndTimes[x]:trialStartTimes[x+1]]) == 1 else 0)
        else:
            trialData['opto'].append(0)
    
    # use diff to identify select events
    lickingL = np.diff(bData['lickingLOut'])
    lickingR = np.diff(bData['lickingROut'])
    rewarded = np.diff(bData['rewardOut'])
    opto     = np.diff(bData['LEDStim'])

    # get the absolute index, the index of the events, irrespective of your imaging data
    timesData = {'trialStartTimes': trialStartTimes, 'trialEndTimes': trialEndTimes,
                  'lickTimesL': [], 'lickTimesR': [], 'rewardTimes': [], 'optoOnTimes': [], 'info': []}   
    timesData['info'] = 'Behavioral timestamp indices from recording session. Note that these are indices to the behavioral frames. You would have to align these with frameTimes. Note that in python version, zero-indexing is applied. MATLAB saved data add +1 to any indices.'
    for ti in range(len(trialStartTimes)):

        # Ensure the range does not exceed the valid indices of lickingL. This happens if there is a arduino code problem where the experiment doesn't end
        idx_lickL = [i for i in range(trialStartTimes[ti], min(trialEndTimes[ti], len(lickingL))) if lickingL[i] == 1]
        idx_lickR = [i for i in range(trialStartTimes[ti], min(trialEndTimes[ti], len(lickingR))) if lickingR[i] == 1]

        # Find indices where lickingLOut equals 1 within the trial range
        #idx_lickL = [i for i in range(trialStartTimes[ti], trialEndTimes[ti]) if lickingL[i] == 1]
        #idx_lickR = [i for i in range(trialStartTimes[ti], trialEndTimes[ti]) if lickingR[i] == 1]

        # you only need the first index for this variable because it will return all times when digital pulse==1
        idx_rew = [i for i in range(trialStartTimes[ti], min(trialEndTimes[ti], len(lickingL))) if rewarded[i] == 1]
        if len(idx_rew) > 0:
            idx_rew = idx_rew[0]
        else:
            idx_rew = np.nan

        # opto happens in between trials, during the ITI, so search throughout trial onset to trial onset+1
        if ti < len(trialStartTimes) - 1:
            idx_opto = [i for i in range(trialStartTimes[ti], min(trialStartTimes[ti+1], len(lickingL))) if opto[i] == 1]
        else:
            # search from the start of the last trial to the end of the session
            idx_opto = [i for i in range(trialStartTimes[ti], len(lickingL)) if opto[i] == 1]
        
        # Append found indices to the list
        timesData['lickTimesL'].append(idx_lickL)
        timesData['lickTimesR'].append(idx_lickR)
        timesData['rewardTimes'].append(idx_rew)
        timesData['optoOnTimes'].append(idx_opto)

    # sanity check - thanks copilot
    def assert_same_length(data_dict, dict_name):
        lengths = [len(data_dict[v]) for v in data_dict.keys() if v != 'info']
        assert len(set(lengths)) == 1, f"Not all lists in {dict_name} are of the same length. Lengths found: {lengths}"

    # Check the lengths of timesData and trialData
    assert_same_length(timesData, "timesData")
    assert_same_length(trialData, "trialData")

    # make into a dict
    if imgdata_present:
        beh_dict = {'timesData': timesData,
                    'trialData': trialData,
                    'frameData': frameData,
                    'bData': bData,
                    'frameTimes': frameTimes}
    else:
        beh_dict = {'timesData': timesData,
                    'trialData': trialData,
                    'bData': bData}

    # save the variable as .npy file
    #np.save(os.path.join(bpath,'behPy.npy'), beh_dict, allow_pickle = True)

    # in the case for MATLAB analysis, we want to add a sample to relevant arrays
    timesData['rewardTimes'] = [i+1 if not np.isnan(i) else i for i in timesData['rewardTimes']]
    timesData['trialStartTimes'] = timesData['trialStartTimes'] + 1
    timesData['trialEndTimes'] = timesData['trialEndTimes'] + 1
    trialData['trial'] = [i+1 if not np.isnan(i) else i for i in trialData['trial']]
    
    if imgdata_present:
        frameTimes = frameTimes + 1

    # save - updated on 5/6/2025
    if imgdata_present:
        if good_rec == False:
            sio.savemat(os.path.join(bpath,'beh'+save_tag+'.mat'), {
                #'timesData': timesData,
                'trialData': trialData,
                'bData': bData,
                #'frameData': frameData,
                #'frameTimes': frameTimes,
                }
            )
        else:
            sio.savemat(os.path.join(bpath,'beh.mat'), {
                #'timesData': timesData,            
                'trialData': trialData,
                'bData': bData,
                #'frameData': frameData,
                #'frameTimes': frameTimes,
                }
            )
    else:
        sio.savemat(os.path.join(bpath,'beh.mat'), {
            #'timesData': timesData,            
            'trialData': trialData,
            'bData': bData,
            }
        )

# Code below is for tif stitching
#TODO: This really needs to be cleaned up and documented

from functools import partial              # keeps tqdm nicely separated
from tqdm import tqdm
import concurrent.futures, glob, os
from typing import Union, Iterable

def batch_stitch_folders(
        root_glob: Union[Iterable[str], str],
         fps: float = 16.5,
         folders_parallel = None,
         **stitch_kw):
        
    """
    root_glob       : one or many glob patterns, e.g. r"Z:/data/**/img_*"
    fps             : camera fps to pass to stitch_cam_to_avi
    folders_parallel: # of concurrent folder jobs (defaults to N CPU cores)
    **stitch_kw     : forwarded to stitch_cam_to_avi
                      (workers, pages_per_chunk, max_inflight, delete_tifs…)
    """
    # ‑‑ discover folders --------------------------------------------------
    if isinstance(root_glob, str):
        root_glob = [root_glob]

    folders = []
    for pat in root_glob:
        folders.extend([f for f in glob.glob(pat, recursive=True)
                          if os.path.isdir(f)])

    if not folders:
        raise RuntimeError("No folders matched the pattern(s).")

    print(f"▶  Found {len(folders)} folders to stitch.")

    # ‑‑ run in parallel ---------------------------------------------------
    max_jobs = folders_parallel or max(os.cpu_count() - 1, 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_jobs) as ex:
        futs = {
            ex.submit(stitch_cam_to_avi, fld, fps=fps, **stitch_kw): fld
            for fld in folders
        }

        for fut in concurrent.futures.as_completed(futs):
            fld = futs[fut]
            try:
                fut.result()
                print(f"✅  {fld} done")
            except Exception as e:
                print(f"❌  {fld} failed: {e}")

def stitch_cam_to_avi(tif_folder, fps=16.5, delete_tifs=False,
                      workers=8, pages_per_chunk=345,
                      max_inflight=4):
    """
    • Detects _C1_ and _C2_ multipage‑TIFF stacks.
    • Launches one parallel job per camera so the two AVIs are written
      side‑by‑side instead of sequentially.
    """

    # 1.  discover stacks per camera ---------------------------------
    tif_files = natsorted(
        [os.path.join(tif_folder, f)
         for f in os.listdir(tif_folder)
         if f.lower().endswith('.tif')]
    )
    if not tif_files:
        raise FileNotFoundError("No *.tif files in folder.")

    c1 = [f for f in tif_files if '_C1_' in f]
    c2 = [f for f in tif_files if '_C2_' in f]

    # save the c1 and c2 lists to the tif_folder
    with open(os.path.join(tif_folder, 'c1_files_appended.txt'), 'w') as f:
        f.write('\n'.join(c1))
    with open(os.path.join(tif_folder, 'c2_files_appended.txt'), 'w') as f:
        f.write('\n'.join(c2))

    # identify the creation time of c1 and c2 files and save it to a text file
    # this is useful to identify when the recording started
    import time
    if c1:
        c1_times = [os.path.getctime(f) for f in c1]
        c1_time_str = time.ctime(min(c1_times))
        with open(os.path.join(tif_folder, 'c1_creation_time.txt'), 'w') as f:
            f.write(f"C1 files creation time (earliest): {c1_time_str}\n")
    if c2:
        c2_times = [os.path.getctime(f) for f in c2]
        c2_time_str = time.ctime(min(c2_times))
        with open(os.path.join(tif_folder, 'c2_creation_time.txt'), 'w') as f:
            f.write(f"C2 files creation time (earliest): {c2_time_str}\n")

    # parallel processing for camera conversion
    jobs = []
    if c1:
        out_c1 = os.path.join(
            tif_folder,
            os.path.basename(c1[0]).split('C1_')[0] + 'C1_stitched.avi')
        jobs.append(('C1', c1, out_c1))
    if c2:
        out_c2 = os.path.join(
            tif_folder,
            os.path.basename(c2[0]).split('C2_')[0] + 'C2_stitched.avi')
        jobs.append(('C2', c2, out_c2))

    if not jobs:
        raise RuntimeError("Stacks are present but neither _C1_ nor _C2_ "
                           "pattern was found in filenames.")

    # 2.  run one stitching job per camera in parallel ----------------
    #    (uses the CPU‑only parallel writer we added earlier)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = []
        for cam_tag, files, out_avi in jobs:
            fut = ex.submit(
                stitch_multipage_tifs_to_avi,
                files, out_avi,
                fps=fps,
                delete_tifs=delete_tifs,
                workers=workers,
                pages_per_chunk=pages_per_chunk,
                max_inflight=max_inflight,
            )
            futs.append((cam_tag, fut))

        # optional – nice progress / error handling
        for cam_tag, fut in futs:
            try:
                fut.result()                      # propagate exceptions
            except Exception as e:
                print(f"❌  {cam_tag} failed:", e)
                raise

import cv2, tifffile as tf, numpy as np, os, concurrent.futures, queue, threading
from tqdm import tqdm

def _load_tif(idx_path):
    """Worker: read one multipage‑TIF and return (idx, [frames])."""
    idx, path = idx_path
    with tf.TiffFile(path) as tif:
        pages = tif.pages
        out = []
        for pg in pages:
            fr = pg.asarray()

            # ---- dtype / channel normalisation (same rules you used) ----
            if fr.dtype != np.uint8:
                if fr.dtype == np.uint16:
                    lo, hi = int(fr.min()), int(fr.max())
                    fr = np.zeros_like(fr, dtype=np.uint8) if hi <= lo \
                         else ((fr.astype(np.float32)-lo)*255/(hi-lo)).astype(np.uint8)
                else:
                    fr = cv2.normalize(fr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if fr.ndim == 2:                         # grey → BGR
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            elif fr.ndim == 3 and fr.shape[2] == 1:
                fr = cv2.cvtColor(fr[..., 0], cv2.COLOR_GRAY2BGR)
            elif fr.ndim == 3 and fr.shape[2] > 3:  # drop α or extra channels
                fr = fr[..., :3]

            out.append(fr)
    return idx, out

import os, cv2, tifffile as tf, numpy as np
import concurrent.futures, queue, threading
from natsort import natsorted
from tqdm import tqdm

# ------------------------------------------------------------------
def _load_chunk(idx_path, pages_per_chunk):
    """Process‑pool worker: read one (sub‑)chunk of a multipage TIFF."""
    idx, path = idx_path
    with tf.TiffFile(path) as tif:
        pages = tif.pages
        start = 0
        while start < len(pages):
            sub = pages[start:start + pages_per_chunk]
            frames = []
            for pg in sub:
                fr = pg.asarray()

                # ---- normalise dtype / channels --------------------
                if fr.dtype != np.uint8:
                    fr = ((fr.astype(np.float32) - fr.min()) *
                          255.0 / (fr.max() - fr.min() + 1e-5)
                          ).astype(np.uint8)
                if fr.ndim == 2:
                    fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                elif fr.ndim == 3 and fr.shape[2] == 1:
                    fr = cv2.cvtColor(fr[..., 0], cv2.COLOR_GRAY2BGR)
                elif fr.ndim == 3 and fr.shape[2] > 3:
                    fr = fr[..., :3]

                frames.append(fr)
            yield (idx, start // pages_per_chunk, frames)
            start += pages_per_chunk

# ------------------------------------------------------------------
def stitch_multipage_tifs_to_avi(tif_files, output_avi, fps=16.5,
                                 delete_tifs=False,
                                 workers=None,
                                 pages_per_chunk=500,
                                 max_inflight=8):
    """
    CPU‑only, parallel reader / ordered writer.
    tif_files      : list of paths, already natsorted
    pages_per_chunk: slice large stacks so RAM stays bounded
    max_inflight   : number of chunks allowed in RAM simultaneously
    """
    if not tif_files:
        raise ValueError("No TIFF files given.")

    # ── discover frame size from very first page ───────────────────
    with tf.TiffFile(tif_files[0]) as t0:
        probe = t0.pages[0].asarray()
        if probe.ndim == 2:                       # grayscale
            h, w = probe.shape
        elif probe.ndim == 3:                     # rgb/bgr
            h, w = probe.shape[:2]
        else:
            raise ValueError("Unsupported TIFF frame shape.")

    # ── open CPU MJPG writer (universally readable) ────────────────
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(output_avi, fourcc, fps, (w, h), isColor=True)
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter '{output_avi}'")

    # ── queues & pool setup ────────────────────────────────────────
    workers    = workers or max(os.cpu_count() - 1, 1)
    futures_q  = queue.Queue(maxsize=max_inflight)
    done_q     = queue.PriorityQueue()            # (stack, sub, frames)

    def producer(executor):
        for idx, path in enumerate(tif_files):
            fut = executor.submit(_load_chunk, (idx, path), pages_per_chunk)
            futures_q.put(fut)                    # back‑pressure
        futures_q.put(None)                       # poison pill

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        threading.Thread(target=producer, args=(ex,), daemon=True).start()

        next_stack, next_sub = 0, 0
        pbar = tqdm(total=len(tif_files), desc="Writing AVI")

        while True:
            fut = futures_q.get()
            if fut is None:
                break
            for stack_idx, sub_idx, frames in fut.result():
                done_q.put((stack_idx, sub_idx, frames))

            # flush ready‑in‑order chunks
            while (not done_q.empty() and
                   done_q.queue[0][:2] == (next_stack, next_sub)):
                _, _, frs = done_q.get()
                for fr in frs:
                    vw.write(fr)
                next_sub += 1
                # finished one TIFF?
                if next_sub * pages_per_chunk >= len(
                        tf.TiffFile(tif_files[next_stack]).pages):
                    next_stack += 1
                    next_sub = 0
                    pbar.update(1)

        pbar.close()

    vw.release()
    if delete_tifs:
        import gc; gc.collect()
        for f in tif_files:
            try: os.remove(f)
            except OSError:
                print(f"⚠ couldn’t delete {f}")
    print("✅  Done:", output_avi)

def check_tif_stitch(tif_folder):
    # function to check if tif stitching results in identical data
    tif_folder = r"G:\LAYER 6\A10-FLEX_GcaMP_CC\SD2_whisker_day1_optoRec_cam"
    tif_files = natsorted(os.listdir(tif_folder))
    c1_stitched = [f for f in tif_files if '_C1_' in f and 'stitched' in f]
    c1_files = [f for f in tif_files if '_C1_' in f and 'stitched' not in f and '.tif' in f]

    # load the avi file into memory
    print("Reading AVI data from:", os.path.join(tif_folder, c1_stitched[0]))
    avi_data = read_full_avi(os.path.join(tif_folder,c1_stitched[0]))

    # load the tif files into memory
    tif_data = []
    for tif_file in c1_files:
        tif_path = os.path.join(tif_folder, tif_file)
        with tf.TiffFile(tif_path) as tif:
            for page in tif.pages:
                print(f"Reading page {page.index} from {tif_file}")
                tif_data.append(page.asarray())
    tif_data = np.array(tif_data)

    # now check that the data are identical
    if len(avi_data[0]) != len(tif_data):
        print("AVI and TIF data have different number of frames.")
        return False

    # take the average of the avi rows and columns
    avi_avg = np.mean(avi_data[0], axis=(0, 1))
    tif_avg = np.mean(tif_data, axis=(0, 1))
    diff_avi_tif = avi_avg - tif_avg

    plt.figure(figsize=(12, 6))
    plt.plot(diff_avi_tif, label='Difference between AVI and TIF averages')

# read AVI data - useful for image capture
def lazy_avi_loader(file_path):
    """
    Lazily loads an AVI file and yields frames one by one.

    Args:
        file_path (str): Path to the AVI file.

    Yields:
        numpy.ndarray: A single frame from the video as a NumPy array.
    """
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Cannot open the video file at {file_path}")

    # Lazily read and yield frames
    while True:
        ret, frame = cap.read()  # Read the next frame
        if not ret:  # End of video
            break
        yield frame  # Yield the frame as a NumPy array

    # Release the video capture object when done
    cap.release()

def read_full_avi(file_path):
    """
    Reads the full AVI file and loads all frames into memory.
    
    Args:
        file_path (str): Path to the AVI file.

    Returns:
        list: A list containing all frames as numpy arrays.
        dict: Metadata about the video, such as frame width, height, rate, and total frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Cannot open the video file at {file_path}")
    
    # Get video properties
    metadata = {
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_rate": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

    print(f"Video Properties: {metadata}")
    
    # Load all frames into memory
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Append each frame to the list
    
    # Release the video capture object
    cap.release()
    return frames, metadata


'''
    animals = "L6_A03"
    session = "SD1_d1"
    recording_type = "recordings_L5CT" #"recordings_panneuronal"

    animals = "L6-02"
    session = "SD1_whisker_d1"
    recording_type = 'recordings_panneuronal' #"recordings_L5CT" #"recordings_panneuronal"

    animals = "L6-02"
    session = "SH5"
    recording_type = 'recordings_panneuronal' #"recordings_L5CT" #"recordings_panneuronal"

    fname = os.path.join(r"F:\John\L6 Experiments",recording_type,animals,"sessions",session,"img","img.tif")
    bname = os.path.join(r"F:\John\L6 Experiments",recording_type,animals,"sessions",session,"beh")
    bfile = [i for i in os.listdir(bname) if '.h5' in i]
    assert len(bfile)==1, "This code does not support multiple episodes"
    bname = os.path.join(bname,bfile[0])
    
    data = h5py.File(bname,'r')

    print("Reading behavioral data from:",bname)
    bData = dict()
    for i in data['DI'].keys():
        bData[i] = data['DI'][i][:]
        if np.max(bData[i]) > 0:
            bData[i] = np.ravel(bData[i]/np.max(bData[i]))
    
    # Index of frame times for behavior
    frameIdx = np.where(np.diff(bData['FrameOut'],axis=0)==1)

    if check_piezo and len(imgpath) > 0:

        # piezo cycles
        piezo=data['AI']['PiezoMonitor'][:]
        piezo_norm = np.ravel(piezo/np.max(piezo)) # scale to 1 and convert to 1D

        assert bData['FrameOut'].shape[0]==piezo.shape[0], "Your piezo monitor and behavioral data are misaligned"

        # frames with piezo - this is where you find misalignments
        piezo_frames = np.ravel(piezo_norm[frameIdx])

        # can add one frame to this data
        img = tf.memmap(fname, mode='r')
        img.shape

        if len(piezo_frames) != (img.shape[0]*4):
            offset = len(piezo_frames)-(img.shape[0]*4)
            print("Your pre-downsampled data are off by",offset,"samples")

            # bc the piezo motor is mechanical, our rescaling to 0 and 1 should always give us similar answers
            # as such, here is our thresholding technique to find problematic time points
            idxMiss = np.ravel(np.where(piezo_frames < 0.35))

            fig, ax = plt.subplots(nrows=3,ncols=1)
            ax[0].plot(piezo_frames); ax[0].set_title("Full frame")
            ax[1].plot(piezo_frames); ax[1].set_xlim((0,100)); ax[1].set_title("First samples")
            ax[2].plot(piezo_frames); ax[2].set_xlim((len(piezo_frames)-100,len(piezo_frames))); ax[1].set_title("Last samples")

            shave_out = input("Should we shave off time points at the start or end? [start/end]")
            if shave_out == 'end':
                piezo_frames = piezo_frames[:-offset]
                frameIdx     = np.ravel(frameIdx)[:-offset]
            elif shave_out == 'start':
                piezo_frames = piezo_frames[offset::] # check
                frameIdx     = np.ravel(frameIdx)[offset::] # check

    # now get variables to save
    frameTimes = frameIdx[0::4] # every 4th datapoint because max_projection
    LEDStim    = bData['LEDStim'][frameTimes]
    irrelLR    = bData['irrelLROut'][frameTimes]
    trialLR    = bData['trialLROut'][frameTimes]
    setID      = bData['setIDOut'][frameTimes]
    trial      = bData['trialOut'][frameTimes]
    reward     = bData['rewardOut'][frameTimes]
    lickTimesL = bData['lickingLOut'][frameTimes]
    lickTimesR = bData['lickingROut'][frameTimes]
    behCam     = bData['BehaviorCam'][frameTimes]

    # build other parts of the matlab code here


    # SCRAP


    # if you convert with matlab, this matches perfectly, but if you convert with python it doesnt
    # this is because the very last frame is a blank framein matlab and python tosses it. This must have something
    # to do with the way the thor software works
    len_match = len(frameIdx) == img.shape[0] * 4
    if len_match == False:
        len_match = len(frameIdx) == (img.shape[0]+1) * 4
        # reassess whether the data are now a match
        if len_match == True:
            converter = 'python'
            rec_shape = (img.shape[0]+1) * 4

            # correct for offset
            img_offset = img.shape[0] * 4 - len(frameIdx)

            # correct the frameIdx
            print("Removing",img_offset,"samples from frameIdx to correct for blank frame in thorSync")
            frameIdx = frameIdx[:img_offset]
    else:
        converter = 'matlab'
        rec_shape = img.shape[0] * 4

    # TODO STOPPED HERE
    if len_match == False:
        print("Attempting to align your behavioral and imaging data")


        

        # frames with piezo - this is where you find misalignments
        piezo_frames = np.ravel(piezo_norm[frameIdx])







        if converter == 'python':
            # trim off n number of samples from piezeo_frames
            piezo_frames = piezo_frames + img_offset

        if len(piezo_frames) != rec_shape:
            offset = len(piezo_frames)-(img.shape[0]*4)
            print("Your pre-downsampled data are off by",offset,"samples")

            # bc the piezo motor is mechanical, our rescaling to 0 and 1 should always give us similar answers
            # as such, here is our thresholding technique to find problematic time points
            idxMiss = np.ravel(np.where(piezo_frames < 0.35))

            fig, ax = plt.subplots(nrows=3,ncols=1)
            ax[0].plot(piezo_frames); ax[0].set_title("Full frame")
            ax[1].plot(piezo_frames); ax[1].set_xlim((0,100)); ax[1].set_title("First samples")
            ax[2].plot(piezo_frames); ax[2].set_xlim((len(piezo_frames)-100,len(piezo_frames))); ax[1].set_title("Last samples")

            shave_out = input("Should we shave off time points at the start or end? [start/end]")
            if shave_out == 'end':
                piezo_frames = piezo_frames[:-offset]
                frameIdx     = np.ravel(frameIdx)[:-offset]
            elif shave_out == 'start':
                piezo_frames = piezo_frames[offset::] # check
                frameIdx     = np.ravel(frameIdx)[offset::] # check

'''
