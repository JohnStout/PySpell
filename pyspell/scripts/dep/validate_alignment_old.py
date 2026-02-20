
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add pyspell to path
try:
    script_dir = Path(__file__).resolve().parent
    pyspell_dir = script_dir.parent
    sys.path.insert(0, str(pyspell_dir))
except NameError:
    pyspell_dir = Path.cwd()
    sys.path.insert(0, str(pyspell_dir))

from pyspell.cellregpy import CellRegPy, CellRegConfig, list_session_folders, get_cellreg_files, get_mean_image

def validate_pair(folder_path, idx_fixed, idx_moving):
    print(f"\n--- Validating Pair: {idx_fixed} vs {idx_moving} ---")
    print(f"Mouse Folder: {folder_path}")
    
    # 1. Initialize
    config = CellRegConfig(figures_visibility='on')
    # config.use_parallel_processing = False # Parallel effectively enabled by default
    cellreg = CellRegPy(config)
    
    # 2. Load Images
    mouse_folder = Path(folder_path)
    plane0_folders = list_session_folders(mouse_folder)
    cellreg_files = get_cellreg_files(plane0_folders)
    
    if not cellreg_files:
        print("No CellReg files found!")
        return

    print(f"Found {len(cellreg_files)} sessions.")
    
    # Load mean images manually
    mean_images = []
    print("Loading mean images...")
    for sess in cellreg_files:
        mean_images.append(get_mean_image(sess.parent))
    
    if idx_fixed >= len(mean_images) or idx_moving >= len(mean_images):
        print(f"Indices out of bounds. Max index is {len(mean_images)-1}")
        return

    fixed = mean_images[idx_fixed]
    moving = mean_images[idx_moving]
    
    # 3. Plot Raw Input
    print("Plotting raw inputs...")
    cellreg.aligner.plot_image_pair(fixed, moving, title=f"Raw Input: Sess {idx_fixed} vs {idx_moving}")
    
    # 4. Run Alignment
    print("Running highpass alignment...")
    
    # Matches _check_alignment params
    registered, method, peak, tform, filter_mode, outliers = cellreg.aligner.align(
        fixed, 
        moving, 
        filter_mode='highpass', 
        outlier_mode='off',
        plot_fig=False 
    )
    
    # 5. Report Results
    print(f"\nOptimization Result:")
    print(f"  Method: {method}")
    print(f"  Peak Correlation: {peak:.5f}")
    print(f"  Transform: {tform.params if hasattr(tform, 'params') else tform}")
    print(f"  Threshold: {config.alignable_threshold}")
    
    is_alignable = peak > config.alignable_threshold
    print(f"  Result: {'ALIGNABLE' if is_alignable else 'NOT ALIGNABLE'}")
    
    # 6. Visualize Result
    print("Plotting registration result...")
    cellreg.aligner.plot_alignment_result(
        fixed, moving, registered, method, filter_mode, peak
    )

def validate_all_sessions(folder_path):
    print(f"\n--- Validating ALL Sessions in: {folder_path} ---")
    
    # 1. Initialize
    config = CellRegConfig(figures_visibility='on')
    # config.use_parallel_processing = False # Parallel enabled by default
    cellreg = CellRegPy(config)
    
    # 2. Load Images
    mouse_folder = Path(folder_path)
    plane0_folders = list_session_folders(mouse_folder)
    cellreg_files = get_cellreg_files(plane0_folders)
    
    if not cellreg_files:
        print("No CellReg files found!")
        return

    print(f"Found {len(cellreg_files)} sessions.")
    
    # Load mean images manually
    mean_images = []
    print("Loading mean images...")
    for sess in cellreg_files:
        mean_images.append(get_mean_image(sess.parent))
    
    # 3. Get Overview
    print("Running get_alignable_sessions...")
    alignable = cellreg.get_alignable_sessions(mean_images, plane0_folders)
    
    # 4. Check Quality
    print("Running check_alignment_quality...")
    cellreg.check_alignment_quality(alignable, mean_images)

def validate_cell_overlap(folder_path, idx_fixed, idx_moving):
    """
    Validates that the alignment transform actually aligns the CELLS (footprints).
    
    Uses cellregpy functions directly to ensure faithful reproduction.
    """
    print(f"\n--- Validating Cell Overlap: {idx_fixed} vs {idx_moving} ---")
    
    # Imports - use cellregpy functions directly
    from pyspell.cellregpy import (
        get_spatial_footprints, compute_footprint_projections, 
        list_session_folders, CellRegConfig, CellRegPy, get_mean_image,
        load_fall_mat, _norm01
    )
    from skimage import transform as sktransform
    
    # Setup
    config = CellRegConfig(figures_visibility='on')
    cellreg = CellRegPy(config)
    
    mouse_folder = Path(folder_path)
    plane0_folders = list_session_folders(mouse_folder)
    
    if idx_fixed >= len(plane0_folders) or idx_moving >= len(plane0_folders):
        print("Indices out of bounds.")
        return

    # 1. Load Mean Images using get_mean_image (matches production cellregpy)
    sess1_path = plane0_folders[idx_fixed]
    sess2_path = plane0_folders[idx_moving]
    
    # Get drift-corrected images (matches what cellregpy uses in production)
    img1 = get_mean_image(sess1_path, apply_drift_correction=True)
    img2 = get_mean_image(sess2_path, apply_drift_correction=True)
    
    # Also get raw for comparison
    img1_raw = get_mean_image(sess1_path, apply_drift_correction=False)
    img2_raw = get_mean_image(sess2_path, apply_drift_correction=False)
    
    # Get drift values for display
    fall1 = load_fall_mat(sess1_path)
    fall2 = load_fall_mat(sess2_path)
    ops1 = fall1.get('ops', {})
    ops2 = fall2.get('ops', {})
    dx1, dy1 = np.mean(ops1.get('xoff', [0])), np.mean(ops1.get('yoff', [0]))
    dx2, dy2 = np.mean(ops2.get('xoff', [0])), np.mean(ops2.get('yoff', [0]))
    print(f"  Session {idx_fixed+1} drift: dx={dx1:.2f}, dy={dy1:.2f}")
    print(f"  Session {idx_moving+1} drift: dx={dx2:.2f}, dy={dy2:.2f}")
    
    # Get registered mean images for optional display
    img1_reg = ops1.get('meanImgE', None)
    img2_reg = ops2.get('meanImgE', None)
    if img1_reg is not None: img1_reg = np.array(img1_reg)
    if img2_reg is not None: img2_reg = np.array(img2_reg)
    
    print(f"Loaded images. Using Drift-Corrected Mean Images (via get_mean_image).")
    
    print("Calculating alignment transform...")
    # Matches production alignment
    _, method, peak, tform, _, _ = cellreg.aligner.align(
        img1, img2, filter_mode='highpass', outlier_mode='off', plot_fig=False
    )
    print(f"Alignment Score: {peak:.3f} (Method: {method})")
    
    # DEBUG: Print transform parameters
    if tform is not None:
        print(f"Transform type: {type(tform).__name__}")
        print(f"Transform matrix:\n{tform.params}")
        # Extract translation
        if hasattr(tform, 'translation'):
            print(f"Translation: {tform.translation}")
        if hasattr(tform, 'rotation'):
            print(f"Rotation: {np.degrees(tform.rotation):.2f} degrees")
        if hasattr(tform, 'scale'):
            print(f"Scale: {tform.scale}")
    else:
        print("WARNING: Transform is None!")
    
    # 2. Load Spatial Footprints
    print("Loading spatial footprints (this may take a moment)...")
    
    def find_cellreg(p):
        # Look for *CellReg.mat or CellReg.mat
        matches = list(p.glob('*CellReg.mat'))
        if not matches:
            # Try case insensitive?
            matches = [x for x in p.iterdir() if 'cellreg.mat' in x.name.lower()]
        return matches[0] if matches else None

    f1 = find_cellreg(sess1_path)
    f2 = find_cellreg(sess2_path)
    
    if not f1 or not f2:
        print(f"Could not find *CellReg.mat in {sess1_path} or {sess2_path}")
        return

    fp1 = get_spatial_footprints(f1)
    fp2 = get_spatial_footprints(f2)
    
    print(f"Footprint 1 shape: {fp1.shape}")
    print(f"Footprint 2 shape: {fp2.shape}")
    print(f"Mean image 1 shape: {img1.shape}")
    print(f"Mean image 2 shape: {img2.shape}")
    
    # 3. Compute Projections
    # Simply sum normalized footprints
    print("Computing footprint projections...")
    projs = compute_footprint_projections([fp1, fp2])
    proj1 = projs[0]
    proj2 = projs[1]
    
    # 4. Warp Projection 2
    print("Warping Session 2 footprints...")
    proj2_warped = sktransform.warp(
        proj2,
        tform.inverse,  # Match production code
        output_shape=proj1.shape,
        order=1,
        preserve_range=True
    )
    
    # Also warp mean image 2 for comparison
    img2_warped = sktransform.warp(
        img2,
        tform.inverse,  # Match production code
        output_shape=img1.shape,
        order=1,
        preserve_range=True
    )
    
    # 5. Plot Overlay - BEFORE and AFTER comparison
    import matplotlib.pyplot as plt
    
    # Use cellregpy's _norm01 for normalization (handles NaNs properly)
    
    # Create RGB overlays
    p1n = _norm01(proj1)
    p2n_before = _norm01(proj2)  # Before warp
    p2n_after = _norm01(proj2_warped)  # After warp
    
    img1n = _norm01(img1)
    img2n_before = _norm01(img2)
    img2n_after = _norm01(img2_warped)
    
    # Footprint overlays
    rgb_before = np.zeros((p1n.shape[0], p1n.shape[1], 3))
    rgb_before[..., 0] = p1n
    rgb_before[..., 1] = p2n_before
    
    rgb_after = np.zeros((p1n.shape[0], p1n.shape[1], 3))
    rgb_after[..., 0] = p1n
    rgb_after[..., 1] = p2n_after
    
    # Mean image overlays
    mi_before = np.zeros((img1n.shape[0], img1n.shape[1], 3))
    mi_before[..., 0] = img1n
    mi_before[..., 1] = img2n_before
    
    mi_after = np.zeros((img1n.shape[0], img1n.shape[1], 3))
    mi_after[..., 0] = img1n
    mi_after[..., 1] = img2n_after
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(mi_before)
    axes[0, 0].set_title(f"Mean Images Corrected (Pre-Align)\n(Red=Sess{idx_fixed+1}, Grn=Sess{idx_moving+1})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mi_after)
    axes[0, 1].set_title(f"Mean Images ALIGNED\n(Should be yellow)")
    axes[0, 1].axis('off')
    
    # Row 2: Footprints
    axes[1, 0].imshow(rgb_before)
    axes[1, 0].set_title(f"Footprints RAW\n(Red=Sess{idx_fixed+1}, Grn=Sess{idx_moving+1})")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rgb_after)
    axes[1, 1].set_title(f"Footprints ALIGNED\n(Should be yellow)")
    axes[1, 1].axis('off')
    
    plt.suptitle(f"Alignment Validation (Score: {peak:.3f})", fontsize=14)
    plt.tight_layout()
    plt.show()

    # DEBUG: Plot Raw vs Reg for Session 1 to see differences
    if img1_reg is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img1_raw, cmap='gray')
        plt.title(f"Session {idx_fixed+1} Raw MeanImg")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(img1, cmap='gray') # img1 is corrected
        plt.title(f"Session {idx_fixed+1} Drift-Corrected")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(img1_reg, cmap='gray')
        plt.title(f"Session {idx_fixed+1} Registered MeanImgE")
        plt.axis('off')
        plt.tight_layout()
        plt.show()    


# ============================================================================ #
#                     PART A: ALIGNMENT STEP VALIDATION                        #
# ============================================================================ #

def validate_alignment_steps(folder_path, idx_fixed=0, idx_moving=1):
    """
    Validate each step of the alignment pipeline with visualizations.
    
    Steps:
    1. Input Loading & Drift Correction
    2. Outlier Suppression
    3. Filter Preprocessing (Highpass)
    4. Coarse Alignment (Phase Cross-Correlation)
    5. Transform Optimization
    6. Image Warping
    7. Footprint Warping
    8. Centroid Transformation
    
    All steps directly call cellregpy functions to ensure faithful reproduction.
    """
    from pyspell.cellregpy import (
        CellRegPy, CellRegConfig, get_mean_image, load_fall_mat, get_spatial_footprints,
        compute_footprint_projections, compute_centroids, list_session_folders,
        normalize_footprints, adjust_fov_size, _norm01, _rgb_overlay
    )
    from scipy.ndimage import gaussian_filter
    from skimage import transform as sktransform
    from skimage.registration import phase_cross_correlation
    
    print(f"\n{'='*60}")
    print("PART A: ALIGNMENT STEP VALIDATION")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}")
    print(f"Sessions: {idx_fixed} (fixed) vs {idx_moving} (moving)")
    
    config = CellRegConfig(figures_visibility='on')
    cellreg = CellRegPy(config)
    aligner = cellreg.aligner
    
    mouse_folder = Path(folder_path)
    plane0_folders = list_session_folders(mouse_folder)
    
    if idx_fixed >= len(plane0_folders) or idx_moving >= len(plane0_folders):
        print(f"Error: indices out of bounds. Max index: {len(plane0_folders)-1}")
        return
    
    sess1_path = plane0_folders[idx_fixed]
    sess2_path = plane0_folders[idx_moving]
    
    # =========== STEP 1: Input Loading & Drift Correction ===========
    # Uses get_mean_image() from cellregpy which now includes drift correction
    print("\n--- STEP 1: Input Loading & Drift Correction ---")
    print("  Using get_mean_image() with apply_drift_correction=True (matches cellregpy)")
    
    # Get drift-corrected images (matches what cellregpy uses in production)
    fixed = get_mean_image(sess1_path, apply_drift_correction=True)
    moving = get_mean_image(sess2_path, apply_drift_correction=True)
    
    # Also load raw for visualization comparison
    fixed_raw = get_mean_image(sess1_path, apply_drift_correction=False)
    moving_raw = get_mean_image(sess2_path, apply_drift_correction=False)
    
    # Get drift values for display
    fall1 = load_fall_mat(sess1_path)
    fall2 = load_fall_mat(sess2_path)
    ops1 = fall1.get('ops', {})
    ops2 = fall2.get('ops', {})
    dx1, dy1 = np.mean(ops1.get('xoff', [0])), np.mean(ops1.get('yoff', [0]))
    dx2, dy2 = np.mean(ops2.get('xoff', [0])), np.mean(ops2.get('yoff', [0]))
    
    print(f"  Session {idx_fixed+1} drift: dx={dx1:.2f}, dy={dy1:.2f}")
    print(f"  Session {idx_moving+1} drift: dx={dx2:.2f}, dy={dy2:.2f}")
    
    # Get registered mean images for comparison if available
    img1_reg = ops1.get('meanImgE', None)
    img2_reg = ops2.get('meanImgE', None)
    if img1_reg is not None: img1_reg = np.array(img1_reg)
    if img2_reg is not None: img2_reg = np.array(img2_reg)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (raw, corr, reg, title) in enumerate([
        (fixed_raw, fixed, img1_reg, f"Session {idx_fixed+1}"),
        (moving_raw, moving, img2_reg, f"Session {idx_moving+1}")
    ]):
        axes[i, 0].imshow(raw, cmap='gray')
        axes[i, 0].set_title(f"{title} Raw")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(np.nan_to_num(corr), cmap='gray')
        axes[i, 1].set_title(f"{title} Drift-Corrected")
        axes[i, 1].axis('off')
        
        if reg is not None:
            axes[i, 2].imshow(reg, cmap='gray')
            axes[i, 2].set_title(f"{title} MeanImgE")
        else:
            axes[i, 2].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[i, 2].axis('off')
    plt.suptitle("STEP 1: Input Loading & Drift Correction (via get_mean_image)")
    plt.tight_layout()
    plt.show()

    
    # =========== STEP 2: Outlier Suppression ===========
    print("\n--- STEP 2: Outlier Suppression ---")
    
    fixed_f = aligner._to_float(fixed)
    moving_f = aligner._to_float(moving)
    
    fixed_sup, mask_f = aligner._suppress_outliers(fixed_f)
    moving_sup, mask_m = aligner._suppress_outliers(moving_f)
    
    print(f"  Fixed outliers: {mask_f.sum()} pixels ({100*mask_f.mean():.2f}%)")
    print(f"  Moving outliers: {mask_m.sum()} pixels ({100*mask_m.mean():.2f}%)")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (orig, mask, sup, title) in enumerate([
        (fixed_f, mask_f, fixed_sup, "Fixed"),
        (moving_f, mask_m, moving_sup, "Moving")
    ]):
        vmin, vmax = np.nanpercentile(orig, [1, 99])
        axes[i, 0].imshow(np.nan_to_num(orig), cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{title} Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='Reds')
        axes[i, 1].set_title(f"{title} Outlier Mask")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(sup, cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"{title} Inpainted")
        axes[i, 2].axis('off')
    plt.suptitle("STEP 2: Outlier Suppression")
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 3: Filter Preprocessing ===========
    print("\n--- STEP 3: Filter Preprocessing (Highpass) ---")
    
    blur_hp = config.blur_hp
    fixed_blur = gaussian_filter(np.nan_to_num(fixed_sup), blur_hp)
    moving_blur = gaussian_filter(np.nan_to_num(moving_sup), blur_hp)
    fixed_hp = fixed_sup - fixed_blur
    moving_hp = moving_sup - moving_blur
    
    print(f"  Highpass blur sigma: {blur_hp}")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (orig, blur, hp, title) in enumerate([
        (fixed_sup, fixed_blur, fixed_hp, "Fixed"),
        (moving_sup, moving_blur, moving_hp, "Moving")
    ]):
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title(f"{title} (after outlier sup)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(blur, cmap='gray')
        axes[i, 1].set_title(f"{title} Blurred (σ={blur_hp})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(hp, cmap='gray')
        axes[i, 2].set_title(f"{title} Highpass")
        axes[i, 2].axis('off')
    plt.suptitle("STEP 3: Filter Preprocessing")
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 4: Coarse Alignment (Phase Cross-Correlation) ===========
    print("\n--- STEP 4: Coarse Alignment (Phase Cross-Correlation) ---")
    
    # Use highpass for coarse alignment
    fix_co = gaussian_filter(fixed_hp, 2)
    mov_co = gaussian_filter(moving_hp, 2)
    
    shift_detected, error, diffphase = phase_cross_correlation(fix_co, mov_co, upsample_factor=1)
    print(f"  Detected shift: dy={shift_detected[0]:.2f}, dx={shift_detected[1]:.2f}")
    
    # Create overlay showing shift (using cellregpy's _norm01)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_norm01(fix_co), cmap='gray')
    axes[0].set_title("Fixed (Highpass)")
    axes[0].axis('off')
    
    axes[1].imshow(_norm01(mov_co), cmap='gray')
    axes[1].set_title("Moving (Highpass)")
    axes[1].axis('off')
    
    # RGB overlay with shift arrow
    rgb = np.zeros((*fix_co.shape, 3))
    rgb[..., 1] = _norm01(fix_co)  # Green = fixed
    rgb[..., 0] = _norm01(mov_co)  # Red = moving
    axes[2].imshow(rgb)
    cy, cx = fix_co.shape[0]//2, fix_co.shape[1]//2
    axes[2].arrow(cx, cy, shift_detected[1]*5, shift_detected[0]*5, 
                  head_width=10, head_length=5, fc='yellow', ec='yellow', linewidth=2)
    axes[2].set_title(f"Overlay + Shift Vector\n(dy={shift_detected[0]:.1f}, dx={shift_detected[1]:.1f})")
    axes[2].axis('off')
    plt.suptitle("STEP 4: Coarse Alignment (Phase Cross-Correlation)")
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 5: Transform Optimization ===========
    print("\n--- STEP 5: Transform Optimization ---")
    
    # Run full alignment to get scores for all methods
    scores = {}
    transforms = {}
    for method in ['identity', 'translation', 'rigid', 'similarity', 'affine']:
        pc_hp, pc_lp, pc_bp, tform, ov = aligner._eval_one_combo(
            fixed_sup, moving_sup, method, outlier_mode=False,
            do_hp=True, do_lp=False, do_bp=False
        )
        scores[method] = pc_hp
        transforms[method] = tform
        print(f"  {method}: HP correlation = {pc_hp:.4f}")
    
    best_method = max(scores, key=scores.get)
    best_tform = transforms[best_method]
    best_score = scores[best_method]
    print(f"  Best: {best_method} (score={best_score:.4f})")
    
    # Plot scores
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(scores.keys())
    vals = [scores[m] for m in methods]
    colors = ['green' if m == best_method else 'steelblue' for m in methods]
    ax.barh(methods, vals, color=colors)
    ax.axvline(config.alignable_threshold, color='red', linestyle='--', label=f'Threshold ({config.alignable_threshold})')
    ax.set_xlabel("Highpass Correlation Score")
    ax.set_title("STEP 5: Transform Optimization Scores")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 6: Image Warping ===========
    print("\n--- STEP 6: Image Warping ---")
    
    if best_tform is not None:
        print(f"  Transform type: {type(best_tform).__name__}")
        print(f"  Transform matrix:\n{best_tform.params}")
        
        # Use aligner._apply_transform to match production code exactly
        moving_warped = aligner._apply_transform(moving_sup, best_tform, fixed_sup.shape)
    else:
        print("  WARNING: No transform found, using identity")
        moving_warped = moving_sup.copy()
    
    # Visualize using cellregpy's _norm01 and _rgb_overlay
    f_n = _norm01(fixed_sup)
    m_n_before = _norm01(moving_sup)
    m_n_after = _norm01(moving_warped)
    
    rgb_before = np.zeros((*f_n.shape, 3))
    rgb_before[..., 0] = m_n_before  # Red = moving
    rgb_before[..., 1] = f_n         # Green = fixed
    
    rgb_after = np.zeros((*f_n.shape, 3))
    rgb_after[..., 0] = m_n_after
    rgb_after[..., 1] = f_n
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(fixed_sup, cmap='gray')
    axes[0].set_title("Fixed")
    axes[0].axis('off')
    
    axes[1].imshow(moving_sup, cmap='gray')
    axes[1].set_title("Moving (Before)")
    axes[1].axis('off')
    
    axes[2].imshow(rgb_before)
    axes[2].set_title("Overlay BEFORE\n(Red=Moving, Green=Fixed)")
    axes[2].axis('off')
    
    axes[3].imshow(rgb_after)
    axes[3].set_title("Overlay AFTER Warp\n(Yellow = Aligned)")
    axes[3].axis('off')
    
    plt.suptitle("STEP 6: Image Warping")
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 7: Footprint Warping ===========
    print("\n--- STEP 7: Footprint Warping ---")
    
    # Find CellReg files
    def find_cellreg(p):
        matches = list(p.glob('*CellReg.mat'))
        if not matches:
            matches = [x for x in p.iterdir() if 'cellreg.mat' in x.name.lower()]
        return matches[0] if matches else None
    
    f1 = find_cellreg(sess1_path)
    f2 = find_cellreg(sess2_path)
    
    if f1 and f2 and best_tform is not None:
        fp1 = get_spatial_footprints(f1)
        fp2 = get_spatial_footprints(f2)
        
        # Warp footprints using aligner._apply_transform (matches cellregpy _register_fov)
        print(f"  Warping {fp2.shape[0]} footprints from session {idx_moving+1}...")
        fp2_warped = np.zeros_like(fp2)
        for c in range(fp2.shape[0]):
            fp2_warped[c] = sktransform.warp(
                fp2[c], best_tform.inverse,
                output_shape=fp2[c].shape,
                order=1, preserve_range=True,
                mode='constant', cval=0.0  # Match cellregpy: cval=0.0 for footprints
            )
        
        proj1 = compute_footprint_projections([fp1])[0]
        proj2_before = compute_footprint_projections([fp2])[0]
        proj2_after = compute_footprint_projections([fp2_warped])[0]
        
        # Use cellregpy's _norm01 for normalization
        p1n = _norm01(proj1)
        p2n_before = _norm01(proj2_before)
        p2n_after = _norm01(proj2_after)
        
        rgb_fp_before = np.zeros((*p1n.shape, 3))
        rgb_fp_before[..., 0] = p1n
        rgb_fp_before[..., 1] = p2n_before
        
        rgb_fp_after = np.zeros((*p1n.shape, 3))
        rgb_fp_after[..., 0] = p1n
        rgb_fp_after[..., 1] = p2n_after
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_fp_before)
        axes[0].set_title("Footprints BEFORE Warp\n(Red=Sess1, Green=Sess2)")
        axes[0].axis('off')
        
        axes[1].imshow(rgb_fp_after)
        axes[1].set_title("Footprints AFTER Warp\n(Yellow = Aligned)")
        axes[1].axis('off')
        plt.suptitle("STEP 7: Footprint Warping")
        plt.tight_layout()
        plt.show()
        
        # =========== STEP 8: Centroid Transformation ===========
        print("\n--- STEP 8: Centroid Transformation ---")
        
        cents1 = compute_centroids([fp1], config.microns_per_pixel)[0]
        cents2 = compute_centroids([fp2], config.microns_per_pixel)[0]
        
        # Transform centroids
        if len(cents2) > 0:
            coords = np.column_stack([cents2[:, 0], cents2[:, 1], np.ones(len(cents2))])
            cents2_warped = (best_tform.params @ coords.T).T[:, :2]
        else:
            cents2_warped = cents2.copy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(cents1[:, 0], cents1[:, 1], c='red', s=20, alpha=0.6, label=f'Session {idx_fixed+1}')
        ax.scatter(cents2[:, 0], cents2[:, 1], c='blue', s=20, alpha=0.3, label=f'Session {idx_moving+1} (Before)')
        ax.scatter(cents2_warped[:, 0], cents2_warped[:, 1], c='green', s=20, alpha=0.6, label=f'Session {idx_moving+1} (After)')
        
        # Draw lines connecting before/after
        for j in range(min(50, len(cents2))):
            ax.plot([cents2[j, 0], cents2_warped[j, 0]], 
                   [cents2[j, 1], cents2_warped[j, 1]], 'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(0, fp1.shape[2])
        ax.set_ylim(fp1.shape[1], 0)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title("STEP 8: Centroid Transformation")
        plt.tight_layout()
        plt.show()
    else:
        print("  Skipping footprint/centroid steps (CellReg.mat not found or no transform)")
    
    print("\n✓ Part A: Alignment validation complete!")
    return best_tform, best_score, best_method


# ============================================================================ #
#                     PART B: MODELING STEP VALIDATION                         #
# ============================================================================ #

def validate_modeling_steps(folder_path, idx_fixed=0, idx_moving=1):
    """
    Validate each step of the probabilistic modeling pipeline.
    
    Steps:
    9.  Data Distribution
    10. Centroid Distance Model
    11. Spatial Correlation Model
    12. Model Selection
    13. P(same) Curves
    14. Initial Registration
    15. Clustering
    16. Accuracy Estimation
    """
    from pyspell.cellregpy import (
        CellRegPy, CellRegConfig, list_session_folders, get_spatial_footprints,
        compute_centroids, compute_data_distribution, compute_footprint_projections,
        compute_centroid_distances_model_custom, compute_spatial_correlations_model,
        choose_best_model, compute_p_same, initial_registration_spatial_corr,
        cluster_cells, estimate_registration_accuracy, normalize_footprints,
        adjust_fov_size, estimate_num_bins
    )
    from skimage import transform as sktransform
    
    print(f"\n{'='*60}")
    print("PART B: MODELING STEP VALIDATION")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}")
    
    config = CellRegConfig(figures_visibility='on')
    cellreg = CellRegPy(config)
    
    mouse_folder = Path(folder_path)
    plane0_folders = list_session_folders(mouse_folder)
    
    # Find CellReg files
    def find_cellreg(p):
        matches = list(p.glob('*CellReg.mat'))
        if not matches:
            matches = [x for x in p.iterdir() if 'cellreg.mat' in x.name.lower()]
        return matches[0] if matches else None
    
    cellreg_files = []
    for p in plane0_folders:
        f = find_cellreg(p)
        if f:
            cellreg_files.append(f)
    
    if len(cellreg_files) < 2:
        print("Error: Need at least 2 sessions with CellReg.mat files")
        return
    
    print(f"Found {len(cellreg_files)} sessions with CellReg.mat")
    
    # Load footprints
    print("\nLoading spatial footprints...")
    spatial_footprints = []
    for f in cellreg_files[:min(4, len(cellreg_files))]:  # Limit for speed
        fp = get_spatial_footprints(f)
        spatial_footprints.append(fp)
    
    # Normalize and adjust
    print("Normalizing footprints...")
    normalized_fps = normalize_footprints(spatial_footprints)
    adjusted_fps, adj_fov, adj_x, adj_y, padding = adjust_fov_size(normalized_fps)
    
    # Compute centroids
    print("Computing centroids...")
    centroid_locations = compute_centroids(adjusted_fps, config.microns_per_pixel)
    
    # =========== STEP 9: Data Distribution ===========
    print("\n--- STEP 9: Data Distribution ---")
    
    max_dist_px = config.maximal_distance / config.microns_per_pixel
    data_dist = compute_data_distribution(adjusted_fps, centroid_locations, max_dist_px)
    
    neighbor_dists = data_dist['neighbors_centroid_distances']
    neighbor_corrs = data_dist['neighbors_spatial_correlations']
    
    print(f"  Neighbor pairs: {len(neighbor_dists)}")
    print(f"  Distance range: [{np.min(neighbor_dists):.2f}, {np.max(neighbor_dists):.2f}] px")
    print(f"  Correlation range: [{np.min(neighbor_corrs):.3f}, {np.max(neighbor_corrs):.3f}]")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(neighbor_dists, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Centroid Distance (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Neighbor Distance Distribution")
    
    axes[1].hist(neighbor_corrs, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel("Spatial Correlation")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Neighbor Correlation Distribution")
    
    plt.suptitle("STEP 9: Data Distribution")
    plt.tight_layout()
    plt.show()
    
    # =========== STEPS 10-11: Model Fitting ===========
    print("\n--- STEPS 10-11: Model Fitting ---")
    
    number_of_bins, _ = estimate_num_bins(adjusted_fps, max_dist_px)
    centers_of_bins = (
        np.linspace(0, max_dist_px, number_of_bins, dtype=np.float64),
        np.linspace(0, 1, number_of_bins, dtype=np.float64),
    )
    
    # Centroid distance model
    (p_same_centroid, cent_same, cent_diff, cent_mix, cent_int, cent_best, cent_mse) = \
        compute_centroid_distances_model_custom(neighbor_dists, number_of_bins, centers_of_bins)
    
    # Spatial correlation model
    (p_same_corr, corr_same, corr_diff, corr_mix, corr_int, corr_best, corr_mse) = \
        compute_spatial_correlations_model(neighbor_corrs, number_of_bins, centers_of_bins)
    
    print(f"  Centroid model intersection: {cent_int:.2f} px")
    print(f"  Correlation model intersection: {corr_int:.3f}")
    print(f"  Centroid overlap MSE: {cent_mse:.4f}")
    print(f"  Correlation overlap MSE: {corr_mse:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distance histogram + model
    axes[0, 0].hist(neighbor_dists, bins=number_of_bins, density=True, alpha=0.5, label='Data')
    x_dist = centers_of_bins[0]
    axes[0, 0].plot(x_dist, cent_same, 'g-', linewidth=2, label='Same-cell model')
    axes[0, 0].plot(x_dist, cent_diff, 'r-', linewidth=2, label='Diff-cell model')
    axes[0, 0].plot(x_dist, cent_mix, 'b--', linewidth=2, label='Mixture')
    axes[0, 0].axvline(cent_int, color='purple', linestyle=':', label=f'Intersection={cent_int:.1f}')
    axes[0, 0].set_xlabel("Centroid Distance (px)")
    axes[0, 0].set_title("STEP 10: Centroid Distance Model")
    axes[0, 0].legend()
    
    # P(same) for distance
    axes[0, 1].plot(x_dist, p_same_centroid, 'b-', linewidth=2)
    axes[0, 1].axhline(0.5, color='gray', linestyle='--')
    axes[0, 1].axvline(cent_int, color='purple', linestyle=':')
    axes[0, 1].set_xlabel("Centroid Distance (px)")
    axes[0, 1].set_ylabel("P(same)")
    axes[0, 1].set_title("P(same) from Distance")
    
    # Correlation histogram + model
    axes[1, 0].hist(neighbor_corrs, bins=number_of_bins, density=True, alpha=0.5, label='Data')
    x_corr = centers_of_bins[1]
    axes[1, 0].plot(x_corr, corr_same, 'g-', linewidth=2, label='Same-cell model')
    axes[1, 0].plot(x_corr, corr_diff, 'r-', linewidth=2, label='Diff-cell model')
    axes[1, 0].plot(x_corr, corr_mix, 'b--', linewidth=2, label='Mixture')
    axes[1, 0].axvline(corr_int, color='purple', linestyle=':', label=f'Intersection={corr_int:.2f}')
    axes[1, 0].set_xlabel("Spatial Correlation")
    axes[1, 0].set_title("STEP 11: Spatial Correlation Model")
    axes[1, 0].legend()
    
    # P(same) for correlation
    axes[1, 1].plot(x_corr, p_same_corr, 'b-', linewidth=2)
    axes[1, 1].axhline(0.5, color='gray', linestyle='--')
    axes[1, 1].axvline(corr_int, color='purple', linestyle=':')
    axes[1, 1].set_xlabel("Spatial Correlation")
    axes[1, 1].set_ylabel("P(same)")
    axes[1, 1].set_title("P(same) from Correlation")
    
    plt.suptitle("STEPS 10-11: Model Fitting + STEP 13: P(same) Curves")
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 12: Model Selection ===========
    print("\n--- STEP 12: Model Selection ---")
    
    model_used = choose_best_model(cent_mse, corr_mse, 
                                    centroid_intersection=cent_int, 
                                    corr_intersection=corr_int)
    print(f"  Selected model: {model_used}")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    mses = [cent_mse, corr_mse]
    labels = ['Centroid Distance', 'Spatial Correlation']
    colors = ['green' if labels[i].startswith(model_used[:4]) else 'steelblue' for i in range(2)]
    ax.bar(labels, mses, color=colors)
    ax.set_ylabel("Overlap MSE (lower = better)")
    ax.set_title("STEP 12: Model Selection")
    plt.tight_layout()
    plt.show()
    
    # =========== STEPS 14-16: Registration & Clustering ===========
    print("\n--- STEPS 14-16: Registration & Clustering ---")
    
    if model_used == "Spatial correlation":
        initial_threshold = corr_int if np.isfinite(corr_int) else config.sufficient_correlation_footprints
        
        (cell_map, reg_corrs, non_reg_corrs, corr_map) = initial_registration_spatial_corr(
            adjusted_fps, max_dist_px, initial_threshold
        )
        
        print(f"  Initial registration: {cell_map.shape[0]} clusters")
        print(f"  Registered pairs: {len(reg_corrs)}, Non-registered: {len(non_reg_corrs)}")
        
        # Compute p_same for clustering
        p_same_cent, p_same_spat = compute_p_same(
            data_dist['all_to_all_centroid_distances'],
            data_dist['all_to_all_spatial_correlations'],
            centers_of_bins,
            p_same_centroid,
            p_same_corr
        )
        
        # Cluster
        cell_map_opt, reg_cents, cluster_scores = cluster_cells(
            cell_map, p_same_spat, data_dist['all_to_all_indexes'],
            max_dist_px, config.p_same_threshold, centroid_locations
        )
        
        print(f"  After clustering: {cell_map_opt.shape[0]} clusters")
        
        # Accuracy
        p_same_vec, p_diff_vec, acc_scores = estimate_registration_accuracy(
            cell_map_opt, p_same_spat, data_dist['all_to_all_indexes'], config.p_same_threshold
        )
        
        print(f"  Mean P(same) for registered pairs: {np.mean(p_same_vec):.3f}")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Cell-to-index map
        im = axes[0].imshow(cell_map_opt[:min(100, cell_map_opt.shape[0])], aspect='auto', cmap='viridis')
        axes[0].set_xlabel("Session")
        axes[0].set_ylabel("Cluster ID")
        axes[0].set_title(f"STEP 15: cell_to_index_map\n({cell_map_opt.shape[0]} clusters)")
        plt.colorbar(im, ax=axes[0], label='Cell Index')
        
        # Registered vs non-registered correlations
        axes[1].hist(reg_corrs, bins=30, alpha=0.6, label='Registered', color='green')
        axes[1].hist(non_reg_corrs, bins=30, alpha=0.6, label='Non-registered', color='red')
        axes[1].set_xlabel("Spatial Correlation")
        axes[1].set_title("STEP 14: Initial Registration")
        axes[1].legend()
        
        # P(same) distribution
        axes[2].hist(p_same_vec, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(config.p_same_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({config.p_same_threshold})')
        axes[2].set_xlabel("P(same)")
        axes[2].set_title(f"STEP 16: Accuracy\n(mean P(same)={np.mean(p_same_vec):.3f})")
        axes[2].legend()
        
        plt.suptitle("STEPS 14-16: Registration, Clustering & Accuracy")
        plt.tight_layout()
        plt.show()
    else:
        print("  (Skipping - would use centroid distance model)")
    
    print("\n✓ Part B: Modeling validation complete!")


# ============================================================================ #
#                     PART C: FULL PIPELINE VALIDATION                         #
# ============================================================================ #

def validate_full_pipeline(folder_path):
    """
    Validate the full multi-session pipeline:
    
    Steps:
    17. Seed Selection: Test each session as reference
    18. Redundancy Removal: Remove duplicate FOV groupings
    19. Transitive Alignment: Build union-find across FOVs
    20. Final Table: Create mouse_table with cellRegID assignments
    """
    from pyspell.cellregpy import (
        CellRegPy, CellRegConfig, list_session_folders, get_cellreg_files,
        get_mean_image, get_iscell
    )
    import networkx as nx
    
    print(f"\n{'='*60}")
    print("PART C: FULL PIPELINE VALIDATION")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}")
    
    config = CellRegConfig(figures_visibility='on')
    cellreg = CellRegPy(config)
    
    mouse_folder = Path(folder_path)
    plane0_folders = list_session_folders(mouse_folder)
    sess_fovs = get_cellreg_files(plane0_folders)
    
    if len(sess_fovs) < 2:
        print("Error: Need at least 2 sessions")
        return
    
    print(f"Found {len(sess_fovs)} sessions with CellReg.mat")
    
    # Load mean images
    print("\nLoading mean images...")
    mean_images = []
    for sess in sess_fovs:
        plane0_path = sess.parent
        mean_images.append(get_mean_image(plane0_path))
    
    # =========== STEP 17: Seed Selection ===========
    print("\n--- STEP 17: Seed Selection (Each Session as Reference) ---")
    
    alignable = cellreg.get_alignable_sessions(mean_images, sess_fovs)
    
    # Display alignment matrix
    n = len(sess_fovs)
    corr_matrix = np.zeros((n, n))
    for i, all_corr in enumerate(alignable['all_correlations']):
        corr_matrix[i, :] = all_corr
    
    print(f"  Alignment correlation matrix shape: {corr_matrix.shape}")
    print(f"  Mean correlation: {np.mean(corr_matrix):.3f}")
    print(f"  Min/Max: {np.min(corr_matrix):.3f} / {np.max(corr_matrix):.3f}")
    
    # Sessions alignable per seed
    for i, idx_aligned in enumerate(alignable['index_aligned']):
        n_aligned = len(idx_aligned)
        print(f"  Seed {i+1}: {n_aligned}/{n} sessions alignable")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    im = axes[0].imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_xlabel("Moving Session")
    axes[0].set_ylabel("Reference Session")
    axes[0].set_title("STEP 17: Alignment Correlation Matrix\n(Each Session as Reference)")
    plt.colorbar(im, ax=axes[0], label='Correlation')
    axes[0].axhline(config.alignable_threshold, color='red', linestyle='--', alpha=0.5)
    
    # Threshold mask
    thresh_mask = corr_matrix > config.alignable_threshold
    axes[1].imshow(thresh_mask, cmap='Greens')
    axes[1].set_xlabel("Moving Session")
    axes[1].set_ylabel("Reference Session")
    axes[1].set_title(f"Alignable Pairs (threshold={config.alignable_threshold})")
    for i in range(n):
        for j in range(n):
            if thresh_mask[i, j]:
                axes[1].text(j, i, '✓', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 18: Redundancy Removal ===========
    print("\n--- STEP 18: Redundancy Removal ---")
    
    # Show original FOV groupings
    print(f"  Before removal: {len(alignable['session_names'])} FOV groups")
    for i, sess_list in enumerate(alignable['session_names']):
        idx_set = set(alignable['index_aligned'][i].tolist())
        med_corr = np.median(alignable['correlations'][i]) if len(alignable['correlations'][i]) > 0 else 0
        print(f"    FOV {i+1}: {len(sess_list)} sessions, indices={idx_set}, median_corr={med_corr:.3f}")
    
    # Apply redundancy removal
    filtered = cellreg._remove_redundancies(alignable)
    
    print(f"  After removal: {len(filtered['session_names'])} unique FOV groups")
    for i, sess_list in enumerate(filtered['session_names']):
        idx_set = set(filtered['index_aligned'][i].tolist())
        med_corr = np.median(filtered['correlations'][i]) if len(filtered['correlations'][i]) > 0 else 0
        print(f"    FOV {i+1}: {len(sess_list)} sessions, indices={idx_set}, median_corr={med_corr:.3f}")
    
    # Visualize as graph
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before removal
    G_before = nx.Graph()
    for i in range(n):
        G_before.add_node(i, label=f"S{i+1}")
    for i, idx_aligned in enumerate(alignable['index_aligned']):
        for j in idx_aligned:
            if i != j:
                G_before.add_edge(i, j)
    
    pos = nx.spring_layout(G_before, seed=42)
    nx.draw(G_before, pos, ax=axes[0], with_labels=True, 
            node_color='lightblue', node_size=500, font_size=10,
            labels={i: f"S{i+1}" for i in range(n)})
    axes[0].set_title(f"Before Removal\n({len(alignable['session_names'])} groups)")
    
    # After removal - show which FOV each session belongs to
    fov_colors = plt.cm.Set3(np.linspace(0, 1, max(1, len(filtered['session_names']))))
    node_colors = ['lightgray'] * n
    for fov_i, idx_aligned in enumerate(filtered['index_aligned']):
        for sess_idx in idx_aligned:
            node_colors[sess_idx] = fov_colors[fov_i]
    
    nx.draw(G_before, pos, ax=axes[1], with_labels=True,
            node_color=node_colors, node_size=500, font_size=10,
            labels={i: f"S{i+1}" for i in range(n)})
    axes[1].set_title(f"After Removal\n({len(filtered['session_names'])} unique groups, colors=FOVs)")
    
    plt.suptitle("STEP 18: Redundancy Removal")
    plt.tight_layout()
    plt.show()
    
    # =========== STEP 19-20: Transitive Alignment & Table ===========
    print("\n--- STEPS 19-20: Transitive Alignment & Final Table ---")
    
    # Run the full pipeline to get mouse_data and mouse_table
    print("  Running full CellRegPy pipeline...")
    results_dir = mouse_folder / '1_CellReg'
    
    # Check if results already exist
    table_path = results_dir / 'mouse_table.pkl'
    data_path = results_dir / 'mouse_data.npy'
    
    if table_path.exists():
        print(f"  Loading existing results from {results_dir}")
        import pandas as pd
        mouse_table = pd.read_pickle(table_path)
        mouse_data = np.load(data_path, allow_pickle=True).item()
    else:
        print("  No existing results found - running pipeline (this may take a while)...")
        cellreg.run([mouse_folder])  # run() expects a list of folders
        if table_path.exists():
            mouse_table = pd.read_pickle(table_path)
            mouse_data = np.load(data_path, allow_pickle=True).item()
        else:
            print("  ERROR: Pipeline did not produce results")
            return
    
    print(f"\n  Final table shape: {mouse_table.shape}")
    print(f"  Columns: {list(mouse_table.columns)}")
    
    # Summary statistics
    n_cells_total = len(mouse_table)
    n_unique_cellreg = mouse_table['cellRegID'].nunique()
    sessions = mouse_table['Session'].unique()
    
    print(f"\n  Total cell instances: {n_cells_total}")
    print(f"  Unique cellRegIDs: {n_unique_cellreg}")
    print(f"  Sessions: {len(sessions)}")
    
    # Cells per session
    cells_per_session = mouse_table.groupby('Session').size()
    print(f"\n  Cells per session:")
    for sess, count in cells_per_session.items():
        print(f"    {sess}: {count}")
    
    # CellRegID distribution (how many sessions each cell appears in)
    cellreg_counts = mouse_table[mouse_table['cellRegID'] > 0].groupby('cellRegID').size()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cells per session bar chart
    axes[0].bar(range(len(cells_per_session)), cells_per_session.values)
    axes[0].set_xticks(range(len(cells_per_session)))
    axes[0].set_xticklabels([f"S{i+1}" for i in range(len(cells_per_session))], rotation=45)
    axes[0].set_ylabel("Number of Cells")
    axes[0].set_title("Cells per Session")
    
    # CellRegID occurrence histogram
    axes[1].hist(cellreg_counts.values, bins=range(1, len(sessions)+2), 
                 edgecolor='black', alpha=0.7, align='left')
    axes[1].set_xlabel("Number of Sessions")
    axes[1].set_ylabel("Number of Cells")
    axes[1].set_title("Cell Tracking Across Sessions\n(How many sessions each cell appears in)")
    
    # Pivot table visualization (top 50 cellRegIDs)
    if n_unique_cellreg > 0:
        top_ids = cellreg_counts.nlargest(min(50, len(cellreg_counts))).index
        pivot_data = mouse_table[mouse_table['cellRegID'].isin(top_ids)]
        pivot = pivot_data.pivot_table(
            index='cellRegID', columns='Session', values='suite2pID', 
            aggfunc='first', fill_value=0
        )
        im = axes[2].imshow(pivot.values > 0, aspect='auto', cmap='Greens')
        axes[2].set_xlabel("Session")
        axes[2].set_ylabel("cellRegID (top 50)")
        axes[2].set_title("STEP 20: Final Cell Registration Map\n(Green = Cell Present)")
    
    plt.suptitle("STEPS 19-20: Transitive Alignment & Final Table")
    plt.tight_layout()
    plt.show()
    
    # Show sample of the table
    print("\n  Sample of mouse_table:")
    print(mouse_table.head(20).to_string())
    
    print("\n✓ Part C: Full pipeline validation complete!")
    return mouse_table, mouse_data


if __name__ == "__main__":

    # USER CONFIGURATION
    target_folder = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_superalignment\L612_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN"
    target_folder = r"C:\Users\spell\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_superalignment\L612_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN"
    
    # --- TOGGLE MODES ---
    RUN_MEAN_ALIGNMENT_CHECK = True   # Original: checks mean image correlations
    RUN_CELL_OVERLAP_CHECK   = True   # Original: checks footprint overlap
    RUN_STEP_BY_STEP_ALIGN   = True   # Part A: step-by-step alignment validation
    RUN_STEP_BY_STEP_MODEL   = True   # Part B: step-by-step modeling validation
    RUN_FULL_PIPELINE        = True    # Part C: full multi-session pipeline validation
    
    print(f"Target: {target_folder}\n")
    
    if RUN_MEAN_ALIGNMENT_CHECK:
        validate_all_sessions(target_folder)
        
    if RUN_CELL_OVERLAP_CHECK:
        validate_cell_overlap(target_folder, 0, 8)
    
    if RUN_STEP_BY_STEP_ALIGN:
        validate_alignment_steps(target_folder, 0, 1)
    
    if RUN_STEP_BY_STEP_MODEL:
        validate_modeling_steps(target_folder)
    
    if RUN_FULL_PIPELINE:
        validate_full_pipeline(target_folder)
