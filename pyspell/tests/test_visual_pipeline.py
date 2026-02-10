
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Add package root to path
# Try to find 'pyspell' in path or add it relative to this script
current_dir = Path(__file__).resolve().parent
pyspell_root = current_dir.parent
if str(pyspell_root) not in sys.path:
    sys.path.insert(0, str(pyspell_root))

try:
    from cellregpy import (
        CellRegPy, 
        CellRegConfig, 
        MeanImageAligner,
        compute_data_distribution,
        compute_centroid_distances_model_custom,
        compute_spatial_correlations_model,
        initial_registration_centroid_distances_custom,
        cluster_cells
    )
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please run this script from the pyspell directory or ensure pyspell is in PYTHONPATH.")
    sys.exit(1)

def create_synthetic_session(n_cells=50, size=(256, 256), shift=(0, 0), theta=0):
    """Generates synthetic mean image and footprints."""
    mean_img = np.zeros(size, dtype=np.float32)
    footprints = np.zeros((n_cells, size[0], size[1]), dtype=np.float32)
    centroids = np.zeros((n_cells, 2))
    
    # Randomly place cells (same seed for consistency before transform)
    np.random.seed(42)
    base_centroids = np.random.rand(n_cells, 2) * [size[1]-40, size[0]-40] + [20, 20]
    
    # Transform centroids
    # Rotation
    rad = np.deg2rad(theta)
    c, s = np.cos(rad), np.sin(rad)
    rot_mat = np.array([[c, -s], [s, c]])
    center = np.array(size)/2
    
    for i in range(n_cells):
        # Apply transform to centroid
        pt = base_centroids[i] - center
        pt_rot = rot_mat @ pt + center
        pt_final = pt_rot + np.array(shift)
        
        centroids[i] = pt_final
        
        # Draw cell (Gaussian blob)
        cx, cy = int(pt_final[0]), int(pt_final[1])
        if 0 <= cx < size[1] and 0 <= cy < size[0]:
            y, x = np.ogrid[-10:11, -10:11]
            mask = x**2 + y**2 <= 25
            
            # fill footprint
            y_min, y_max = max(0, cy-10), min(size[0], cy+11)
            x_min, x_max = max(0, cx-10), min(size[1], cx+11)
            
            dy_min, dy_max = 10 - (cy - y_min), 10 + (y_max - cy)
            dx_min, dx_max = 10 - (cx - x_min), 10 + (x_max - cx)
            
            blob_mask = mask[10-(cy-y_min):10+(y_max-cy), 10-(cx-x_min):10+(x_max-cx)]
            
            footprints[i, y_min:y_max, x_min:x_max] = blob_mask.astype(float)
            
            # Add to mean image with some noise/texture
            mean_img[y_min:y_max, x_min:x_max] += blob_mask * np.random.uniform(0.5, 1.0)

    # Add background noise/structure
    bg = gaussian_filter(np.random.randn(*size), sigma=10)
    mean_img = mean_img + (bg - bg.min()) * 0.2
    
    return mean_img, footprints, centroids

def visualize_results(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Step 1: Generating Synthetic Data...")
    # Session 1: Reference
    mean1, fp1, cent1 = create_synthetic_session(shift=(0,0), theta=0)
    # Session 2: Moved (Shifted 10px, Rotated 5 degrees)
    mean2, fp2, cent2 = create_synthetic_session(shift=(10, 5), theta=5)
    
    # Simulating data loading
    mean_images = [mean1, mean2]
    spatial_footprints = [fp1, fp2]
    centroid_locations = [cent1, cent2] # These are ground truth, but we should re-compute them if being strict. 
                                        # But let's assume they are "loaded".
    
    # -------------------------------------------------------------
    # Step 2: Mean Image Alignment
    # -------------------------------------------------------------
    print("Step 2: Aligning Mean Images...")
    config = CellRegConfig()
    aligner = MeanImageAligner(config)
    
    # Config for viz
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title("Session 1 (Fixed)")
    plt.imshow(mean1, cmap='gray')
    plt.subplot(132)
    plt.title("Session 2 (Moving)")
    plt.imshow(mean2, cmap='gray')
    
    # Align
    fixed = mean1
    moving = mean2
    registered_img, method, peak, tform, best_filt, best_out = aligner.align(
        fixed, moving, filter_mode='highpass', outlier_mode='off'
    )
    
    plt.subplot(133)
    plt.title(f"Reg: {method} (corr={peak:.2f})")
    plt.imshow(registered_img, cmap='gray')
    plt.savefig(output_dir / '1_mean_image_alignment.png')
    print(f"  Alignment: {method}, Correlation: {peak:.4f}")

    # -------------------------------------------------------------
    # Step 3: Register Footprints/Centroids
    # -------------------------------------------------------------
    print("Step 3: Registering Centroids...")
    # Apply transform to centroids of Session 2
    from skimage import transform as sktransform
    
    # Transform centroids (session 2 -> session 1 frame)
    # Note: tform maps moving -> fixed. 
    # Points need to be transformed using the matrix.
    # scikit-image tform.inverse applies to images (pull-back), 
    # tform.params applies to forward coordinate mapping if defined typically?
    # Actually, in aligner code we used: warp_img(moving, tform) which implies tform is the forward map?
    # Wait, skimage warp(img, tform) uses inverse mapping. 
    # If aligner returned 'tform' such that warp(moving, tform) works, then tform describes coordinate map Fixed -> Moving (inverse).
    # Correct usage for POINTS from Moving -> Fixed is tform(points)? No.
    # Let's check how cellregpy does it in _register_fov:
    #   coords = ...
    #   transformed = (tform.params @ coords.T).T 
    # Let's verify this logic. 
    
    c2 = np.column_stack([cent2[:, 0], cent2[:, 1], np.ones(len(cent2))])
    # If tform was found by matching moving to fixed...
    # The transform found by phase_cross_correlation (via EuclideanTransform) usually gives expected shift.
    
    # Let's test apply
    cent2_reg = tform(cent2) # skimage transforms are callable on coords
    
    plt.figure(figsize=(8, 8))
    plt.title("Centroids: Fixed (Red) vs Moving (Green) vs Reg (Blue)")
    plt.scatter(cent1[:, 0], cent1[:, 1], c='r', marker='o', alpha=0.5, label='Fixed (S1)')
    plt.scatter(cent2[:, 0], cent2[:, 1], c='g', marker='x', alpha=0.5, label='Moving (S2)')
    plt.scatter(cent2_reg[:, 0], cent2_reg[:, 1], c='b', marker='+', alpha=0.9, label='Registered (S2)')
    plt.legend()
    plt.savefig(output_dir / '2_centroid_registration.png')
    
    # -------------------------------------------------------------
    # Step 4: Probabilistic Modeling
    # -------------------------------------------------------------
    print("Step 4: Probabilistic Modeling...")
    aligned_centroids = [cent1, cent2_reg]
    aligned_fps = [fp1, fp1] # skipping fp warp for brevity, just using placeholders as logic mostly uses centroids or correlations
    # Note: we need correlations to be valid for the model to work if using Spatial Correlation.
    # So we should actually warp footprints.
    
    # Warp footprints 2
    fp2_reg = np.zeros_like(fp2)
    for i in range(len(fp2)):
        fp2_reg[i] = sktransform.warp(fp2[i], tform.inverse, output_shape=fp1[0].shape, preserve_range=True)
        
    aligned_fps = [fp1, fp2_reg]
    
    data_dist = compute_data_distribution(aligned_fps, aligned_centroids, config.maximal_distance / config.microns_per_pixel)
    
    # Fit Centroid Model
    import pyspell.cellregpy as crp
    n_bins, centers = crp.estimate_num_bins(aligned_fps, 15/2.0)
    
    # We need centers to be a tuple (dist_centers, corr_centers)
    # estimate_num_bins returns (n, centers_corr). We need dist centers too.
    # Re-impl logic:
    max_dist = 15.0 / 2.0
    centers_dist = np.linspace(0, max_dist, n_bins)
    centers_corr = centers # from estimate_num_bins (0..1)
    
    (p_same_dist, same_pdf, diff_pdf, mix_pdf, intersect, fit_str, mse) = \
        compute_centroid_distances_model_custom(
            data_dist['neighbors_centroid_distances'],
            n_bins,
            (centers_dist, centers_corr)
        )
        
    plt.figure(figsize=(10, 5))
    plt.title(f"Centroid Distance Model: {fit_str}")
    # Plot histogram
    d = data_dist['neighbors_centroid_distances']
    plt.hist(d, bins=50, density=True, alpha=0.3, color='k', label='Data')
    plt.plot(centers_dist, same_pdf, 'g-', label='Same (Logn)')
    plt.plot(centers_dist, diff_pdf, 'r-', label='Diff (Logistic)')
    plt.plot(centers_dist, mix_pdf, 'b--', label='Mixture')
    plt.axvline(intersect, color='k', linestyle=':', label=f'Intersect {intersect:.2f}')
    plt.legend()
    plt.savefig(output_dir / '3_probabilistic_model.png')

    # -------------------------------------------------------------
    # Step 5: Clustering / Final Map
    # -------------------------------------------------------------
    print("Step 5: Clustering...")
    # Compute P_same tables
    p_same_dist_tbl, p_same_corr_tbl = crp.compute_p_same(
        data_dist['all_to_all_centroid_distances'],
        data_dist['all_to_all_spatial_correlations'],
        (centers_dist, centers_corr),
        p_same_dist,
        np.zeros_like(centers_corr) # placeholder if not using corr model
    )
    
    # Initial reg
    (cell_map, reg_dists, non_reg_dists, dist_map) = initial_registration_centroid_distances_custom(
        aligned_centroids, max_dist, intersect
    )
    
    # Cluster
    final_map, cl_cents, cl_scores = cluster_cells(
        cell_map,
        p_same_dist_tbl,
        data_dist['all_to_all_indexes'],
        max_dist,
        0.5, # threshold
        aligned_centroids
    )
    
    print(f"  Found {final_map.shape[0]} clusters.")
    print(f"  First 10 mappings:\n{final_map[:10]}")
    
    # Visualize Mappings
    plt.figure(figsize=(8, 8))
    plt.title("Final Registration Links")
    plt.scatter(cent1[:, 0], cent1[:, 1], c='r', marker='.', alpha=0.2)
    plt.scatter(cent2_reg[:, 0], cent2_reg[:, 1], c='g', marker='.', alpha=0.2)
    
    count = 0
    for r in range(final_map.shape[0]):
        idx1 = final_map[r, 0]
        idx2 = final_map[r, 1]
        
        if idx1 > 0 and idx2 > 0:
            c1 = cent1[idx1-1]
            c2 = cent2_reg[idx2-1]
            plt.plot([c1[0], c2[0]], [c1[1], c2[1]], 'k-', alpha=0.5, linewidth=0.5)
            count += 1
            
    plt.savefig(output_dir / '4_final_links.png')
    print(f"  Visualized {count} links.")
    print(f"Done! Check {output_dir} for images.")

if __name__ == "__main__":
    visualize_results("cellreg_visual_test_output")
