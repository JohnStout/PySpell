
import unittest
import numpy as np
from pathlib import Path
import sys
import shutil
import tempfile
import os

# Add package root to path
sys.path.append(r"c:\Users\johnj\SpellmanLab Dropbox\timspellman\Python\John\PySpell\pyspell")

try:
    from cellregpy import MeanImageAligner, CellRegConfig, CellRegPy, compute_spatial_correlation
except ImportError:
    # If not in path, try relative import if running from same dir
    try:
        from cellregpy import MeanImageAligner, CellRegConfig, CellRegPy, compute_spatial_correlation
    except ImportError:
        print("Could not import cellregpy. Please ensure it is in the PYTHONPATH.")
        sys.exit(1)

class TestCellRegPy(unittest.TestCase):
    
    def setUp(self):
        self.config = CellRegConfig()
        self.aligner = MeanImageAligner(self.config)
        
    def test_compute_spatial_correlation(self):
        # Test basic correlation
        a = np.random.rand(10, 10)
        b = a.copy()
        corr = compute_spatial_correlation(a, b)
        self.assertAlmostEqual(corr, 1.0)
        
        c = -a
        corr_neg = compute_spatial_correlation(a, c)
        self.assertAlmostEqual(corr_neg, -1.0)
        
    def test_mean_image_aligner_fundamentals(self):
        # specific bug check: min_area / strel_radius in suppress_outliers
        img = np.zeros((100, 100))
        # Add some "outliers" (bright spots)
        img[50, 50] = 100
        img[20:25, 20:25] = 50 
        
        # Should run without error
        out, mask = self.aligner._suppress_outliers(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(mask.shape, img.shape)
        
    def test_align_identity(self):
        fixed = np.random.rand(100, 100).astype(np.float32)
        moving = fixed.copy()
        
        reg, method, peak, tform, best_filt, best_out = self.aligner.align(
            fixed, moving, filter_mode='highpass', outlier_mode='off'
        )
        
        self.assertEqual(method, 'identity')
        # Correlation might not be exactly 1.0 due to filtering, but high
        self.assertGreater(peak, 0.95)
        
    def test_align_translation(self):
        fixed = np.zeros((100, 100))
        fixed[30:70, 30:70] = 1
        
        moving = np.zeros((100, 100))
        moving[40:80, 40:80] = 1 # Shifted by (10, 10)
        
        reg, method, peak, tform, best_filt, best_out = self.aligner.align(
            fixed, moving, filter_mode='highpass', outlier_mode='off'
        )
        
        # Ideally picks translation
        print(f"Detected method: {method}, peak: {peak}")
        self.assertTrue(method in ['translation', 'rigid', 'affine']) # Rigid/Affine can also capture translation
        self.assertGreater(peak, 0.8)

    def test_config_defaults(self):
        cfg = CellRegConfig()
        self.assertEqual(cfg.microns_per_pixel, 2.0)
        self.assertEqual(cfg.min_area, 25)

if __name__ == '__main__':
    unittest.main()
