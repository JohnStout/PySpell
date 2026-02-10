# SingleSession_FullPipeline.py
"""
Quick script to convert, run suite2p, and postprocess a single session.
Extracts the core logic from recurseConvert.py for one-off processing.

John Stout - 2/9/2026
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import tifffile as tf
import thorfuns
import s2pfuns
from s2pfuns import cellClassifier
import sessreg
import rootfun as rf
'''
import importlib
import thorfuns
importlib.reload(thorfuns)
'''
# ============================================================================ #
# ========================= USER INPUT ======================================= #
# ============================================================================ #

# Path to the session folder containing Image----.raw file
session_path = r"Z:\John\Subjects - GCaMP Recordings\L628_M_mdlxGCaMP_L6Chrimson\SDswitch_day13_fov3\SDswitch_day13_fov3_img"

# Set to True if this is an opto session (will run LED artifact correction)
is_opto = True

# Replace existing files if they exist?
replace_img = True
replace_s2p = True

# Test upsampling? Saves only 1000 frames to img_upsample_test.tif for Fiji verification
test_upsample = False  # Set to False for full conversion

# ============================================================================ #
# ========================= PIPELINE ========================================= #
# ============================================================================ #

if __name__ == '__main__':
    
    print(f"\n{'='*60}")
    print(f"Processing session: {session_path}")
    print(f"{'='*60}\n")
    
    dir_contents = os.listdir(session_path)
    
    # Check for raw file
    raw_search = [k for k in dir_contents if '.raw' in k and 'Image' in k]
    if len(raw_search) == 0:
        raise FileNotFoundError(f"No Image----.raw file found in {session_path}")
    
    # --------------------- 1. CONVERT RAW TO TIF --------------------- #
    img_found = len([j for j in dir_contents if 'img.tif' in j])
    led_artifact = 'y' if is_opto else 'n'
    
    if img_found == 0 or replace_img:
        print("\n[1/4] Converting raw data to img.tif...")
        code_start = time.process_time()
        
        thorfuns.RawToTif(filepath=session_path).convert(
            method='max_proj',
            chunker=1000,
            led_artifacts=led_artifact,
            wipe_and_replace=replace_img,
            preview_upsample = False,
            test_upsample=test_upsample
        )
        
        print(f"  Done in {(time.process_time() - code_start)/60:.2f} minutes")
    else:
        print("\n[1/4] img.tif already exists, skipping conversion...")
    
    # --------------------- 2. RUN SUITE2P ---------------------------- #
    s2p_found = len([j for j in dir_contents if 'suite2p' in j])
    
    if s2p_found == 0 or replace_s2p:
        print("\n[2/4] Running suite2p...")
        code_start = time.process_time()
        
        s2pfuns.fast_suite2p(
            imgpath=os.path.join(session_path, 'img.tif'),
            savepath='',
            gcamp='6f',
            alt_ops=None,  # uses default ops
            wipe_and_replace=replace_s2p
        )
        
        # Save summary images
        print("  Saving summary images...")
        _, _, _, _, ops, _, _ = s2pfuns.read_s2p(fpath=session_path)
        tf.imwrite(os.path.join(session_path, 'meanImg.tif'), ops['meanImg'], bigtiff=True)
        tf.imwrite(os.path.join(session_path, 'maxProj.tif'), ops['max_proj'], bigtiff=True)
        del ops
        
        print(f"  Done in {(time.process_time() - code_start)/60:.2f} minutes")
    else:
        print("\n[2/4] suite2p folder already exists, skipping...")
    
    # --------------------- 3. POSTPROCESS (DECONVOLUTION) ------------ #
    print("\n[3/4] Running postprocessing (deconvolution)...")
    code_start = time.process_time()
    
    s2pfuns.postProcess(
        s2ppath=os.path.join(session_path, 'suite2p', 'plane0')
    ).cleanup_raw_traces(n_jobs=-1, verbose=1)
    
    print(f"  Done in {(time.process_time() - code_start)/60:.2f} minutes")
    
    # --------------------- 4. CELLREG PREP --------------------------- #
    print("\n[4/4] Preparing CellReg file...")
    
    reg_file_name = os.path.split(session_path)[-1][0:20]
    if reg_file_name[-1] == '_':
        reg_file_name = reg_file_name + 'CellReg.mat'
    else:
        reg_file_name = reg_file_name + '_CellReg.mat'
    
    #sessreg.suite2pToCellReg(fnames=session_path, mask_overlap=True, save_name=reg_file_name)
    
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}\n")
