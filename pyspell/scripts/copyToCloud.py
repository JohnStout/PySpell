# copyToCloud.py
# Script to synchronize local data folders to our permanent cloud storage
#
# This code will ONLY copy files from local folders to the mapped cloud drive (Z:).
# This code will ONLY copy files if the local files are newer than the cloud files. This preserves your cloud
#       data and analyses.
# If you set DRY_RUN = True, the code will only print what it WOULD do, without actually copying any files.
#
# To use:
# 1) Ensure that your cloud drive is mapped to Z:
# 2) Edit the syncpaths list below to add any local/cloud folder pairs you want to synchronize
# 3) Hit run


# import standard modules
import os; import matplotlib.pyplot as plt; import tifffile as tf
path_added = os.path.split(os.getcwd())[0]; os.chdir(path_added); print("Added path:",path_added)
from pathlib import Path
import numpy as np
import time
import csv
from datetime import datetime, timedelta
from suite2p.io.save import save_mat
import shutil, ntpath

# custom modules
import rootfun as rf # we can import this if our cwd is local
root = rf.dropbox_root(dropbox_folder='timspellman')

# paths to synchronize
syncpaths = dict()
syncpaths = [

    # Folder represents where the data is stored locally, while Cloud is the mapped cloud drive location
    {'Folder': r"F:\John\L6 Experiments", 'Cloud': r"Z:\John\L6IMGDRIVE\L6 Experiments"},
    {'Folder': r"E:\L6 Experiments",      'Cloud': r"Z:\John\L6IMGDRIVE2\L6 Experiments"},
    {'Folder': r"H:\Layer6",              'Cloud': r"Z:\John\L6IMGDRIVE3\Layer6"},
        
        ]

# --- guard: ensure Cloud paths are on Z: ---
cloud_check = [sp for sp in syncpaths
               if ntpath.splitdrive(os.path.normpath(sp['Cloud']))[0].upper() != "Z:"]
if cloud_check:
    print("ERROR: One or more Cloud paths are not on Z: — fix before proceeding")
    for c in cloud_check:
        print(c)
    exit()

# --- config ---
TOLERANCE_SECONDS = 2.0   # allow small clock skew between local + mapped drive
DRY_RUN = False           # set True to see actions without copying

def scan_tree(base_dir: str) -> dict:
    """Map relative path -> {path, mtime, size} for all files under base_dir."""
    out = {}
    if not os.path.isdir(base_dir):
        return out
    for dirpath, _, filenames in os.walk(base_dir):
        for name in filenames:
            src = os.path.join(dirpath, name)
            try:
                st = os.stat(src)
            except FileNotFoundError:
                continue  # file changed mid-walk
            rel = os.path.relpath(src, base_dir)
            out[rel] = {"path": src, "mtime": st.st_mtime, "size": st.st_size}
    return out

def copy_from_origin_to_cloud(origin_root: str, cloud_root: str, dry_run: bool = False):
    """ONE-WAY sync: origin → cloud. Never copies cloud → origin."""
    origin = scan_tree(origin_root)
    cloud  = scan_tree(cloud_root) if os.path.isdir(cloud_root) else {}

    copied = skipped = newer_in_cloud = errors = 0

    for rel, o in origin.items():

        # determine if we need to copy
        dest = os.path.join(cloud_root, rel)
        c = cloud.get(rel)

        # ensure destination folder exists
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if c is None:
            # file missing in cloud → copy
            action = "new"
            do_copy = True
        else:

            # file exists in cloud → compare
            dt = o["mtime"] - c["mtime"]
            same_time = abs(dt) <= TOLERANCE_SECONDS
            same_size = (o["size"] == c["size"])
            if same_time and same_size:
                skipped += 1
                continue

            # copy if local is newer (beyond tolerance) OR timestamps ~same but size changed
            if dt > TOLERANCE_SECONDS or (same_time and not same_size):
                action = "newer/size-diff"
                do_copy = True
            else:
                # cloud appears newer; respect one-way rule and do nothing
                newer_in_cloud += 1
                print(f"Cloud newer, skipping: {rel}")
                continue

        if do_copy:
            if dry_run:
                print(f"DRY-RUN copy ({action}): {o['path']} -> {dest}")
                copied += 1
            else:
                try:
                    shutil.copy2(o["path"], dest)
                    copied += 1
                    print(f"Copied ({action}): {o['path']} -> {dest}")
                except Exception as e:
                    errors += 1
                    print(f"ERROR copying {o['path']} -> {dest}: {e}")

    print(f"Done. Copied={copied}, Skipped(up-to-date)={skipped}, "
          f"Cloud-newer(skipped)={newer_in_cloud}, Errors={errors}")

# --- run the sync for each mapping ---
for sp in syncpaths:
    print(f"\n→ Syncing\n  Origin: {sp['Folder']}\n  Cloud : {sp['Cloud']}")
    copy_from_origin_to_cloud(sp['Folder'], sp['Cloud'], dry_run=DRY_RUN)
