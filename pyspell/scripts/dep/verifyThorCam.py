import os, cv2, tifffile as tf, numpy as np
from natsort import natsorted
import builtins # used just to clear the exitingg next value
from tqdm import tqdm 

folder = r"G:\LAYER 6\A10-FLEX_GcaMP_CC\SD2_whisker_day1_optoRec_cam"
def load_cam_data(folder, cam="C2", normalise=False, roi=(800,500,1000,1000)):
 
    # ── locate files ───────────────────────────────────────────────────
    #tif_files = natsorted([i for i in os.listdir(folder) if 'C1' in i and 'stitched' not in i])
    tif_paths = natsorted([os.path.join(folder, f) for f in os.listdir(folder)
                           if f.endswith(".tif") and f"_{cam}_" in f
                           and "stitched" not in f])
    avi_path  = builtins.next((os.path.join(folder, f) for f in os.listdir(folder)
                      if f.endswith(".avi") and f"_{cam}_" in f
                      and "stitched" in f), None)

    if not tif_paths or avi_path is None:
        raise FileNotFoundError("Need at least one raw TIFF stack and the "
                                "stitched AVI for the selected camera tag.")
    print('loaded')
    xmin, ymin, xmax, ymax = roi
    # ── helper for TIFF normalisation (same as in stitcher) ────────────
    def _norm(frame):
        if not normalise or frame.dtype == np.uint8:
            return frame
        lo, hi = int(frame.min()), int(frame.max())
        frame  = np.zeros_like(frame, dtype=np.uint8) if hi <= lo \
                 else ((frame.astype(np.float32) - lo) * 255/(hi-lo)).astype(np.uint8)
        return frame

    # ── read AVI into memory ───────────────────────────────────────────
    aviData = []

    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {avi_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # total frame count
    pbar_avi = tqdm(total=n_frames, desc="[AVI]  frames", unit="fr")

    ok, fr = cap.read()
    while ok:
        fr = fr[ymin:ymax, xmin:xmax]    # crop to ROI
        aviData.append(fr)               # uint8 BGR
        pbar_avi.update(1)               # advance bar by 1
        ok, fr = cap.read()

    cap.release()
    pbar_avi.close()
    # ── read every TIFF frame ──────────────────────────────────────────
    tiffData = []
    pbar = tqdm(tif_paths, desc="[TIFF] stacks", unit="stack")
    for path in pbar:
        with tf.TiffFile(path) as tif:
            for pg in tif.pages:
                frame = pg.asarray()
                if frame.ndim == 2:                    # grey → BGR
                    frame = cv2.cvtColor(_norm(frame), cv2.COLOR_GRAY2BGR)
                else:
                    frame = _norm(frame)
                    if frame.shape[2] > 3:             # drop alpha/channel 4
                        frame = frame[..., :3]
                frame = frame[ymin:ymax, xmin:xmax]
                tiffData.append(frame)
    pbar.close()
    print('readTiff')
    # quick sanity check
    if len(aviData) != len(tiffData):
        print(f"⚠ mismatch: AVI has {len(aviData)} frames, "
              f"TIFFs have {len(tiffData)}")

    return aviData, tiffData
