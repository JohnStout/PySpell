# GIF Creator
import tifffile as tiff
from PIL import Image
import numpy as np
import os

# set path
tiff_path = r"D:\L6 Experiments\L612\FOV1\SEDS_day3_LBC2_p70_optoRec_FOV1\SEDS_day3_LBC2_p70_optoRec_FOV1_img\img.tif"

# whether to grayscale the image
gray_scale_image = False
background_subtract = True

# Parameters for subsampling
frame_skip = 100  # Process every nth frame to reduce the number of frames in the GIF

# Open the TIFF file and load frames
print(f"Loading TIFF file: {tiff_path}")
with tiff.TiffFile(tiff_path) as tif:
    num_frames = len(tif.pages)  # Total number of frames
    print(f"Total frames in TIFF: {num_frames}")
    
    frames = []  # To store processed frames for the GIF
    for i, page in enumerate(tif.pages):
        if i % frame_skip == 0:  # Subsample frames by skipping
            # Read the current frame
            frame = page.asarray()

            # Normalize frame intensity to 0-255 for grayscale GIF
            if gray_scale_image == True:
                frame = (255 * (frame - np.min(frame)) / np.ptp(frame)).astype(np.uint8)

            # Clean the frame by subtracting the background
            if background_subtract == True:
                threshold = 100
                background_mask = frame < threshold  # Create a mask for background (e.g., pixels below a threshold)
                background_level = np.median(frame[background_mask])  # Compute the median background value
                frame_cleaned = frame - background_level  # Subtract the background
                frame_cleaned[frame_cleaned < 0] = 0  # Clip to avoid negative values
                frame = frame_cleaned

            # Convert frame to a PIL Image
            frame_img = Image.fromarray(frame)
            frames.append(frame_img)

    print(f"Processed {len(frames)} frames for the GIF (subsampling every {frame_skip} frames)")

# Save the frames as a GIF
root = os.path.dirname(tiff_path)
gif_output_path = os.path.join(root,'output.gif')
frames[0].save(
    gif_output_path, 
    save_all=True, 
    append_images=frames[1:], 
    loop=0,  # Infinite looping
    duration=133  # Duration of each frame in milliseconds (e.g., 7.5Hz ≈ 133ms per frame)
)

print(f"GIF saved successfully at {gif_output_path}")
