import tkinter as tk
from tkinter import filedialog, messagebox, IntVar, BooleanVar, Toplevel
from PIL import Image, ImageTk
import tifffile as tiff
import numpy as np
import os
from scipy.ndimage import median_filter, convolve  # Import for median and sharpening filters


class GIFCreatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GIF Creator")

        # Parameters
        self.gray_scale_image = BooleanVar(value=False)  # Default unchecked
        self.background_subtract = BooleanVar(value=False)  # Default unchecked
        self.apply_median_filter = BooleanVar(value=False)  # Default unchecked
        self.apply_sharpening = BooleanVar(value=False)  # Default unchecked
        self.threshold = tk.IntVar(value=100)
        self.threshold.trace_add("write", lambda *args: self.process_frames())  # Trace for dynamic updates
        self.frame_skip = tk.IntVar(value=100)  # Default subsampling parameter
        self.tiff_path = None
        self.original_frames = []  # To store original, unprocessed frames
        self.frames = []  # To store processed frames
        self.current_frame_index = 0

        # UI Components
        tk.Button(root, text="Load TIFF File", command=self.load_tiff).pack(pady=5)
        tk.Checkbutton(root, text="Grayscale Image", variable=self.gray_scale_image).pack()
        tk.Checkbutton(root, text="Background Subtraction", variable=self.background_subtract).pack()
        tk.Checkbutton(root, text="Apply Median Filter", variable=self.apply_median_filter, command=self.process_frames).pack()
        tk.Checkbutton(root, text="Apply Sharpening", variable=self.apply_sharpening, command=self.process_frames).pack()

        tk.Label(root, text="Threshold:").pack()
        tk.Entry(root, textvariable=self.threshold).pack()

        tk.Label(root, text="Frame Skip (Subsampling):").pack()
        tk.Entry(root, textvariable=self.frame_skip).pack()

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=5)

        navigation_frame = tk.Frame(root)
        navigation_frame.pack()
        tk.Button(navigation_frame, text="Previous", command=self.show_previous_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(navigation_frame, text="Next", command=self.show_next_frame).pack(side=tk.LEFT, padx=5)

        tk.Button(root, text="Preview GIF", command=self.preview_gif).pack(pady=5)
        tk.Button(root, text="Save GIF", command=self.save_gif).pack(pady=5)

        self.feedback_label = tk.Label(root, text="", fg="blue")
        self.feedback_label.pack(pady=5)

    def set_feedback(self, message):
        self.feedback_label.config(text=message)
        self.root.update_idletasks()

    def load_tiff(self):
        file_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select a TIFF File",
            filetypes=[("TIFF files", "*.tif")],
            initialdir=os.path.expanduser("~")
        )
        if not file_path:
            return

        self.set_feedback("Loading frames...")
        self.tiff_path = file_path
        with tiff.TiffFile(file_path) as tif:
            num_frames = len(tif.pages)  # Total frames in the TIFF
            self.set_feedback(f"Total frames in TIFF: {num_frames}")

            # Subsample frames based on the frame skip parameter
            self.original_frames = [
                page.asarray() for i, page in enumerate(tif.pages)
                if i % self.frame_skip.get() == 0
            ]

        self.frames = list(self.original_frames)  # Initialize processed frames as a copy of the original
        self.process_frames()
        self.current_frame_index = 0
        self.show_frame()
        self.set_feedback(f"Subsampled {len(self.frames)} frames successfully!")

    def process_frames(self):
        self.set_feedback("Processing frames...")
        self.frames = []  # Reset processed frames

        for frame in self.original_frames:  # Always start from original frames
            if self.apply_median_filter.get():
                frame = median_filter(frame, size=3)  # Median filter to reduce noise
                self.set_feedback("Median filter applied")

            if self.apply_sharpening.get():
                sharpening_kernel = np.array([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ])
                frame = convolve(frame, sharpening_kernel)  # Apply sharpening filter
                self.set_feedback("Sharpening filter applied")

            if self.gray_scale_image.get():
                frame = (255 * (frame - np.min(frame)) / np.ptp(frame)).astype(np.uint8)

            if self.background_subtract.get():
                threshold = self.threshold.get()
                background_mask = frame < threshold
                background_level = np.median(frame[background_mask])
                frame_cleaned = frame - background_level
                frame_cleaned[frame_cleaned < 0] = 0
                frame = frame_cleaned

            self.frames.append(frame)

        self.set_feedback("Frames processed successfully!")
        self.show_frame()  # Refresh the currently displayed frame

    def show_frame(self):
        if not self.frames:
            return

        frame = self.frames[self.current_frame_index]
        frame_image = Image.fromarray(frame)
        frame_image.thumbnail((500, 500))  # Resize for display
        photo = ImageTk.PhotoImage(frame_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def show_previous_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.show_frame()

    def show_next_frame(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.show_frame()

    def preview_gif(self):
        if not self.frames:
            messagebox.showerror("Error", "No frames loaded to preview.")
            return

        self.set_feedback("Creating GIF preview...")
        gif_window = Toplevel(self.root)
        gif_window.title("GIF Preview")
        gif_window.geometry("800x600")

        # Create GIF from frames
        images = [Image.fromarray(frame) for frame in self.frames]
        gif_frames = [ImageTk.PhotoImage(img) for img in images]

        gif_label = tk.Label(gif_window)
        gif_label.pack(expand=True)

        def animate(index=0):
            gif_label.config(image=gif_frames[index])
            index = (index + 1) % len(gif_frames)
            gif_window.after(133, animate, index)

        animate()
        self.set_feedback("GIF preview ready!")

    def save_gif(self):
        if not self.frames:
            messagebox.showerror("Error", "No frames loaded to save.")
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF files", "*.gif")])
        if not output_path:
            return

        self.set_feedback("Saving GIF...")
        pil_frames = [Image.fromarray(frame) for frame in self.frames]
        pil_frames[0].save(
            output_path, save_all=True, append_images=pil_frames[1:], loop=0, duration=133
        )
        self.set_feedback("GIF saved successfully!")
        messagebox.showinfo("Success", f"GIF saved successfully at {output_path}")


# Run the application
root = tk.Tk()
app = GIFCreatorGUI(root)
root.mainloop()