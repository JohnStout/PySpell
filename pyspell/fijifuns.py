# filepath: pyspell/pyspell/fijifuns.py
# FIJIfuns
#
# Code that allows user to use fiji ROI's
#
# The user must load an image into fiji, then split the image into separate images
# Next, the user must draw ROI's on the images, separately, then save out the .zip file
# using that .zip file, we can read the ROIs and compare them
# 1) install read-roi:      pip install read-roi
#
#
#
# TODO: This code will be wrapped into a GUI that allows dynamic visualization of parameter setting from the user.
# For example, maybe the user wants to adjust the smoothing_sigma or the threshold value. 
# This is necessary because different flourophores have different results in terms of relative brightness to the background image.
# So the user is required to ensure the ROIs are set correctly.

from PyQt5.QtWidgets import QMessageBox
from read_roi import read_roi_zip
from skimage.draw import polygon, ellipse
from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import tifffile as tf

from scipy.ndimage    import gaussian_filter
from skimage.morphology import disk, white_tophat
from matplotlib.patches import Circle

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature   import peak_local_max
from skimage.filters   import threshold_otsu
from skimage.morphology import disk
from skimage.draw      import disk as draw_disk

import sys
import os
import numpy as np
from skimage import io
from skimage.transform import rotate as sk_rotate

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton,
    QVBoxLayout, QWidget, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, Slider

def rasterize_rois(rois, shape):
    h, w = shape
    masks = {}
    for name, meta in rois.items():
        t = meta['type']
        if t == 'oval':
            cy = meta['top'] + meta['height'] / 2
            cx = meta['left'] + meta['width'] / 2
            ry = meta['height'] / 2
            rx = meta['width'] / 2
            rr, cc = ellipse(cy, cx, ry, rx, shape=(h, w))
        elif t == 'rectangle':
            left, top = meta['left'], meta['top']
            hgt, wdt = meta['height'], meta['width']
            xs = [left, left + wdt, left + wdt, left]
            ys = [top, top, top + hgt, top + hgt]
            rr, cc = polygon(ys, xs, shape=(h, w))
        else:
            raise RuntimeError(f"Unsupported ROI type: {t}")
        mask = np.zeros((h, w), bool)
        mask[rr, cc] = True
        masks[name] = mask
    return masks

def compare_rois(red_zip, green_zip, img_shape):
    rois_red = read_roi_zip(red_zip)
    rois_green = read_roi_zip(green_zip)
    masks_red = rasterize_rois(rois_red, img_shape)
    masks_green = rasterize_rois(rois_green, img_shape)

    shared_pairs = []
    reds_with_overlap = set()
    greens_with_overlap = set()

    for r_name, r_mask in masks_red.items():
        for g_name, g_mask in masks_green.items():
            ov = np.logical_and(r_mask, g_mask).sum()
            if ov > 0:
                pct_r = 100 * ov / r_mask.sum()
                pct_g = 100 * ov / g_mask.sum()
                shared_pairs.append((r_name, g_name, ov, pct_r, pct_g))
                reds_with_overlap.add(r_name)
                greens_with_overlap.add(g_name)
                print(f"RED '{r_name}' ↔ GREEN '{g_name}': "
                      f"{ov} px overlap "
                      f"({pct_r:.1f}% of red, {pct_g:.1f}% of green)")

    shared_red = sorted(reds_with_overlap)
    shared_green = sorted(greens_with_overlap)
    separate_red = [r for r in masks_red if r not in reds_with_overlap]
    separate_green = [g for g in masks_green if g not in greens_with_overlap]

    n_red = len(masks_red)
    n_green = len(masks_green)
    total_rois = n_red + n_green
    n_shared = len(reds_with_overlap) + len(greens_with_overlap)
    pct_shared_total = 100 * n_shared / total_rois

    print("\nSUMMARY:")
    print(f"  Total ROIs (red + green): {total_rois}")
    print(f"  Shared ROIs: {n_shared} → {pct_shared_total:.1f}% of all ROIs")
    print(f"    • red shared:   {len(shared_red)} of {n_red} ({100 * len(shared_red) / n_red:.1f}%)")
    print(f"    • green shared: {len(shared_green)} of {n_green} ({100 * len(shared_green) / n_green:.1f}%)")
    print(f"  Separate red ROIs:   {len(separate_red)} → {separate_red}")
    print(f"  Separate green ROIs: {len(separate_green)} → {separate_green}\n")

    return (
        shared_pairs,
        shared_red,
        shared_green,
        separate_red,
        separate_green,
        pct_shared_total,
        masks_red,
        masks_green
    )

def plot_roi_overlay(image, masks_red, masks_green, alpha=0.4):
    h, w = image.shape[:2]
    red_union = np.zeros((h, w), bool)
    green_union = np.zeros((h, w), bool)
    for m in masks_red.values(): red_union |= m
    for m in masks_green.values(): green_union |= m

    overlap = red_union & green_union
    red_only = red_union & ~overlap
    green_only = green_union & ~overlap

    overlay = np.zeros((h, w, 3), float)
    overlay[red_only, 0] = 1.0
    overlay[green_only, 1] = 1.0
    overlay[overlap] = [1.0, 1.0, 0.0]

    plt.figure(figsize=(8, 8))
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.imshow(overlay, alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_image():
    Tk().withdraw()  # Prevents the root window from appearing
    filename = askopenfilename(filetypes=[("TIFF files", "*.tif;*.tiff")])
    if filename:
        return io.imread(filename)
    else:
        print("No file selected.")
        return None

def crop_image(image):
    plt.imshow(image)
    plt.title("Select an area of interest and press Enter")
    plt.axis('on')
    plt.show()

    # Get the coordinates of the selected area
    coords = plt.ginput(n=2)  # Get two points
    plt.close()

    # Convert to integer pixel indices
    x1, y1 = map(int, coords[0])
    x2, y2 = map(int, coords[1])

    # Crop the image
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img
'''
class CropperWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIFF Cropper")
        self.img = None
        self.crop_coords = None
        self.file_path = None  # <-- Add this line

        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()
        self.rect_selector = None

        btn_load = QPushButton("Load TIFF")
        btn_load.clicked.connect(self.load_image)
        btn_crop = QPushButton("Crop and Close")
        btn_crop.clicked.connect(self.finish_crop)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(btn_load)
        layout.addWidget(btn_crop)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open TIFF", "", "TIFF files (*.tif *.tiff)")
        if file_path:
            self.img = io.imread(file_path)
            self.file_path = file_path  # <-- Store the file path
            self.ax.clear()
            self.ax.imshow(self.img)
            self.canvas.draw()
            if self.rect_selector:
                self.rect_selector.disconnect_events()
            self.rect_selector = RectangleSelector(
                self.ax, self.on_select, useblit=True, button=[1], interactive=True, spancoords='pixels'
            )

    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.crop_coords = (min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2))

    def finish_crop(self):
        if self.img is not None and self.crop_coords is not None:
            self.close()

    def get_file_path(self):
        return self.file_path
'''

"""fijifuns_crop_rotate.py  •  v7 (complete)
---------------------------------------------------------------------
Interactive TIFF **rotate → crop** GUI using PyQt5 + Matplotlib.

Fixed:
• Completed `crop_tiff_gui()` implementation (load‑existing path and new‑crop
  branch both work and return values).
• Added `__main__` smoke‑test so the module can be run directly.
---------------------------------------------------------------------
"""

import sys, os
import numpy as np
from skimage import io
from skimage.transform import rotate as sk_rotate

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton,
    QVBoxLayout, QWidget, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, Slider

# ------------------------------------------------------------------
# GUI class
# ------------------------------------------------------------------
"""fijifuns_crop_rotate.py — v8 (indentation + selector fix)
----------------------------------------------------------------------------
Interactive TIFF **rotate → crop** GUI using PyQt5 + Matplotlib.
This version compiles on Matplotlib ≥3.1 without deprecated kwargs
and with correct indentation.
----------------------------------------------------------------------------
"""

import sys
import os
from typing import Optional, Tuple
import numpy as np
from skimage import io
from skimage.transform import rotate as sk_rotate

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QLabel,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, Slider

class CropperWindow(QMainWindow):
    """Main window allowing arbitrary rotation followed by cropping."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TIFF Cropper: rotate → crop")
        self.resize(850, 720)

        # ---------------- state ----------------
        self.orig_img: Optional[np.ndarray] = None
        self.img_rot: Optional[np.ndarray] = None
        self.angle: float = 0.0  # degrees CCW
        self.crop_coords: Optional[Tuple[int, int, int, int]] = None
        self.file_path: Optional[str] = None
        self.rotation_ui: bool = False  # mode flag

        # ---------------- canvas --------------
        self.canvas = FigureCanvas(Figure(constrained_layout=True))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_axis_off()

        # slider (lazy)
        self.slider_ax = None
        self.angle_slider: Optional[Slider] = None

        # rectangle selector
        self.rect_selector: Optional[RectangleSelector] = None

        # ---------------- widgets -------------
        btn_load = QPushButton("Load TIFF", clicked=self.load_image)
        self.btn_rot = QPushButton("Rotate Image", clicked=self.toggle_rotate_mode)
        btn_crop = QPushButton("Crop and Close", clicked=self.finish_crop)
        self.label_status = QLabel("<i>Load an image to begin…</i>")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(btn_load)
        layout.addWidget(self.btn_rot)
        layout.addWidget(btn_crop)
        layout.addWidget(self.label_status)
        container = QWidget(); container.setLayout(layout)
        self.setCentralWidget(container)

    # ------------------------------------------------------------
    # Image I/O
    # ------------------------------------------------------------
    def load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open TIFF", "", "TIFF files (*.tif *.tiff)")
        if not path:
            return
        self.file_path = path
        self.orig_img = io.imread(path)
        self.angle = 0.0
        self._ensure_slider()
        self.slider_ax.set_visible(False)
        self.rotation_ui = False
        self.btn_rot.setText("Rotate Image")
        self._update_view()
        self._set_status("Crop Mode: drag yellow rectangle to crop.")

    # ------------------------------------------------------------
    # Rotate Mode toggle
    # ------------------------------------------------------------
    def toggle_rotate_mode(self) -> None:
        if self.orig_img is None:
            return
        self.rotation_ui = not self.rotation_ui
        if self.rotation_ui:
            self.btn_rot.setText("Done Rotating")
            self.slider_ax.set_visible(True)
            if self.rect_selector:
                self.rect_selector.set_active(False)
            self._set_status("Rotate Mode: adjust slider, then click Done Rotating.")
        else:
            self.btn_rot.setText("Rotate Image")
            self.slider_ax.set_visible(False)
            self._reset_selector(force_new=True)
            self._set_status("Crop Mode: drag yellow rectangle to crop.")
        self.canvas.draw_idle()

    # ------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------
    def _ensure_slider(self) -> None:
        if self.angle_slider is not None:
            return
        self.slider_ax = self.canvas.figure.add_axes([0.15, 0.05, 0.7, 0.03])
        self.angle_slider = Slider(
            self.slider_ax,
            "Angle (°)",
            -180,
            180,
            valinit=0,
            valstep=0.5,
            color="#1f77b4",
        )
        self.angle_slider.on_changed(self._on_slider)

    def _on_slider(self, val):
        if not self.rotation_ui:
            return
        self.angle = float(val)
        self.crop_coords = None
        self._update_view()

    # ------------------------------------------------------------
    # View refresh helpers
    # ------------------------------------------------------------
    def _update_view(self) -> None:
        if self.orig_img is None:
            return
        self.img_rot = sk_rotate(
            self.orig_img, self.angle, resize=False, order=1, mode="edge", preserve_range=True
        )
        self.ax.clear(); self.ax.set_axis_off()
        self.ax.imshow(self.img_rot.astype(self.orig_img.dtype))
        if not self.rotation_ui:
            self._reset_selector(force_new=False)
        self.canvas.draw_idle()

    def _reset_selector(self, *, force_new: bool) -> None:
        if self.rotation_ui:
            return
        if self.rect_selector and not force_new:
            self.rect_selector.set_active(True)
            return
        if self.rect_selector:
            self.rect_selector.disconnect_events()
            self.rect_selector = None
        # Use default styling to stay version‑agnostic
        self.rect_selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

    # ------------------------------------------------------------
    # Selector callback
    # ------------------------------------------------------------
    def _on_select(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
            return  # ignore tiny drags
        self.crop_coords = (
            min(y1, y2),
            max(y1, y2),
            min(x1, x2),
            max(x1, x2),
        )
        self._set_status("Crop set. Press 'Crop and Close' or draw again.")

    # ------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------
    def finish_crop(self) -> None:
        if self.orig_img is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        if self.crop_coords is None:
            QMessageBox.warning(self, "No crop", "Draw a crop rectangle first.")
            return
        y1, y2, x1, x2 = self.crop_coords
        img_final = sk_rotate(
            self.orig_img, self.angle, resize=False, order=3, mode="edge", preserve_range=True
        )
        self.cropped_img = img_final[y1:y2, x1:x2].astype(self.orig_img.dtype)
        self.close()

    # ------------------------------------------------------------
    def get_results(self):
        return getattr(self, "cropped_img", None), self.file_path

    def _set_status(self, text: str) -> None:
        self.label_status.setText(f"<b>{text}</b>")

# ------------------------------------------------------------------
# Public helper
# ------------------------------------------------------------------

def crop_tiff_gui():
    """Launch the rotate‑crop GUI and return (cropped_img, original_path)."""
    app = QApplication.instance() or QApplication(sys.argv)

    dlg = QMessageBox(icon=QMessageBox.Question, windowTitle="Crop or Load")
    dlg.setText("Do you want to crop a new image or load an already cropped image?")
    crop_b = dlg.addButton("Crop new image", QMessageBox.AcceptRole)
    load_b = dlg.addButton("Load already cropped", QMessageBox.AcceptRole)
    cancel_b = dlg.addButton(QMessageBox.Cancel)
    dlg.exec_()

    if dlg.clickedButton() == cancel_b:
        return None, None

    if dlg.clickedButton() == load_b:
        path, _ = QFileDialog.getOpenFileName(
            None, "Open Cropped TIFF", "", "TIFF files (*.tif *.tiff)"
        )
        if not path:
            return None, None
        return io.imread(path), path

    win = CropperWindow()
    win.show()
    app.exec_()
    return win.get_results()

'''
def crop_tiff_gui():
    app = QApplication.instance() or QApplication(sys.argv)

    # Dialog to choose crop or load
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("Crop or Load")
    msg.setText("Do you want to crop a new image or load an already cropped image?")
    crop_btn = msg.addButton("Crop new image", QMessageBox.AcceptRole)
    load_btn = msg.addButton("Load already cropped", QMessageBox.AcceptRole)
    cancel_btn = msg.addButton(QMessageBox.Cancel)
    msg.exec_()

    clicked = msg.clickedButton()
    if clicked == cancel_btn:
        print("Operation cancelled.")
        return None, None

    if clicked == load_btn:
        file_path, _ = QFileDialog.getOpenFileName(None, "Open Cropped TIFF", "", "TIFF files (*.tif *.tiff)")
        if file_path:
            img = io.imread(file_path)
            plt.imshow(img)
            plt.title("Loaded Cropped Image")
            plt.axis('off')
            plt.show()
            return img, file_path
        else:
            print("No file selected.")
            return None, None

    # Otherwise, proceed with cropping
    win = CropperWindow()
    win.show()
    app.exec_()
    if win.img is not None and win.crop_coords is not None:
        y1, y2, x1, x2 = win.crop_coords
        cropped_img = win.img[y1:y2, x1:x2]
        plt.imshow(cropped_img)
        plt.title("Cropped Image")
        plt.axis('off')
        plt.show()

        # save the cropped image
        file_path = win.get_file_path()
        #if file_path:
            #new_path = os.path.splitext(file_path)[0] + "_cropped.tif"
            #tf.imsave(new_path, cropped_img)
            #print(f"Cropped image saved to: {new_path}")

        # Return both the cropped image and the file path
        return cropped_img, file_path
    else:
        print("No crop performed.")
        return None, None
'''

def crop_img_with_gui(img):
    """
    Display a GUI to crop the given image and return the cropped image and crop coordinates.
    No file dialog is used; the image is passed directly.

    Parameters
    ----------
    img : np.ndarray
        The image to crop.

    Returns
    -------
    cropped_img : np.ndarray
        The cropped image.
    crop_coords : tuple
        (y1, y2, x1, x2) crop coordinates.
    """
    class SimpleCropper(QMainWindow):
        def __init__(self, img):
            super().__init__()
            self.setWindowTitle("Crop Image")
            self.img = img
            self.crop_coords = None

            self.canvas = FigureCanvas(Figure())
            self.ax = self.canvas.figure.subplots()
            self.rect_selector = None

            self.ax.imshow(self.img)
            self.canvas.draw()
            #self.rect_selector = RectangleSelector(
            #    self.ax, self.on_select, useblit=True, button=[1], interactive=True, spancoords='pixels'
            #)
            self.rect_selector = RectangleSelector(
                self.ax, self._on_select, useblit=True,
                button=[1], minspanx=5, minspany=5,
                spancoords="pixels", interactive=True
            )
            btn_crop = QPushButton("Crop and Close")
            btn_crop.clicked.connect(self.finish_crop)

            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            layout.addWidget(btn_crop)
            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)

        def on_select(self, eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.crop_coords = (min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2))

        def finish_crop(self):
            if self.img is not None and self.crop_coords is not None:
                self.close()

    app = QApplication.instance() or QApplication(sys.argv)
    win = SimpleCropper(img)
    win.show()
    app.exec_()

    if win.img is not None and win.crop_coords is not None:
        y1, y2, x1, x2 = win.crop_coords
        cropped_img = win.img[y1:y2, x1:x2]
        plt.imshow(cropped_img)
        plt.title("Cropped Image")
        plt.axis('off')
        plt.show()
        return cropped_img, win.crop_coords
    else:
        print("No crop performed.")
        return
      
def calibrate_cell_size(image):
    """
    1) Use the toolbar to zoom/pan however you like.
    2) Press 'm' to enter measurement mode.
    3) Click TWO points; a line will be drawn and its length returned.
       Press ESC at any time to cancel.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.set_title("Zoom/pan with toolbar.  Press 'm' to measure, ESC to cancel.")
    
    state = {'measuring': False, 'points': []}
    
    def onclick(evt):
        if not state['measuring'] or evt.inaxes != ax or evt.button != 1:
            return
        state['points'].append((evt.xdata, evt.ydata))
        ax.plot(evt.xdata, evt.ydata, 'ro')
        fig.canvas.draw()
        if len(state['points']) == 2:
            (x1, y1), (x2, y2) = state['points']
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
            fig.canvas.draw()
            plt.close(fig)
    
    def onkey(evt):
        if evt.key == 'escape':
            state['points'].clear()
            plt.close(fig)
        elif evt.key.lower() == 'm':
            state['measuring'] = True
            ax.set_title("MEASURE: click TWO points across the cell (ESC to cancel)")
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event',    onkey)
    
    # <— this forces the window to block until it’s closed by your callbacks
    plt.show(block=True)
    
    if len(state['points']) == 2:
        (x1, y1), (x2, y2) = state['points']
        diameter = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        print(f"Calibrated cell diameter: {diameter:.1f} pixels")
        return diameter
    else:
        print("Calibration cancelled.")
        return None

def auto_detect_circular_rois(
    img_bs: np.ndarray,
    cell_diameter,
    min_distance=None,
    threshold_abs=None,
    min_fraction: float = 0.5,
    threshold_factor: float = 1.25,
    smoothing_sigma: float = None
):
    """
    1) Smooth + background‑subtract
    2) Compute Otsu threshold → base_thresh
    3) Raise to detection_thresh = base_thresh * threshold_factor
    4) Find local maxima ≥ detection_thresh with spacing=min_distance
    5) Keep only those circles where >= min_fraction of pixels inside
       are ≥ base_thresh.
    Returns list of (y, x, radius).
    """
    # — sanity
    if img_bs.ndim != 2:
        raise ValueError("`channel_img` must be 2D")
    if cell_diameter <= 0:
        raise ValueError("`cell_diameter` > 0")
    
    # — defaults
    radius = max(1, int(round(cell_diameter / 2)))
    if min_distance is None:
        min_distance = radius
    if smoothing_sigma is None:
        # a quarter‑cell for smoothing
        smoothing_sigma = max(1, cell_diameter / 4)
    
    # — 1) smooth & background‑subtract
    #img_smooth = gaussian_filter(channel_img, sigma=smoothing_sigma)
    #bg = np.median(img_smooth)
    #img_bs = img_smooth - bg
    #img_bs[img_bs < 0] = 0
    
    # — 2) base threshold & raised detection threshold
    base_thresh = threshold_otsu(img_bs)
    det_thresh  = (threshold_abs or base_thresh) * threshold_factor
    
    # — 3) local maxima detection
    footprint = disk(min_distance)
    coords = peak_local_max(
        img_bs,
        footprint=footprint,
        threshold_abs=det_thresh,
        exclude_border=False
    )
    
    # — 4) fraction‑inside filter
    kept = []
    for y, x in coords:
        rr, cc = draw_disk((int(y), int(x)), radius, shape=img_bs.shape)
        patch = img_bs[rr, cc]
        frac = np.mean(patch >= base_thresh)
        if frac >= min_fraction:
            kept.append((int(y), int(x), radius))
    
    return kept

def subtract_background(
    image: np.ndarray,
    tophat_radius: int,
    smoothing_sigma: float = None,
    show: bool = False
) -> np.ndarray:
    """
    Remove low‐frequency background from `image` via:
      1) Gaussian smoothing (σ = smoothing_sigma or tophat_radius/4)
      2) Morphological white‐tophat with a disk of radius `tophat_radius`
      3) Median subtraction + clipping to [0, ∞)

    Parameters
    ----------
    image : 2D array
        Input image.
    tophat_radius : int
        Radius of the disk structuring element for white_tophat.
    smoothing_sigma : float, optional
        Gaussian blur sigma. If None, defaults to tophat_radius/4.
    show : bool, default=False
        If True, displays side‐by‐side original vs background‐subtracted.

    Returns
    -------
    img_bs : 2D array
        The background‐subtracted image.
    """
    if image.ndim != 2:
        raise ValueError("subtract_background: `image` must be 2D")
    if smoothing_sigma is None:
        smoothing_sigma = max(1.0, tophat_radius / 4)

    # 1) smooth to kill speckle
    img_smooth = gaussian_filter(image, sigma=smoothing_sigma)

    # 2) white‐tophat to remove broad background
    selem    = disk(tophat_radius)
    img_toph = white_tophat(img_smooth, selem)

    # 3) subtract median and zero‐clip
    bg     = np.median(img_toph)
    img_bs = img_toph - bg
    img_bs[img_bs < 0] = 0

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(img_bs, cmap='gray')
        axes[1].set_title("Background‐subtracted")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return img_bs

def filter_overlapping_rois(rois, image, overlap_thresh=0.75):
    """
    rois: list of (cy, cx, radius)
    image: 2D array, used to score each ROI by intensity at its center
    overlap_thresh: fraction of area overlap above which to suppress weaker ROI
    """
    # score each ROI by image intensity at its center
    scores = [image[int(cy), int(cx)] for cy, cx, _ in rois]
    # sort indices by descending score
    idxs = sorted(range(len(rois)), key=lambda i: scores[i], reverse=True)

    keep = []
    for i in idxs:
        cy_i, cx_i, r_i = rois[i]
        too_much_overlap = False

        for j in keep:
            cy_j, cx_j, r_j = rois[j]
            d = np.hypot(cx_i - cx_j, cy_i - cy_j)

            # no overlap if centers are too far apart
            if d >= (r_i + r_j):
                continue

            # compute intersection area for two circles of equal radius r_i
            r = r_i
            # lens‐area formula for intersection of two identical circles
            inter = 2 * r**2 * np.arccos(d / (2*r)) \
                  - 0.5 * d * np.sqrt(max(0, 4*r**2 - d**2))

            frac = inter / (np.pi * r**2)
            if frac > overlap_thresh:
                too_much_overlap = True
                break

        if not too_much_overlap:
            keep.append(i)

    return [rois[i] for i in keep]

def plot_circular_rois(image, rois):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray' if image.ndim==2 else None)
    for cy, cx, r in rois:
        circ = plt.Circle((cx, cy), r, edgecolor='yellow', facecolor='none', linewidth=2)
        ax.add_patch(circ)
    plt.title("Auto-detected Circular ROIs")
    plt.show()

from math import hypot

def find_shared_rois(rois_red, rois_green):
    """
    Given:
      rois_red   = [(y, x, r), …]
      rois_green = [(y, x, r), …]
    Returns:
      shared_red_indices,  # indices i in rois_red whose circle overlaps any in rois_green
      shared_green_indices # indices j in rois_green whose circle overlaps any in rois_red
    """

    shared_red   = set()
    shared_green = set()
    for i, (y1, x1, r1) in enumerate(rois_red):
        for j, (y2, x2, r2) in enumerate(rois_green):
            # center‐to‐center distance
            d = hypot(x1 - x2, y1 - y2)
            # if circles overlap at all
            if d < (r1 + r2):
                shared_red.add(i)
                shared_green.add(j)
    return sorted(shared_red), sorted(shared_green)

def plot_shared_rois(cropped_red, cropped_green, cropped_blue,
                     rois_red, rois_green,
                     shared_red_idx, shared_green_idx):
    """
    Reconstruct an RGB image from three single‑channel arrays and
    overlay the shared ROIs between red and green channels.

    Parameters
    ----------
    cropped_red : 2D array
        Red channel image.
    cropped_green : 2D array
        Green channel image.
    cropped_blue : 2D array
        Blue channel image.
    rois_red : list of (y, x, r) tuples
        Detected ROIs in the red channel.
    rois_green : list of (y, x, r) tuples
        Detected ROIs in the green channel.
    shared_red_idx : list of int
        Indices into `rois_red` that overlap any green ROI.
    shared_green_idx : list of int
        Indices into `rois_green` that overlap any red ROI.

    Returns
    -------
    None
        Displays a figure with:
          - the reconstructed RGB image
          - yellow circles for each shared red ROI
          - cyan circles for each shared green ROI
    """
    # Stack channels into an RGB image and normalize
    rgb = np.stack([cropped_red, cropped_green, cropped_blue], axis=-1).astype(float)
    rgb /= np.max(rgb)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis('off')

    # Plot shared red ROIs in yellow
    for i in shared_red_idx:
        y, x, r = rois_red[i]
        circ = Circle((x, y), r, edgecolor='yellow', facecolor='none', linewidth=2)
        ax.add_patch(circ)

    # Plot shared green ROIs in cyan
    for j in shared_green_idx:
        y, x, r = rois_green[j]
        circ = Circle((x, y), r, edgecolor='cyan', facecolor='none', linewidth=2)
        ax.add_patch(circ)

    ax.set_title("Shared ROIs: Yellow (red→green), Cyan (green→red)")
    plt.show()

from skimage.feature import blob_log
import matplotlib.pyplot as plt

def simple_blob_analysis(image, min_sigma=1, max_sigma=10, num_sigma=5, threshold=10, threshold_factor=1.0, cmap='gray', smooth_img = False):
    
    if smooth_img == True:
        image = gaussian_filter(image, sigma=1.5)

    # Perform simple blob detection using Laplacian of Gaussian (LoG)
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    
    # Determine the base threshold and detection threshold
    #threshold_factor = 1.25  # Default factor to adjust the threshold
    base_thresh = threshold_otsu(image)
    det_thresh  = base_thresh * threshold_factor

    # Filter blobs based on the detection threshold
    # For each blob, we check if the maximum intensity is above the detection threshold
    blobs = [(y, x, sigma) for y, x, sigma in blobs if image[int(y), int(x)] >= det_thresh]
    
    # blob_log returns (y, x, sigma) for each blob
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap)
    for y, x, sigma in blobs:
        c = plt.Circle((x, y), sigma * 1.5, color='black', linewidth=2, fill=False)
        ax.add_patch(c)
    plt.title("Simple Blob Detection")
    plt.show()
    return blobs

# ensure float in [0,1]
def rescale_image(img):

    # convert to HSV
    hsv = rgb2hsv(img)

    # stretch only the V channel between its 2nd and 98th percentiles:
    p2, p98 = np.percentile(hsv[..., 2], (2, 98))
    hsv[..., 2] = exposure.rescale_intensity(hsv[..., 2],
                                            in_range=(p2, p98),
                                            out_range=(0, 1))

    # (optional) gamma‐correct to taste:
    gamma = 0.5
    hsv[..., 2] = hsv[..., 2] ** gamma

    # back to RGB
    adjusted = hsv2rgb(hsv)

    return adjusted

import numpy as np
from scipy.ndimage import uniform_filter, shift
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# Example usage:
if __name__ == "__main__":
          
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import exposure, img_as_float
    from skimage.color import rgb2hsv, hsv2rgb
    import os

    # crop the image using the GUI
    cropped, file_path = crop_tiff_gui()

    # check to save if a cropped image is already saved
    file_save = file_path.replace(".tif", "_cropped.tif")
    if os.path.exists(file_save):
        # add a digit
        i = 1
        while os.path.exists(file_save.replace("_cropped.tif", f"_cropped_{i}.tif")):
            i += 1
        file_path = file_path.replace(".tif", f"_cropped_{i}.tif")
    else:
        file_path = file_path.replace(".tif", "_cropped.tif")
    tf.imsave(file_path, cropped)

    # adjust
    adjusted = rescale_image(cropped)

    # gen a save path for figures
    fig_save_path = os.path.split(file_path)[0]

    # save the cropped image
    fig1, axs1 = plt.subplots(figsize=(8, 8))
    axs1.imshow(adjusted)
    axs1.axis('off')
    fig1.savefig(os.path.join(fig_save_path,'CroppedOverlay_HSVstretch.eps'),
                format='eps',
                dpi=1000, # controls raster‐embedded image resolution
                bbox_inches='tight') 
    
    # plot OG
    og_img = tf.imread(file_path.replace("_cropped.tif", ".tif"))
    og_img = rescale_image(og_img)
    #og_img_crop = crop_img_with_gui(og_img)
    #fig2, axs2 = plt.subplots(figsize=(8, 8))
    #axs2.imshow(og_img)
    #axs2.axis('off')
    #fig2.savefig(os.path.join(fig_save_path,'OGimg_HSVstretch.eps'),
    #            format='eps',
    #            dpi=1000, # controls raster‐embedded image resolution
    #            bbox_inches='tight')     


    # Separate channels
    cropped_red = cropped[:,:,0]
    cropped_green = cropped[:,:,1]
    cropped_blue = cropped[:,:,2]
    new_path_red = os.path.splitext(file_path)[0] + "_cropped_red.tif"
    new_path_green = os.path.splitext(file_path)[0] + "_cropped_green.tif"
    new_path_blue = os.path.splitext(file_path)[0] + "_cropped_blue.tif"
    tf.imsave(new_path_red, cropped_red)
    tf.imsave(new_path_green, cropped_green)
    tf.imsave(new_path_blue, cropped_blue)
    print(f"Cropped image saved to: {os.path.splitext(file_path)[0]}")

    # show subplots per channel
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    axs2[0].imshow(cropped_red, cmap='Reds')
    axs2[0].set_title("Red Channel")
    axs2[0].axis('off')
    axs2[1].imshow(cropped_green, cmap='Greens')
    axs2[1].set_title("Green Channel")
    axs2[1].axis('off')
    axs2[2].imshow(cropped_blue, cmap='Blues')
    axs2[2].set_title("Blue Channel")
    axs2[2].axis('off')
    plt.tight_layout()
    plt.show()   
    fig2.savefig(os.path.join(fig_save_path,'three_channels.eps'),
                format='eps',
                dpi=1000,           # controls raster‐embedded image resolution
                bbox_inches='tight')    

    # check for pixels with perfect correlation
    if cropped_red.shape != cropped_green.shape or cropped_red.shape != cropped_blue.shape:
        raise ValueError("Cropped images must have the same shape for all channels.")

    # remove background - *2 allows for larger cells
    background_subtract = True
    if background_subtract:
        print("Subtracting background from red and green channels...")
        cropped_red_bs   = subtract_background(cropped_red, tophat_radius=5, show=True, smoothing_sigma=1.5)
        cropped_green_bs = subtract_background(cropped_green, tophat_radius=5, show=True, smoothing_sigma=1.5)
        cropped_blue_bs  = subtract_background(cropped_blue, tophat_radius=5, show=True, smoothing_sigma=1.5)    
    else:
        cropped_red_bs   = cropped_red.copy()
        cropped_green_bs = cropped_green.copy()
        cropped_blue_bs  = cropped_blue.copy()

    # return cell diameter
    cell_diam = calibrate_cell_size(cropped_green_bs)
    cell_diam = 5

    # save out the background subtracted images
    print("Saving background-subtracted images...")
    new_path_red_bs = os.path.splitext(file_path)[0] + "_cropped_red_bs.tif"
    new_path_green_bs = os.path.splitext(file_path)[0] + "_cropped_green_bs.tif"
    new_path_blue_bs = os.path.splitext(file_path)[0] + "_cropped_blue_bs.tif"    
    tf.imsave(new_path_red_bs, cropped_red_bs)
    tf.imsave(new_path_green_bs, cropped_green_bs)
    tf.imsave(new_path_blue_bs, cropped_blue_bs)    

    # blob analysis
    
    # Assume cropped_red, cropped_green, cropped_blue already defined
    #rois_blue  = auto_detect_circular_rois(cropped_blue, cell_diameter=cell_diam, threshold_factor=0, smoothing_sigma=2.0, min_fraction=0.1)
    #rois_red   = auto_detect_circular_rois(cropped_red_bs, cell_diameter=cell_diam, threshold_factor=0, smoothing_sigma=2.0, min_fraction=0.1)
    #rois_green = auto_detect_circular_rois(cropped_green_bs, cell_diameter=cell_diam, threshold_factor=0, smoothing_sigma=2.0, min_fraction=0.1)

    # filter out ROIs with too much overlap
    #rois_red = filter_overlapping_rois(rois_red, cropped_red_bs, overlap_thresh=0.1)
    #rois_green = filter_overlapping_rois(rois_green, cropped_green_bs, overlap_thresh=0.1)

    # 
    # beads9
    #blobs_red   = simple_blob_analysis(cropped_red_bs, min_sigma=2, max_sigma=9, num_sigma=5, threshold=10, threshold_factor=1, cmap='Reds')
    #blobs_green = simple_blob_analysis(cropped_green_bs, min_sigma=2, max_sigma=9, num_sigma=5, threshold=10, threshold_factor=1, cmap='Greens')

    from scipy.stats import zscore
    auto_thresh = ((np.ceil(cell_diam)/2)**2)*3.14
    blobs_red   = simple_blob_analysis(cropped_red_bs, cmap='Reds', smooth_img=True)
    blobs_green = simple_blob_analysis(cropped_green_bs, cmap='Greens', smooth_img=True)

    # shared ROIs
    shared_red_idx, shared_green_idx = find_shared_rois(blobs_red, blobs_green)

    plot_shared_rois(cropped_red_bs, cropped_green_bs, cropped_blue_bs,
                 blobs_red, blobs_green,
                 shared_red_idx, shared_green_idx)
    
    # store and save data
    blob_data = dict()
    blob_data['blobs_red'] = blobs_red
    blob_data['blobs_green'] = blobs_green  
    blob_data['shared_red_idx'] = shared_red_idx
    blob_data['shared_green_idx'] = shared_green_idx
    blob_data['cell_diameter'] = cell_diam  
    # save the blob data to a .mat file
    import scipy.io as sio
    mat_save_path = os.path.splitext(file_path)[0] + "_blobs.mat"
    sio.savemat(mat_save_path, blob_data)
    
    # now identify percentage shared
    percentage_red_in_green = len(shared_red_idx) / len(blobs_red) * 100 if blobs_red else 0
    percentage_green_in_red = len(shared_green_idx) / len(blobs_green) * 100 if blobs_green else 0
    print(f"Percentage of red ROIs in green: {percentage_red_in_green:.1f}%")
    print(f"Percentage of green ROIs in red: {percentage_green_in_red:.1f}%")

    # make a bar plot of the percentages
    plt.figure(figsize=(8, 4))
    plt.bar(['Red in Green', 'Green in Red'], 
            [percentage_red_in_green, percentage_green_in_red], 
            color=['red', 'green'])
    plt.box(False)
    plt.ylabel('Percentage of ROIs')
    plt.title('Percentage of Shared ROIs')

    # 3) visualize on the red and green channels as subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Red channel with ROIs (puncta analyzer)
    axs[0].imshow(cropped_red_bs, cmap='Reds')
    for cy, cx, radius in blobs_red:
        circ = Circle(
            (cx, cy),                # center
            radius,                  # radius
            edgecolor='black', 
            facecolor='none',
            linewidth=2
        )
        axs[0].add_patch(circ)
    axs[0].set_title("Auto-detected circular ROIs (Red Channel)")

    # Green channel with ROIs
    axs[1].imshow(cropped_green_bs, cmap='Greens')
    for cy, cx, radius in blobs_green:
        circ = Circle(
            (cx, cy),                # center
            radius,                  # radius
            edgecolor='black', 
            facecolor='none',
            linewidth=2
        )
        axs[1].add_patch(circ)
    axs[1].set_title("Auto-detected circular ROIs (Green Channel)")
    plt.tight_layout()
    plt.show()

    '''
    # build an svm to classify red_blobs and green_blobs using coordinates
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE

    # make dataframe of red and green blobs
    import pandas as pd
    def make_blob_dataframe(blobs, label):
        data = []
        for cy, cx, radius in blobs:
            data.append({'y': cy, 'x': cx, 'radius': radius, 'label': label})
        return pd.DataFrame(data)
    
    # make blob dataframes
    df_red   = make_blob_dataframe(blobs_red, 'red')

    # concatenate red and green blobs into a single dataframe
    df_green = make_blob_dataframe(blobs_green, 'green')
    df = pd.concat([df_red, df_green], ignore_index=True)

    # drop the radius column
    df = df.drop(columns=['radius'])

    # split into features and labels
    X = df[['y', 'x']].values  # features: y and x
    y = df['label'].values      # labels: 'red' or 'green'
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # svm classifier
    svm = SVC(kernel='poly', degree=5, C=1.0, random_state=42)

    # fit the model
    svm.fit(X_train, y_train)

    # make predictions
    y_pred = svm.predict(X_test)

    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Accuracy: {accuracy:.2f}")

    # show the classification line on the coordinates in X
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 'red', 1], X_train[y_train == 'red', 0], color='red', label='Red Blobs', alpha=0.5)
    plt.scatter(X_train[y_train == 'green', 1], X_train[y_train == 'green', 0], color='green', label='Green Blobs', alpha=0.5)
    plt.scatter(X_test[y_pred == 'red', 1], X_test[y_pred == 'red', 0], color='darkred', label='Predicted Red Blobs', marker='x')
    plt.scatter(X_test[y_pred == 'green', 1], X_test[y_pred == 'green', 0], color='darkgreen', label='Predicted Green Blobs', marker='x')
    plt.gca().invert_yaxis()
    plt.title("SVM Classification of Red and Green Blobs")
    plt.xlabel("X Coordinate")


    # now plot the decision boundary
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                     np.arange(y_min, y_max, 0.1))  
        
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 10),  # 100 points per axis
        np.linspace(y_min, y_max, 10)
    )    
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, cmap='coolwarm')
    plt.legend()
    plt.colorbar(label='Decision Function Value')
    plt.show()
    '''

def plot_rois(img, rois, cmap):
    # 3) visualize on the red and green channels as subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Red channel with ROIs
    axs[0].imshow(img, cmap=cmap)
    for cy, cx, radius in rois:
        circ = Circle(
            (cx, cy),                # center
            radius,                  # radius
            edgecolor='yellow', 
            facecolor='none',
            linewidth=2
        )
        axs[0].add_patch(circ)
