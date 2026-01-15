# stitch_gui.py
"""
GUI wrapper for `stitch_cam_to_avi` (defined in **thorfuns.py**).

💡 **Usage**
1. Place `stitch_gui.py` in the *same* folder as `thorfuns.py` **or** make
   sure that folder is on PYTHONPATH.
2. Run the script (double‑click or `python stitch_gui.py`).
3. Pick the folder containing your multi‑page TIFFs.
4. Adjust FPS if needed and tick *Delete original TIFFs* if you want the
   source stacks removed after a successful conversion.

The GUI runs the heavy work in a background thread so it stays responsive.
"""

from __future__ import annotations  # -> keeps | union if you later upgrade to 3.10+
import os, sys, threading, queue, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

# ------------------------------------------------------------------
# Make sure we can import thorfuns ------------------------------------------------
# If thorfuns.py sits next to this file, the default import below will work.
# Otherwise, uncomment the two lines and adjust the path:
# MODULE_DIR = r"C:\path\to\your\modules"
# sys.path.insert(0, MODULE_DIR)

try:
    from thorfuns import batch_stitch_folders  # ← your original function lives here
except ImportError as exc:
    # Show a GUI dialog *before* raising so the user understands what happened.
    root = tk.Tk(); root.withdraw()
    messagebox.showerror(
        "Import error",
        "Could not import 'stitch_cam_to_avi' from thorfuns.py.\n\n"
        "• Ensure thorfuns.py is in the same folder OR on PYTHONPATH\n"
        "• Then re‑run this program."
    )
    raise

# ------------------------------------------------------------------
class StitchGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Multi‑Folder TIFF → AVI Stitcher")
        self.resizable(False, False)

        # Tk variables
        self.fps_var = tk.DoubleVar(value=16.5)
        self.del_var = tk.BooleanVar(value=False)

        # internal list of folders
        self.folders: list[str] = []

        self._build_widgets()

    # ---------- layout --------------------------------------------------
    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self); frm.grid(sticky="nsew")

        ttk.Label(frm, text="Folders to stitch:").grid(
            row=0, column=0, sticky="w", **pad)

        # list‑box + scrollbar
        self.listbox = tk.Listbox(frm, height=6, width=48, selectmode=tk.EXTENDED)
        yscroll = ttk.Scrollbar(frm, orient="vertical",
                                command=self.listbox.yview)
        self.listbox.config(yscrollcommand=yscroll.set)
        self.listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", **pad)
        yscroll.grid(row=1, column=2, sticky="ns", **pad)

        ttk.Button(frm, text="Add folder…", command=self._add_folder) \
            .grid(row=2, column=0, sticky="w", **pad)
        ttk.Button(frm, text="Remove selected", command=self._remove_selected) \
            .grid(row=2, column=1, sticky="e", **pad)

        ttk.Label(frm, text="FPS:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.fps_var, width=8).grid(
            row=3, column=0, sticky="e", **pad)

        ttk.Checkbutton(frm, text="Delete original TIFFs after success",
                        variable=self.del_var) \
            .grid(row=4, column=0, columnspan=2, sticky="w", **pad)

        ttk.Button(frm, text="Start", command=self._start) \
            .grid(row=5, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(frm, mode="indeterminate")
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)

        frm.columnconfigure(1, weight=1)

    # ---------- callbacks ----------------------------------------------
    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with TIFFs")
        if folder and folder not in self.folders:
            self.folders.append(folder)
            self.listbox.insert(tk.END, folder)

    def _remove_selected(self):
        sel = list(self.listbox.curselection())[::-1]   # remove bottom‑up
        for idx in sel:
            folder = self.listbox.get(idx)
            self.folders.remove(folder)
            self.listbox.delete(idx)

    def _start(self):
        if not self.folders:
            messagebox.showwarning("No folders",
                                   "Please add at least one folder.")
            return
        try:
            fps = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("Invalid FPS", "FPS must be a number.")
            return

        self._toggle_state(tk.DISABLED)
        self.progress.start()

        threading.Thread(
            target=self._worker,
            args=(self.folders.copy(), fps, self.del_var.get()),
            daemon=True
        ).start()

    # ---------- background stitching -----------------------------------
    def _worker(self, folders, fps, delete_src):
        try:
            batch_stitch_folders(
                folders,
                fps=fps,
                folders_parallel=min(10, os.cpu_count()),   #  adjust if desired
                delete_tifs=delete_src,
                # any extra stitch_cam_to_avi kwargs:
                workers=os.cpu_count()//2,
                pages_per_chunk=345,
                max_inflight=8
            )
            self._notify_done(True, "Finished all folders!")
        except Exception as e:
            self._notify_done(False, str(e))

    # ---------- UI helpers ---------------------------------------------
    def _notify_done(self, success, msg):
        def _finish():
            self.progress.stop()
            self._toggle_state(tk.NORMAL)
            (messagebox.showinfo if success else messagebox.showerror)(
                "Done" if success else "Error", msg)
        self.after(0, _finish)

    def _toggle_state(self, state):
        for child in self.winfo_children():
            for w in child.winfo_children():
                if isinstance(w, (ttk.Entry, ttk.Button, ttk.Checkbutton,
                                  tk.Listbox)):
                    w["state"] = state
                    
if __name__ == "__main__":
    StitchGUI().mainloop()
