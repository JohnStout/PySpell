import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # For video rendering in Tkinter
import threading
import time

class AVIGUIPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("AVI Player with Trimming and Export Options")

        self.file_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.playing = False
        self.current_frame = 0

        # UI Components
        tk.Button(root, text="Load AVI File", command=self.load_video).pack(pady=5)
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Playback Speed Dropdown
        tk.Label(root, text="Playback Speed:").pack()
        self.playback_speed = tk.StringVar(value="1x")
        self.speed_dropdown = tk.OptionMenu(root, self.playback_speed, "1x", "2x", "3x", "4x")
        self.speed_dropdown.pack()

        # Scroll Bar for Video Frames
        self.scroll_bar = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.scroll_video)
        self.scroll_bar.pack(fill=tk.X)

        # Start and Stop Time for Trimming
        tk.Label(root, text="Start Time (seconds):").pack()
        self.start_time_entry = tk.Entry(root)
        self.start_time_entry.pack()
        tk.Label(root, text="Stop Time (seconds):").pack()
        self.stop_time_entry = tk.Entry(root)
        self.stop_time_entry.pack()

        # Trim Options
        tk.Button(root, text="Trim and Save as AVI", command=self.trim_and_save_avi).pack(pady=5)
        tk.Button(root, text="Trim and Save as GIF", command=self.trim_and_save_gif).pack(pady=5)

        # Play and Stop Buttons
        tk.Button(root, text="Play", command=self.play_video).pack(pady=5)
        tk.Button(root, text="Stop", command=self.stop_video).pack(pady=5)

        self.feedback_label = tk.Label(root, text="", fg="blue")
        self.feedback_label.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_feedback(self, message):
        """Set feedback message in the UI."""
        self.feedback_label.config(text=message)
        self.root.update_idletasks()

    def load_video(self):
        """Load an AVI file and initialize parameters."""
        self.file_path = filedialog.askopenfilename(
            title="Select an AVI File",
            filetypes=[("AVI files", "*.avi")]
        )
        if not self.file_path:
            self.set_feedback("No file selected.")
            return

        # Open video file
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            self.set_feedback("Failed to load video file.")
            return

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Configure scroll bar with total frames
        self.scroll_bar.config(to=self.total_frames)
        self.set_feedback(f"Loaded: {self.file_path}")
        self.display_frame(0)

    def display_frame(self, frame_number):
        """Display a specific frame in the video."""
        if not self.cap or not self.cap.isOpened():
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize the frame for display
        max_width, max_height = 640, 360
        height, width, _ = frame.shape
        scale = min(max_width / width, max_height / height)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Convert the frame to RGB for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_label.config(image=frame_image)
        self.video_label.image = frame_image

    def play_video(self):
        """Play the video based on the selected speed."""
        if not self.cap or not self.cap.isOpened():
            self.set_feedback("Please load a video first.")
            return

        if self.playing:  # Prevent multiple threads
            self.set_feedback("Video already playing.")
            return

        self.playing = True
        threading.Thread(target=self._playback, daemon=True).start()

    def _playback(self):
        """Handle video playback with speed control."""
        speed_factor = int(self.playback_speed.get().replace("x", ""))
        playback_delay = 1 / (self.fps * speed_factor)

        while self.playing and self.current_frame < self.total_frames:
            self.display_frame(self.current_frame)
            self.current_frame += 1
            self.scroll_bar.set(self.current_frame)  # Update scroll bar position
            time.sleep(playback_delay)

        self.playing = False

    def stop_video(self):
        """Stop video playback."""
        self.playing = False

    def scroll_video(self, frame):
        """Manually scroll through the video frames."""
        if not self.cap or not self.cap.isOpened():
            return

        self.current_frame = int(frame)
        self.display_frame(self.current_frame)

    def trim_and_save_avi(self):
        """Trim the video and save as a new AVI file."""
        if not self.cap or not self.cap.isOpened():
            self.set_feedback("Please load a video first.")
            return

        # Retrieve start and stop frames directly
        try:
            start_frame = int(self.start_time_entry.get())
            stop_frame = int(self.stop_time_entry.get())
        except ValueError:
            self.set_feedback("Invalid start or stop frame entered.")
            return

        if start_frame >= stop_frame or start_frame < 0 or stop_frame > self.total_frames:
            self.set_feedback("Start frame must be less than stop frame and within range.")
            return

        # Save the trimmed video
        output_file_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            filetypes=[("AVI files", "*.avi")],
            title="Save Trimmed AVI As"
        )
        if not output_file_path:
            self.set_feedback("Save operation canceled.")
            return

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_video = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (frame_width, frame_height))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_number in range(start_frame, stop_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            output_video.write(frame)

        output_video.release()
        self.set_feedback(f"Trimmed AVI saved as: {output_file_path}")
            
    def trim_and_save_gif(self):
        """Trim the video and save as a GIF."""
        if not self.cap or not self.cap.isOpened():
            self.set_feedback("Please load a video first.")
            return

        # Retrieve start and stop frames directly
        try:
            start_frame = int(self.start_time_entry.get())
            stop_frame = int(self.stop_time_entry.get())
        except ValueError:
            self.set_feedback("Invalid start or stop frame entered.")
            return

        if start_frame >= stop_frame or start_frame < 0 or stop_frame > self.total_frames:
            self.set_feedback("Start frame must be less than stop frame and within range.")
            return

        # Save the trimmed GIF
        output_file_path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF files", "*.gif")],
            title="Save Trimmed GIF As"
        )
        if not output_file_path:
            self.set_feedback("Save operation canceled.")
            return

        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_number in range(start_frame, stop_frame + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        # Calculate the frame duration in milliseconds
        # The duration is 1000ms divided by the FPS (frames per second)
        #frame_duration = int(1000 / self.fps)

        frame_duration = int(1000 / self.fps) * 1.5  # Multiply by 2 to slow down the playback        

        # Save the GIF with the calculated duration
        frames[0].save(output_file_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)
        self.set_feedback(f"Trimmed GIF saved as: {output_file_path}")
                
    def on_close(self):
        """Handle application close."""
        self.stop_video()
        if self.cap:
            self.cap.release()
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AVIGUIPlayer(root)
    root.mainloop()