from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSpinBox, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Interactive Plot Navigator')
        self.setGeometry(100, 100, 800, 600)

        # Initialize plot canvas and toolbar
        self.canvas = FigureCanvas(plt.Figure())
        self.ax = self.canvas.figure.subplots()
        self.toolbar = NavigationToolbar(self.canvas, self)

        # QLabel to display the last four folders
        self.folder_label = QLabel("Last Folders: None", self)

        # Buttons to load data and select plotting function
        self.load_button = QPushButton('Load Episode.h5', self)
        self.load_button.clicked.connect(self.import_data)

        self.plot_chunk_button = QPushButton('Plot Chunk (Custom Time)', self)
        self.plot_chunk_button.clicked.connect(self.plot_chunk)

        self.plot_full_signal_button = QPushButton('Plot All (EXPECT LAG)', self)  # Button for full signal plot
        self.plot_full_signal_button.clicked.connect(self.plot_full_signal)

        # Input fields for user-defined time ranges
        self.first_time_label = QLabel("First Segment (min):", self)
        self.first_time_input = QSpinBox(self)
        self.first_time_input.setMinimum(0)  # Minimum time in minutes
        self.first_time_input.setMaximum(60)  # Maximum of 60 minutes
        self.first_time_input.setValue(0)  # Default value

        self.last_time_label = QLabel("Last Segment (min):", self)
        self.last_time_input = QSpinBox(self)
        self.last_time_input.setMinimum(0)  # Minimum time in minutes
        self.last_time_input.setMaximum(60)  # Maximum of 60 minutes
        self.last_time_input.setValue(5)  # Default value

        # Arrange input fields and buttons
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.first_time_label)
        input_layout.addWidget(self.first_time_input)
        input_layout.addWidget(self.last_time_label)
        input_layout.addWidget(self.last_time_input)
        input_layout.addWidget(self.plot_chunk_button)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.toolbar)
        button_layout.addWidget(self.canvas)
        button_layout.addWidget(self.folder_label)  # Add the QLabel here
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.plot_full_signal_button)  # Add button for full signal
        button_layout.addLayout(input_layout)

        container = QWidget()
        container.setLayout(button_layout)
        self.setCentralWidget(container)

        # Initialize file path variables
        self.file_path = None

    def plot_chunk(self):
        if self.file_path is not None:
            frame_rate = 1000  # 1000 Hz
            first_time_min = self.first_time_input.value()  # User-specified first segment in minutes
            last_time_min = self.last_time_input.value()  # User-specified last segment in minutes

            n_samples_first = frame_rate * 60 * first_time_min
            n_samples_last = frame_rate * 60 * last_time_min

            with h5py.File(self.file_path, 'r') as dataIn:
                # Lazily load only the required slices of data
                piezo_first = dataIn['AI']['PiezoMonitor'][:n_samples_first] if n_samples_first > 0 else []
                piezo_last = dataIn['AI']['PiezoMonitor'][-n_samples_last:] if n_samples_last > 0 else []
                piezo_concat = np.concatenate((piezo_first, piezo_last)) if len(piezo_first) + len(piezo_last) > 0 else []
                if len(piezo_concat) > 0:
                    piezo_concat = piezo_concat / np.max(piezo_concat)  # Normalize

                frameout_first = dataIn['DI']['FrameOut'][:n_samples_first] if n_samples_first > 0 else []
                frameout_last = dataIn['DI']['FrameOut'][-n_samples_last:] if n_samples_last > 0 else []
                frameout_concat = np.concatenate((frameout_first, frameout_last)) if len(frameout_first) + len(frameout_last) > 0 else []
                if len(frameout_concat) > 0:
                    frameout_concat = frameout_concat / np.max(frameout_concat)  # Normalize

                self.ax.clear()
                if len(piezo_concat) > 0 and len(frameout_concat) > 0:
                    self.ax.plot(piezo_concat, label=f'Piezo Norm (First {first_time_min} min & Last {last_time_min} min)')
                    self.ax.plot(frameout_concat, label=f'FrameOut (First {first_time_min} min & Last {last_time_min} min)')
                    self.ax.legend()

                max_length = max(len(piezo_concat), len(frameout_concat))
                self.ax.set_xlim(0, max_length - 1)
                self.ax.set_ylim(min(np.min(piezo_concat), np.min(frameout_concat)),
                                 max(np.max(piezo_concat), np.max(frameout_concat)) if len(piezo_concat) > 0 and len(frameout_concat) > 0 else (0, 1))

                self.canvas.draw()

    def plot_full_signal(self):
        if self.file_path is not None:
            with h5py.File(self.file_path, 'r') as dataIn:
                # Load the entire signal from the dataset
                piezo = dataIn['AI']['PiezoMonitor'][:]
                piezo_norm = piezo / np.max(piezo)  # Normalize

                frame_out = dataIn['DI']['FrameOut'][:]
                frame_out_norm = frame_out / np.max(frame_out)  # Normalize

                self.ax.clear()
                self.ax.plot(piezo_norm, label='Piezo Norm (Full Signal)')
                self.ax.plot(frame_out_norm, label='FrameOut (Full Signal)')
                self.ax.legend()

                max_length = max(len(piezo_norm), len(frame_out_norm))
                self.ax.set_xlim(0, max_length - 1)
                self.ax.set_ylim(min(np.min(piezo_norm), np.min(frame_out_norm)),
                                 max(np.max(piezo_norm), np.max(frame_out_norm)))

                self.canvas.draw()

    def import_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Import ThorSync", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if file_path:
            self.file_path = file_path

            # Extract the last 4 folders in the directory path
            folder_path = os.path.dirname(self.file_path)  # Get the directory of the file
            path_parts = folder_path.split(os.sep)  # Split by the system's path separator
            last_folders = path_parts[-4:] if len(path_parts) >= 4 else path_parts  # Get the last 4 folders or fewer if not available

            # Update the QLabel to display the last 4 folders
            self.folder_label.setText(f"Last Folders: {' > '.join(last_folders)}")


if __name__ == "__main__":
    app = QApplication([])
    ex = App()
    ex.show()
    app.exec_()
