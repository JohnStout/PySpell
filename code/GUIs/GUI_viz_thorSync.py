from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
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

        self.canvas = FigureCanvas(plt.Figure())
        self.ax = self.canvas.figure.subplots()

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.button = QPushButton('Load Episode.h5', self)
        self.button.clicked.connect(self.import_data)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.bData = None
        self.piezo_norm = None

    def plot_data(self):
        if self.bData is not None and self.piezo_norm is not None:
            self.ax.clear()
            self.ax.plot(self.piezo_norm, label='Piezo Norm')
            self.ax.plot(self.bData['FrameOut'], label='FrameOut')
            self.ax.legend()
            max_length = max(len(self.piezo_norm), len(self.bData['FrameOut']))
            self.ax.set_xlim(0, max_length - 1)  # Set x-axis limits to the length of the longest array
            self.ax.set_ylim(min(np.min(self.piezo_norm), np.min(self.bData['FrameOut'])),
                             max(np.max(self.piezo_norm), np.max(self.bData['FrameOut'])))  # Set y-axis limits to include all data
            self.canvas.draw()

    def import_data(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Import ThorSync", "", "H5 Files (*.h5);;All Files (*)", options=options)
        if file_path:
            self.bData, self.piezo_norm = importThorsync(file_path)
            self.plot_data()

def importThorsync(bpath):
    bpath = os.path.abspath(bpath)
    ext = os.path.splitext(bpath)[1]

    if not ext.endswith('.h5'):
        dirFiles = os.listdir(bpath)
        fnames = [f for f in dirFiles if f.endswith('.h5')]
        if not fnames:
            raise FileNotFoundError("No .h5 file found in the directory")
        fileName = os.path.join(bpath, fnames[0])
    else:
        fileName = bpath

    dataIn = h5py.File(fileName, 'r')
    bData = {i: np.ravel(dataIn['DI'][i][:] / np.max(dataIn['DI'][i][:])) for i in dataIn['DI'].keys() if np.max(dataIn['DI'][i][:]) > 0}
    frameTimes = np.where(np.diff(bData['FrameOut'], axis=0) == 1)[0]
    piezo = dataIn['AI']['PiezoMonitor'][:]
    piezo_norm = np.ravel(piezo / np.max(piezo))
    idx_offrec = np.where(piezo_norm < 0.3)[0]
    good_rec = (idx_offrec[0] < frameTimes[0]) & (idx_offrec[-1] > frameTimes[-1])
    return bData, piezo_norm

if __name__ == "__main__":
    app = QApplication([])
    ex = App()
    ex.show()
    app.exec_()