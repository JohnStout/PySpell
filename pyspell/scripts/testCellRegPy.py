import sys
from pathlib import Path

# Get the pyspell directory - works in PyCharm, Spyder, IPython, and command line
try:
    # When running as a script
    script_dir = Path(__file__).resolve().parent  # .../pyspell/scripts
    pyspell_dir = script_dir.parent               # .../pyspell
except NameError:
    # When running in IPython/Jupyter interactive mode
    pyspell_dir = Path.cwd()
sys.path.insert(0, str(pyspell_dir))

from pyspell.cellregpy import CellRegPy, CellRegConfig

# 1. Configure parameters
# Set microns_per_pixel to match your microscope calibration
config = CellRegConfig(
    microns_per_pixel=2.0,
    figures_visibility='on'
)

# 2. Initialize the pipeline
cellreg = CellRegPy(config)
cellreg.config.use_parallel_processing = False # for purposes of debugging

# 3. Run for one or more mouse folders
folder = r"X:\John\Subjects - GCaMP Recordings\L632_F_LeftPFC_L6Chrimson_L5CTrec-FLEXgcamp6f"
folder = r"C:\Users\johnj\SpellmanLab Dropbox\OtherData\Manuscripts\in prep\L6CTopto_panneuronal_experiment\data\subjects_superalignment\L612_F_RightPFC_L6Chr_PFCgcamp6f_L6PAN"
mouse_folders = [Path(folder)]
mouse_table, mouse_data = cellreg.run(mouse_folders)

# The results are automatically saved in MouseA/1_CellReg/