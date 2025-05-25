# pyspell

This repository contains the **pyspell** toolbox.

Use the instructions below to **download**, configure, and install the package in an isolated environment.

## Folder Structure

```
/Dropbox/SpellmanToolbox/
├── README.md
├── setup.py
├── environment.yml  # optional
└── pyspell/
    ├── __init__.py
    ├── s2pfuns.py         # code to run suite2p and postprocessing steps (detrending/deconvolution, ROI classification, cell-reg prep)
    ├── thorfuns.py        # code to manage thorlabs 2P imaging files for conversion
    ├── rootfun.py         # some directory management
    ├── deconvolution.py   # Straight from caiman
    ├── nwbfun.py          # In progress code to convert files to NWB format
    ├── sessreg.py         # Code to manage cell-reg preparation and may also house other options for cell alignment later
    └── other modules (.py files)
```

## 1. Clone (Download) the Repository

Replace `<repo_url>` with the URL for this repository.

```bash
# Clone the repo to your local machine
git clone <repo_url>

# Change into the toolbox directory:
cd /path/to/your/PySpell
```

## 2. Create & Activate Conda Environment

We provide a single `environment.yml` that includes all conda and pip dependencies, and installs the package in editable mode.

```bash
# Create the 'pyspell' environment (first time only)
conda env create -f environment.yml

# If you already have the environment, update it after changes:
conda env update -f environment.yml --prune

# Activate:
conda activate pyspell
```

> **Note:** This sets up Python 3.9, NumPy, oasis-deconv (prebuilt), and all other requirements.


## 3. Verify Installation

After activating, confirm you can import the toolbox and its dependencies:

```python
python - << 'EOF'
import s2pfuns          # should load without error
import oasis            # from oasis-deconv
print("pyspell and oasis loaded successfully")
EOF
```
