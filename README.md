# pyspell

This repository contains the **pyspell** toolbox, including a Cython extension (`oasis`) that must be built locally on each machine. Use the instructions below to **download**, configure, and install the package in an isolated environment.

## Folder Structure

```
/Dropbox/SpellmanToolbox/
├── README.md
├── setup.py
├── environment.yml  # optional
└── pyspell/
    ├── __init__.py
    ├── s2pfuns.py
    ├── thorfuns.py       # generated
    ├── rootfun.py         # generated
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

> **Note:** This sets up Python 3.9, NumPy, Cython, oasis-deconv (prebuilt), and all other requirements, including installing this repo with `-e .`.


## 3. Verify Installation

After activating, confirm you can import the toolbox and its dependencies:

```python
python - << 'EOF'
import s2pfuns          # should load without error
import oasis            # from oasis-deconv
print("pyspell and oasis loaded successfully")
EOF
```
