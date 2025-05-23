# pyspell

This repository contains the **pyspell** toolbox, including a Cython extension (**oasis**) that must be built locally on each machine. Follow the instructions below to create a fresh environment, install in editable mode, and keep platform-specific binaries out of Dropbox.

## Folder Structure

```
/Dropbox/SpellmanToolbox/
├── README.md
├── setup.py
├── pyspell/
│   ├── __init__.py
│   ├── oasis.pyx
│   ├── oasis.cpp (generated)
│   ├── oasis.c
│   └── other modules (.py files)
└── environment.yml  (optional)
```

## 1. Create a Fresh Environment

### Using conda

```bash
# Create a new environment named 'pyspell_env' with Python 3.9
conda create -n pyspell_env python=3.9

# Activate the environment
conda activate pyspell_env
```

### (Alternative) Using virtualenv + pip

```bash
# Create a virtual environment
python3 -m venv ~/.venvs/pyspell_env

# Activate
source ~/.venvs/pyspell_env/bin/activate
```

## 2. Install in Editable Mode

From the root of the repository (where `setup.py` lives), run:

```bash
pip install --editable .
```

* This builds the `pyspell.oasis` Cython extension into your active environment’s `site-packages`, not into the `pyspell/` source folder.
* Pure‑Python modules update immediately; to rebuild the Cython extension after editing `oasis.pyx`, re-run the same command.

## 3. Building the Cython Extension Directly

If you ever need to compile in-place (for debugging), you can also run:

```bash
python setup.py build_ext --inplace
```

> **Note:** In-place builds drop platform‑specific binaries (`.so` / `.pyd`) alongside the `.pyx`.
> **Do not sync** these files via Dropbox (see next section).

## 4. Ignoring Compiled Binaries

To prevent Dropbox from syncing platform-specific artifacts, add a `.dropboxignore` (or `.gitignore`) in the `pyspell/` folder:

```
*.so
*.pyd
__pycache__/
```

* If your Dropbox client supports `.dropboxignore`, include the patterns there.
* Otherwise, set selective sync or ignore rules in your Dropbox preferences.

## 5. Reproducing the Environment Elsewhere

Optionally, commit an `environment.yml` (or `requirements.txt`) to reproduce exact dependencies:

### environment.yml (conda)

```yaml
name: pyspell
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - cython
  - pip
  - pip:
    - pyspell==0.1.0
```

```bash
# Create:
conda env create -f environment.yml

# Update later:
conda env update -f environment.yml --prune
```

---

With these steps, you’ll be able to develop **pyspell** on any machine, keep your Dropbox folder clean, and ensure the Cython extension builds correctly in each environment.
