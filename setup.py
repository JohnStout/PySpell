from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name="pyspell",
    version="0.1.0",
    package_dir={"": "pyspell"},
    packages=find_packages(where="pyspell"),

    install_requires=[
        "numpy>=1.18.0",
        "cython",
        "xmltodict",
        "h5py",
        "suite2p>=0.14",   # let suite2p pull in its own preferred hdmf/pynwb versions
        "emd-signal",
    ],
)
