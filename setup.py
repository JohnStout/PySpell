from setuptools import setup, find_packages

setup(
    name="pyspell",
    version="0.1.0",
    description="Spellman Lab Python Toolkit for calcium imaging analysis",
    author="John Stout - Spellman Lab",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core scientific stack
        "numpy>=1.18.0",
        "scipy>=1.5.0",
        "pandas>=1.0.0",
        "matplotlib>=3.2.0",
        
        # Imaging
        "tifffile>=2020.9.0",
        "suite2p>=0.14",
        
        # Machine learning
        "scikit-learn>=0.23.0",
        
        # Signal processing
        "emd-signal",
        "PyEMD",
        
        # File formats
        "xmltodict",
        "h5py",
        
        # Build tools (for oasis)
        "cython",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
