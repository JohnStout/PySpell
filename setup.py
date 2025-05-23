from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

# Map the 'pyspell/' directory as the source for the 'pyspell' package
setup(
    name="pyspell",
    version="0.1.0",
    package_dir={"": "pyspell"},            # Look for packages under pyspell/
    packages=find_packages(where="pyspell"),  # Find any subpackages in pyspell/
    ext_modules=cythonize(
        [
            Extension(
                "pyspell.oasis",                # Importable as pyspell.oasis
                sources=["pyspell/oasis.pyx"],    # Point to the .pyx in pyspell/
                include_dirs=[np.get_include()],
                language="c++",
                extra_compile_args=["-O2"],
            )
        ],
        language_level="3",
    ),
    install_requires=[
        # Runtime dependencies:
        # "numpy>=1.18.0", "cython"
    ],
)
