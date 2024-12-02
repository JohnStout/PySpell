# setup.py
#
# You need to compile oasis before calling it

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extra_compiler_args = ["-O2"]  # Example compiler arguments, adjust as needed

extensions = [
    Extension(
        "code.oasis",
        sources=["code/oasis.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compiler_args,
        extra_link_args=extra_compiler_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)


