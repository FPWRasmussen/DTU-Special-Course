from setuptools import setup
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

setup(
    ext_modules=cythonize("solve_shadow_map.pyx", annotate=True),
    include_dirs=[numpy.get_include()]
)