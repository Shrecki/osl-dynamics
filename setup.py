from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="osl_dynamics.inference.batched_cov",
        sources=["osl_dynamics/inference/batched_cov.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    ),
    Pybind11Extension("osl_dynamics.models._hmmc", ["osl_dynamics/models/_hmmc.cpp"], cxx_std=11),
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
