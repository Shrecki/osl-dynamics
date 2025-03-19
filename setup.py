# This is the build script for setuptools.
# See: https://packaging.python.org/en/latest/tutorials/packaging-projects/

from setuptools import setup

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="osl_dynamics.inference.batched_cov",  # Note the full module name.
        sources=["osl_dynamics/inference/batched_cov.pyx"],  # Path to the .pyx file.
        include_dirs=[np.get_include()],  # Include NumPy headers.
        extra_compile_args=["-O3"],  # Optimization flag.
    )
]

setup(
    name="osl_dynamics",
    ext_modules=cythonize(extensions, language_level="3"),
    packages=["osl_dynamics", "osl_dynamics.inference"],
)


if __name__ == "__main__":
    setup(long_description_content_type='text/markdown')
