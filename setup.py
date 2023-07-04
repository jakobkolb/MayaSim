"""
This is the setup.py for the MayaSim Model

for developers: recommended way of installing is to run in this directory
pip install -e .
This creates a link insteaed of copying the files, so modifications in this
directory are modifications in the installed package.
"""

from setuptools import setup

setup(name="mayasim",
      version="0.0.1",
      description="to be added",
      url="to be added",
      author="Jakob. J. Kolb",
      author_email="kolb@pik-potsdam.de",
      license="MIT",
      packages=["mayasim"],
      include_package_data=True,
      install_requires=[
          "numpy>=1.18.0",
          "pandas>=0.22.0",
          "pymofa @ git+https://github.com/pik-copan/pymofa@master",
          "networkx",
          "pickleshare",
          "scipy",
          "matplotlib", 
          "tqdm"

      ],
      # see http://stackoverflow.com/questions/15869473/what-is-the-advantage-
      # of-setting-zip-safe-to-true-when-packaging-a-python-projec
      zip_safe=False)
