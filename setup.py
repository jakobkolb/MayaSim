"""
This is the setup.py for the MayaSim Model

for developers: recommended way of installing is to run in this directory
pip install -e .
This creates a link insteaed of copying the files, so modifications in this
directory are modifications in the installed package.
"""

from setuptools import setup

setup(
    name='mayasim',
    version='1.3.0',
    description='An agent-based model of the ancient Maya social-ecological system',
    url='https://github.com/pik-copan/MayaSim',
    author='Jakob J. Kolb <kolb@pik-potsdam.de>, Fritz Kuehlein <fritzku@pik-potsdam.de>',
    license='MIT',
    packages=['mayasim'],
    include_package_data=True,
    install_requires=[
        'numpy>=1.26.0, < 2.0',
        'pandas>=2.0.0',
        'networkx',
        'scipy',
        'matplotlib',
        'tqdm',
        'pymofa @ git+https://github.com/pik-copan/pymofa.git',
        'mpi4py',
        'tables'
    ],
    # see http://stackoverflow.com/questions/15869473/what-is-the-advantage-
    # of-setting-zip-safe-to-true-when-packaging-a-python-projec
    zip_safe=False)
