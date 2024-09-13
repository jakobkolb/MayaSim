# MayaSim

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13734595.svg)](https://doi.org/10.5281/zenodo.13734595)

An agent-based model of the ancient Maya social-ecological system.

MayaSim is an integrated agent-based, cellular-automaton and network simulation of the Maya civilization on the Yucatan peninsula throughout the Mayan Classic epoch. Agents are settlements situated on a gridded spatial landscape. These settlements grow and interact with their surrounding forest ecosystem through use of ecosystem services, agriculture induced soil erosion and forest succession. The model aims to reproduce interrelations between climate variability, primary production, hydrology, ecosystem services, forest succession, agricultural production, population growth and the stability of trade networks. The project is based on previous work by Scott Heckbert ([2013](https://www.comses.net/codebases/3063/releases/1.3.0/)) published in Netlogo, and is reimplemented and further developed in Python. The focus lies firstly on the implementation of realistic approaches to modeling agents' income from agriculture and trade. The goal is to first test the model's structural stability and parameter sensitivity, and then to test the plausibility of different hypotheses on the causes of the ancient Maya's spectacular rise and subsequent catastrophic reorganization.

## Install

Install MayaSim in a [`conda`](https://docs.anaconda.com/miniconda/) environment following these steps:

1. Create and activate environment:
    ```
    $> conda create -n mayasim -c conda-forge python=3.11 numpy=1 mpi4py gfortran pandas networkx scipy sympy matplotlib tqdm pytables notebook
    $> conda activate mayasim
    ```
    _Note: MayaSim has successfully been run on `python>=3.9, <=3.11`._

2. Download/clone MayaSim to a local directory. Then navigate to this directory and install with `pip`:
    ```
    $> cd path/to/MayaSim
    $> pip install .
    ```

3. Build MayaSim's Fortran extension module by running the included script:
    ```
    $> cd mayasim/model/_ext
    $> sh f90makefile.sh
    ```

## Run

Import and run MayaSim from a Python script:

```python
from mayasim.model.core import Core as MayaSim

model = MayaSim()
model.run(steps=350)
```

Take a look into the quickstart-tutorial in [docs](docs) for some example settings.
