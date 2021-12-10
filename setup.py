#!/usr/bin/env python

import imp

from setuptools import find_packages, setup

VERSION = imp.load_source("", "atlas_placement_hints/version.py").__version__

setup(
    name="atlas-placement-hints",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing command lines and tools to compute placement hints",
    url="https://bbpgitlab.epfl.ch/nse/atlas-placement-hints",
    download_url="git@bbpgitlab.epfl.ch:nse/atlas-placement-hints.git",
    license="BBP-internal-confidential",
    python_requires=">=3.6.0",
    install_requires=[
        "atlas-commons>=0.1.2",
        "cached-property>=1.5.2",
        "click>=7.0",
        "cgal_pybind>=0.1.1",
        "networkx>=2.4",
        "nptyping>=1.0.1",
        "numpy>=1.15.0",
        "rtree>=0.8.3",
        "scipy>=1.4.1",
        "tqdm>=4.44.1",
        "trimesh>=2.38.10",
        "voxcell>=3.0.0",
    ],
    extras_require={
        "tests": ["pytest>=4.4.0", "mock>=2.0.0"],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["atlas-placement-hints=atlas_placement_hints.app.cli:cli"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
