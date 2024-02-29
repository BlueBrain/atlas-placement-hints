#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.rst") as f:
    README = f.read()

setup(
    name="atlas-placement-hints",
    author="Blue Brain Project, EPFL",
    description="Library containing command lines and tools to compute placement hints",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/BlueBrain/atlas-placement-hints",
    download_url="https://github.com/BlueBrain/atlas-placement-hints",
    license="Apache-2",
    python_requires=">=3.7.0",
    install_requires=[
        "atlas-commons>=0.1.4",
        "cached-property>=1.5.2",
        "click>=7.0",
        "cgal-pybind>=0.1.4",
        "networkx>=2.4",  # soft dep required for trimesh to allow 'repair'
        "numpy>=1.15.0",
        "rtree>=0.8.3",  # soft dep required for trimesh to allow indexing
        "scipy>=1.4.1",
        "tqdm>=4.44.1",
        "trimesh>=2.38.10",
        "voxcell>=3.0.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "tests": [
            "pytest>=4.4.0",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["atlas-placement-hints=atlas_placement_hints.app.cli:cli"]},
    use_scm_version={
        "local_scheme": "no-local-version",
    },
    setup_requires=[
        "setuptools_scm",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
