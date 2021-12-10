Overview
=========

This project contains tools to compute placement hints.
Placement hints are used by the placement algorithm to place 3D cells in the context
of circuit building.

After installation, you can display the available command lines with the following ``bash`` command:
.. code-block:: bash

    atlas-placement-hints --help

Installation
============

This python project depends on Ultraliser_ for the algorithms which
follow a mesh-based approach.

Once Ultraliser_ is installed, run the following ``bash`` commands:

.. code-block:: bash

    git clone git@bbpgitlab.epfl.ch:nse/atlas-placement-hints.git
    cd atlas-placement-hints
    pip install -e .


Instructions for developers
===========================

Run the following commands before submitting your code for review:

.. code-block:: bash

    cd atlas-placement-hints
    isort -l 100 --profile black atlas_placement_hints tests setup.py
    black -l 100 atlas_placement_hints tests setup.py

These formatting operations will help you pass the linting check `testenv:lint` defined in
`tox.ini`.

.. _Ultraliser: https://github.com/BlueBrain/Ultraliser