**************************
Installation Prerequisites
**************************

Operating Systems
=================

    PeekingDuck supports the following operating systems:

    * Windows (10 / 11)
    * Linux
    * MacOS (Intel / M1, Big Sur/ Monterey)

    Optional:

    * Anaconda/Conda


Virtual Environments
====================

    It is recommended to install PeekingDuck in a Python virtual environment, as
    it creates an isolated environment for a Python project to install its own
    dependencies and avoid package version conflicts with other projects.

    There are two ways to create Python virtual environments:

    #. Using `conda`

        Create a new `conda` virtual environment using Python 3.8::

        > conda create -n pkd python=3.8

        To activate the new Python 3.8 environment::

        > conda activate pkd

        To exit::

        > conda deactivate


    #. Using `venv`

        Create a new virtual environment using the current installed version of Python 3::

        > python3 -m venv pkd

        This will create a new folder named ``pkd`` and house the new environment within it.

        To activate the newly created environment::

        > source pkd/bin/activate

        To exit::

        > deactivate

