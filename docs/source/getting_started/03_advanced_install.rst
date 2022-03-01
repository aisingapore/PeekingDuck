****************
Advanced Install
****************

.. include:: /include/substitution.rst


This section covers advanced PeekingDuck installation steps for users with ARM64
devices or M1 Macs.


Arm64
=====

To install PeekingDuck on an ARM-based device, such as a Raspberry Pi, include
the ``--no-dependencies`` flag, and separately install the other dependencies
listed in PeekingDuck's `[requirements.txt]
<https://github.com/aimakerspace/PeekingDuck/blob/dev/requirements.txt>`_:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`pip install peekingduck -\-no-dependencies` \
    | \ :blue:`[~user]` \ > [ install additional dependencies as specified within requirements.txt ]



.. _m1_mac_installation:

M1 Mac
======

Apple released Macs with their advanced `M1 <https://en.wikipedia.org/wiki/Apple_M1>`_
ARM-based chip in late 2020, a significant change from the previous Intel processors.
We've successfully installed PeekingDuck on M1 Macs running macOS Big Sur and
macOS Monterey.

    1. Prerequisites:

        - Install `homebrew <https://brew.sh/>`_
        - Install miniforge using homebrew: |br| |br|

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`brew install miniforge` \

    2. Create conda virtual environment and install base packages:

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`conda create -n pkd python=3.8` \
            | \ :blue:`[~user]` \ > \ :green:`conda activate pkd` \
            | \ :blue:`[~user]` \ > \ :green:`conda install click colorama opencv openblas pyyaml requests scipy shapely tqdm` \

    3. Install Apple's Tensorflow build that is optimized for M1 Macs:

        * For macOS Monterey: |br| |br|

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`conda install -c apple tensorflow-deps` \
            | \ :blue:`[~user]` \ > \ :green:`pip install tensorflow-macos tensorflow-metal` \
            | \ :blue:`[~user]` \ > \ :green:`pip install peekingduck —no-dependencies` \

        * For macOS Big Sur: |br| |br|

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`conda install -c apple tensorflow-deps=2.6.0` \
            | \ :blue:`[~user]` \ > \ :green:`pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0` \
            | \ :blue:`[~user]` \ > \ :green:`pip install tensorflow-metal==0.2.0` \

    4. Install PeekingDuck:

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`pip install peekingduck -\-no-dependencies` \

    5. Create a new PeekingDuck project and run it:

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`mkdir pkd_project` \
            | \ :blue:`[~user]` \ > \ :green:`cd pkd_project` \
            | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck init` \
            | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck run` \

    6. Install PyTorch and TorchVision (Unofficial):

        Some PeekingDuck models are built on PyTorch, and require ``torch`` and 
        ``torchvision`` to be installed. |br|
        This guide will install the ``pytorch-cpu`` package from Conda,
        and download ``torchvision`` from GitHub, compile and install it from source.
        Finally, one line needs to be added to a Python file in the installed ``torch``
        package.

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`conda install pytorch-cpu` \
            | \ :blue:`[~user]` \ > \ :green:`git clone https://github.com/pytorch/vision.git` \
            | \ :blue:`[~user]` \ > \ :green:`cd vision` \
            | # For macOS Big Sur:
            | \ :blue:`[~user/vision]` \ > \ :green:`MACOS_DEVELOPMENT_TARGET=11.6 CC=clang CXX=clang++ python setup.py install` \
            | # For macOS Monterey:
            | \ :blue:`[~user/vision]` \ > \ :green:`MACOS_DEVELOPMENT_TARGET=11.7 CC=clang CXX=clang++ python setup.py install` \

        6a) Edit the file ``torch/ao/quantization/__init__.py``. |br|
        This file is located where the ``torch`` package is installed.
        For instance, using ``miniforge`` ``conda`` environment ``pkd`` and Python 3.8, 
        this file is located (on our M1 Macs) at:
        ``/opt/homebrew/Caskroom/miniforge/base/envs/pkd/lib/python3.8/site-packages/torch/ao/quantization/__init__.py``

        6b) Add this line at the end of the file and save.

        .. code-block:: text

            [ original __init__.py contents here ]
            from .stubs import *

        .. note::
            This Torch installation guide is based on our in-house experimentation and
            testing.  As PyTorch and TorchVision are not officially supported on M1 Macs
            currently, we are unable to provide support for their installation if these
            instructions do not work on your M1 system.



Development Version
===================

You can try out the development version of PeekingDuck direct from the
`PeekingDuck Github repository <https://github.com/aimakerspace/PeekingDuck>`_:

    .. admonition:: Terminal Session

        | \ :blue:`[~user]` \ > \ :green:`git clone https://github.com/aimakerspace/PeekingDuck.git` \

| This will clone a copy of PeekingDuck development version into the folder ``PeekingDuck``.
| Test it with the following commands:

    .. admonition:: Terminal Session

        | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \
        | \ :blue:`[~user/PeekingDuck]` \ > \ :green:`python __main__.py` \
