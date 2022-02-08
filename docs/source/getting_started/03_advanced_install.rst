****************
Advanced Install
****************

.. |br| raw:: html

   <br />

This section covers advanced PeekingDuck installation steps for users with ARM64
devices or M1 Macs.


Arm64
=====

To install PeekingDuck on an ARM-based device, such as a Raspberry Pi, include
the ``--no-dependencies`` flag, and separately install the other dependencies
listed in PeekingDuck's `[requirements.txt]
<https://github.com/aimakerspace/PeekingDuck/blob/dev/requirements.txt>`_::

    > pip install peekingduck --no-dependencies
    > [ install additional dependencies as specified within requirements.txt ]


.. _m1_mac_installation:

M1 Mac
======

Apple released Macs with their advanced `M1 <https://en.wikipedia.org/wiki/Apple_M1>`_
ARM-based chip in late 2020, a significant change from the previous Intel processors.
We've successfully installed PeekingDuck on M1 Macs running MacOS Big Sur and
MacOS Monterey.

    1. Prerequisites:

        - Install [homebrew](https://brew.sh/)
        - Install miniforge using homebrew:

        ::

        > brew install miniforge

    2. Create conda virtual environment and install base packages::

        > conda create -n pkd python=3.8
        > conda activate pkd
        > conda install click colorama opencv openblas pyyaml requests scipy shapely tqdm

    3. Install Apple's Tensorflow build that is optimised for M1 Macs:

        * For MacOS Monterey:

        ::
        
        > conda install -c apple tensorflow-deps
        > pip install tensorflow-macos tensorflow-metal
        > pip install peekingduck —no-dependencies

        * For MacOS Big Sur:

        ::

        > conda install -c apple tensorflow-deps=2.6.0
        > pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0
        > pip install tensorflow-metal==0.2.0

    4. Install PeekingDuck::

        > pip install peekingduck --no-dependencies

    5. Create a new PeekingDuck project and run it::

        > mkdir pkd_project
        > cd pkd_project
        > peekingduck init
        > peekingduck run

    6. ``Todo`` Install PyTorch and TorchVision for torch models::

        ``requires compilation from github source, editing of source files``



Development Version
===================

You can try out the development version of PeekingDuck direct from the
`PeekingDuck Github repository <https://github.com/aimakerspace/PeekingDuck>`_::

    > git clone https://github.com/aimakerspace/PeekingDuck.git

This will clone a copy of PeekingDuck development version into the folder
``PeekingDuck``. |br|
Test it with the following commands::

    > cd PeekingDuck
    > python __main__.py
