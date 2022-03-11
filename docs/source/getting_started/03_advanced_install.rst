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

        * For macOS Big Sur: |br| |br|

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`conda install -c apple tensorflow-deps=2.6.0` \
            | \ :blue:`[~user]` \ > \ :green:`pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0` \
            | \ :blue:`[~user]` \ > \ :green:`pip install tensorflow-metal==0.2.0` \

    4. Install PyTorch (currently CPU-only):

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`pip install torch torchvision` \

    5. Install PeekingDuck and verify installation:

        .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`pip install peekingduck -\-no-dependencies` \
            | \ :blue:`[~user]` \ > \ :green:`peekingduck -\-verify_install` \



