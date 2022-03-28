#!/bin/bash
set -e              # tell bash to exit on error

#
# PeekingDuck Installation Script for M1 macOS Monterey/Big Sur
# by dotw 2021-03-06
#

echo "PeekingDuck Installation Script for M1 macOS Monterey/Big Sur"
echo "-------------------------------------------------------------"

# Global working vars
ARCHI=`uname -p`
CONDA=`which conda`
MACOS=`sw_vers -productVersion`

# Detect hardware
if [[ $ARCHI != arm ]]; then
    echo "hardware $ARCHI unsupported, this script only works for M1 Macs"
    echo "installation aborted"
    exit 1
else
    echo "hardware $ARCHI M1 Mac found"
fi

# Detect macOS
if [[ $MACOS != 12.* && $MACOS != 11.* ]]; then
    echo "macOS $MACOS unsupported (requires 11.* or 12.*) installation aborted"
    exit 1
else
    if [[ $MACOS == 12.* ]]; then
        echo "installing for macOS Monterey $MACOS"
    elif [[ $MACOS == 11.* ]]; then
        echo "installing for macOS Big Sur $MACOS"
    else
        echo "macOS $MACOS unknown, abort installation"
        exit 1
    fi
fi

# Detect conda
if [ -z "$CONDA" ]; then
    echo "conda not found, abort installation"
    exit 1
else
    echo "conda found at $CONDA"
fi

# Detect current conda env
if [ "base" == $CONDA_DEFAULT_ENV ]; then
    echo "cannot install into conda base env, please use a different env"
    echo "installation aborted"
    exit 1
else
    echo "installing into conda env $CONDA_DEFAULT_ENV"
fi

# Install conda packages
conda install -y click colorama opencv openblas pyyaml requests scipy shapely tqdm

# Install pip packages: Tensorflow and PyTorch
if [[ $MACOS == 12.* ]]; then
    conda install -y -c apple tensorflow-deps==2.7.0
    pip install tensorflow-estimator==2.7.0 tensorflow-macos==2.7.0 tensorflow-metal==0.3.0
elif [[ $MACOS == 11.* ]]; then
    conda install -y -c apple tensorflow-deps==2.6.0
    pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0 tensorflow-metal==0.2.0
else
    echo "macOS $MACOS unknown, abort installation"
    exit 1
fi
pip install torch torchvision

# Install PeekingDuck
pip install peekingduck --no-dependencies

echo "PeekingDuck installation done"

