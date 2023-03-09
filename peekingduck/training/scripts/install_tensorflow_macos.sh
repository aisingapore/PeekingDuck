#!/bin/bash
set -e      # tell bash to exit on error

#
# PeekingDuck Installation Script for M1 macOS
# by dotw
# created 2022-03-06 initial version
# updated 2022-04-12 for uninterrupted install
# updated 2022-11-30 add typeguard, pytorch channel
#
# Prerequisites: brew install miniforge
#

echo "PeekingDuck Installation Script for M1 macOS"
echo "--------------------------------------------"

# Global working vars
SELF=`basename $0`
CONDA=`which conda`
MACOS=`sw_vers -productVersion`
ARCHI=`uname -p`
# Defaults
PY_VER=3.8
INSTALL="basic"

# Parse CLI args
while (( $# )); do
    case $1 in
        -n|--name) shift; ENV_NAME=$1
        ;;
        -py) shift; PY_VER=$1
        ;;
        -f|--full) INSTALL="full"
        ;;
    esac
    shift
done

# Detect hardware
if [[ $ARCHI != arm ]]; then
    echo "Hardware $ARCHI unsupported, this script only works for M1 Macs"
    echo "Installation aborted"
    exit 1
else
    echo "Hardware $ARCHI M1 Mac found"
fi

# Detect if conda is already installed
if [ -z "$CONDA" ]; then
    echo "'conda' not found, please install it, installation aborted"
    exit 1
else
    echo "'conda' found at $CONDA"
fi


# Install Tensorflow
if [[ $ARCHI == i386 ]]; then
    echo "Installing for Intel macOS"
    pip install tensorflow==2.7.0
elif [[ $MACOS == 12.* ] || [ $MACOS == 13.* ]]; then
    echo "Installing for macOS Monterey / Ventura"
    conda install -y -c apple tensorflow-deps
    pip install tensorflow-macos tensorflow-metal
elif [[ $MACOS == 11.* ]]; then
    echo "installing for macOS Big Sur"
    conda install -y -c apple tensorflow-deps==2.6.0
    pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0 tensorflow-metal==0.2.0
else
    echo "Unsupported macOS version $MACOS, installation incomplete"
    exit 1
fi
