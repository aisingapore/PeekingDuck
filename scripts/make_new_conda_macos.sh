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

# Check required argument: environment name
if [ -z "$ENV_NAME" ]; then
    echo "Usage:"
    echo "  $SELF -n <env_name> [-py <python_ver>] [-f]"
    echo ""
    echo "  -n | --name  environment name (required)"
    echo "  -py          python version, default=3.8"
    echo "  -f | -full   default install default=basic, full includes CI/CD packages"
    echo ""
    echo "Create new conda environment and install PeekingDuck"
    exit 1
fi

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

# Detect if conda env exists
if { conda env list | grep ".*$ENV_NAME.*"; } >/dev/null 2>&1; then
    echo "$ENV_NAME already exists, please choose a new name"
    exit 1
fi

echo "Make new conda env=$ENV_NAME python=$PY_VER install=$INSTALL archi=$ARCHI macOS=$MACOS"

conda create -y -n $ENV_NAME python=$PY_VER
conda init bash
source ~/.bash_profile
conda activate $ENV_NAME
echo "conda env=$CONDA_DEFAULT_ENV"

# Install basic packages
echo "Installing basic packages"
conda install -y black click colorama opencv pyyaml requests scipy shapely tqdm typeguard

# Install PyTorch
echo "installing PyTorch"
conda install -y -c pytorch pytorch torchvision timm

# Install Tensorflow
if [[ $ARCHI == i386 ]]; then
    echo "Installing for Intel macOS"
    pip install tensorflow==2.7.0
elif [[ $MACOS == 12.* ]]; then
    echo "Installing for macOS Monterey"
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

# Install more packages
if [ $INSTALL == "full" ]; then
    echo "Installing CI/CD packages"
    conda install -y bandit coverage mypy pylint pytest types-PyYAML types-requests \
        types-setuptools types-six beautifulsoup4 myst-parser sphinx-autodoc-typehints \
        texttable
    pip install lap sphinx-rtd-theme
fi

# Install PeekingDuck
echo "Installing PeekingDuck"
pip install peekingduck --no-dependencies

echo "PeekingDuck installation done"
echo "-----------------------------"
echo "Activate new environment with: conda activate $ENV_NAME"
echo "Exiting $SELF"



