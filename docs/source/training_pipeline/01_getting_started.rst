.. include:: /include/substitution.rst

********
Overview
********

==============
 Introduction 
==============

This section will introduce you to the training pipeline, its features, how it works, how you can configure the parameters and how you can train on your own dataset.
PeekingDuck offers this feature for users who intend to perform their own model training.
Training is possible without the need to write boilerplate codes. The training pipeline supports the image classification use case as of PeekingDuck version 2.0.


Features
--------

.. raw:: html

   <h6>TensorFlow and PyTorch Models</h6>
Supports pre-trained models from both PyTorch and TensorFlow. The selection of the model can be customized in the configuration files.

.. raw:: html

   <h6>Train on custom dataset</h6>
Configure the pipeline to train on your own dataset. You can refer to :ref:`using_custom_dataset` for more details.

.. raw:: html

   <h6>Analyze Runs</h6>
The training pipeline uses Weights & Biases for tracking the performance, model versioning and visualizing the results. 
The user has the option to setup an account on Weights & Biases to upload and view the training logs online, or on a local server to view the training logs.  Refer to the :ref:`setting-up-weights-and-biases` section below for more details.

.. raw:: html

   <h6>Configure Training Parameters</h6>
You can configure the training parameters by changing the configuration yaml files or through the command line arguments. Refer to :ref:`configuring_training_parameters` for more details.

.. raw:: html

   <h6>Save the trained model</h6>
The trained models will be saved in the output folder.



How the training pipeline works
===============================

.. raw:: html

   <h6>Overview Diagram</h6>

#. Organize your data with the required format 
#. Place them at the designated directory
#. Personalise your training if necessary 
#. Execute from the terminal

**Component Diagram**

.. image:: /assets/diagrams/C4Diagram-L4_SimplifiedOverview.png



----------

================
 Getting Started
================

This guide explains how you can prepare your environment and test out the default pipeline using either TensorFlow or Pytorch.


Install PeekingDuck Training Pipeline
-------------------------------------

.. tabs::

   .. tab:: Linux

      Plarform:
         - Ubuntu 20.04 LTS / 22.04 LTS
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
         - For CUDA GPU:
            - `Latest Compatible Nvidia Display Driver for your system <https://www.nvidia.com/download/index.aspx>`_
      
      **CPU**

      .. code-block:: bash
         :linenos:

         # Create conda environment & activate
         conda create -n pkd-training python=3.8
         conda activate pkd-training

         # Clone PeekingDuck repository
         git clone <TODO: PeekdingDuck repository>
         cd PeekdingDuck

         # Install required packages
         pip install -r peekingduck/training/requirements.txt

      **CUDA GPU**

      .. code-block:: bash
         :linenos:

         # Create conda environment & activate
         conda create -n pkd-training python=3.8
         conda activate pkd-training

         # Clone PeekingDuck repository
         git clone <TODO: PeekdingDuck repository>
         cd PeekdingDuck

         # Install required packages
         pip install -r peekingduck/training/requirements.txt

         # Install cuda dependencies
         conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
         
         # Initialize cuda
         mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
         && mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d \
         && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/'>\
         $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
         echo 'unset LD_LIBRARY_PATH'>\
         $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

         # Reinitialize environment
         conda activate base && conda activate pkd-training

   .. tab:: Windows

      Platform:
         - Windows 11 / 10 version 2004 or higher
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
         - For CUDA GPU:
            - `Latest Compatible Nvidia Display Driver for your system <https://www.nvidia.com/download/index.aspx>`_
            - `WSL2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_
            - No need to install CUDA Toolkit and cuDNN at this stage. They are bundled with PyTorch, and will be installed via conda for TensorFlow

      **CPU**

      .. code-block:: bash
         :linenos:

         # Create conda environment & activate
         conda create -n pkd-training python=3.8
         conda activate pkd-training

         # Clone PeekingDuck repository
         git clone <TODO: PeekdingDuck repository>
         cd PeekdingDuck

         # Install required packages
         pip install -r peekingduck/training/requirements.txt

      **CUDA GPU**

      .. code-block:: bash
         :linenos:

         # Create conda environment & activate
         conda create -n pkd-training python=3.8
         conda activate pkd-training

         # Clone PeekingDuck repository
         git clone <TODO: PeekdingDuck repository>
         cd PeekdingDuck

         # Install required packages
         pip install -r peekingduck/training/requirements.txt

         # Install cuda dependencies
         conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0

         # Initialize cuda
         mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
         && mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d \
         && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/'>\
         $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
         echo 'unset LD_LIBRARY_PATH'>\
         $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

         # Reinitialize environment
         conda activate base && conda activate pkd-training

   .. tab:: Mac 
    
      Platform:
         - MacOS Intel Chip or MacOS M1/M2 chip
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniforge <https://github.com/conda-forge/miniforge#homebrew>`_

         On macOS, you can install miniforge with Homebrew by running

         .. code-block:: bash
            :linenos:

            brew install miniforge

      Install requirements

         .. code-block:: bash
            :linenos:

            # Clone PeekingDuck repository
            git clone <TODO: PeekdingDuck repository>
            cd PeekdingDuck

            # Create conda environment & activate
            conda env create -f peekingduck/training/requirements_mac.yaml
            conda activate pkd-training

      Install TensorFlow

         .. code-block:: bash
            :linenos:
            
            bash .peekingduck/training/scripts/install_tensorflow_macos.sh

.. _setting-up-weights-and-biases:

Setting up Weights & Biases
---------------------------

Follow the steps below to get configure weights & biases:

1. `Sign up <https://wandb.ai/site>`_ for a free account and then login to your wandb account.
2. (If not yet installed via requirements.txt) Pip install the wandb library on your machine in a Python 3 environment.
3. Login to the wandb library on your machine. You will find your API key `here <https://wandb.ai/authorize>`_

.. code-block:: bash
   :linenos:
   
   pip install wandb

   # Login to weights & biases
   wandb login

Weights & Biases comes out of the box with the training pipeline but in the case where you do not want to use weights & biases you can disable it as such.

.. code-block:: bash
   :linenos:
   
   # Disable weights & biases
   wandb disable

You may refer to the `official guide <https://docs.wandb.ai/quickstart>`_ for setting up an account.


========
Test Run
========

You can test the training pipeline with the default :mod:`cifar10` dataset using the following commands in terminal:

.. code-block:: bash
   :linenos:
   
   # Use the default configurations to test

   cd PeekingDuck

   # Tensorflow
   python ./peekingduck/training/main.py debug=True framework=tensorflow

   # Pytorch
   python ./peekingduck/training/main.py debug=True framework=pytorch

View the results of each run at the specified output folder directory: :mod:`\./Peekingduck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`

After running either of the commands without any errors, read on to the next page to see how you can configure the parameters for a more comprehensive training.