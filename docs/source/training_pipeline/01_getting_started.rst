.. include:: /include/substitution.rst

********
Overview
********

==============
 Introduction 
==============

Training a custom model with custom datasets is one of the key tasks when building computer vision solutions.

Peeking Duck allows you to train a custom computer vision model with your own dataset without the need to write the boilerplate codes.

This section will introduce you the key features of the training pipeline and how to get started.
The following sections will introduce you how to train your own model with your own data by customizing the training parameters.

The current version of PeekingDuck supports image classification model training. Object detection are being developed and will be available in future releases.


Features
--------

.. raw:: html

   <h6>Train with Tensorflow or PyTorch</h6>

PeekingDuck supports official pre-trained models from both `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules>`_ and `PyTorch <https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules>`_.

.. raw:: html

   <h6>Train with your own data</h6>

PeekingDuck supports data loading and model training on your own dataset. Refer to :ref:`using_custom_dataset` for more details.

.. raw:: html

   <h6>Customize training parameters</h6>

With PeekingDuck, you can easily configure various training parameters via configuration yaml files or command line arguments. Refer to :ref:`configuring_training_parameters` for more details.

.. raw:: html

   <h6>Analyze right after training</h6>

PeekingDuck uses Weights & Biases to analyze and visualize the training process and performances of the saved models. Refer to the :ref:`setting-up-weights-and-biases` section below for more details.


How the training pipeline works
===============================

PeekingDuck training pipeline is designed with cross-platform compatibility in mind. A high-level overview of the architecture is shown here:

.. image:: /assets/diagrams/C4Diagram-L4_SimplifiedOverview.png

The general workflow when using the training pipeline is:

#. Organize your data with the required format 
#. Place them at the designated directory
#. Personalise your training if necessary 
#. Execute from the terminal

Refer to the next section to get started.


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
         git clone <https://github.com/aisingapore/PeekingDuck.git>
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
         git clone <https://github.com/aisingapore/PeekingDuck.git>
         cd PeekdingDuck

         # Install required packages
         pip install -r peekingduck/training/requirements.txt

         # Refer to the official guide from Tensorflow (https://www.tensorflow.org/install/pip) to install cuda and cudnn

         # Create the bash file to automatically set the PATH variable for cuda when activating the environment
         mkdir -p $CONDA_PREFIX/etc/conda/activate.d
         
         echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/'>\
         $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
         
         # Create the bash file to automatically reset the PATH variable when deactivating the environment
         mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

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
         git clone <https://github.com/aisingapore/PeekingDuck.git>
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
         git clone <https://github.com/aisingapore/PeekingDuck.git>
         cd PeekdingDuck

         # Install required packages
         pip install -r peekingduck/training/requirements.txt

         # Refer to the official guide from Tensorflow (https://www.tensorflow.org/install/pip) to install cuda and cudnn

         # Create the bash file to automatically set the PATH variable for cuda when activating the environment
         mkdir -p $CONDA_PREFIX/etc/conda/activate.d

         echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/'>\
         $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
         
         # Create the bash file to automatically reset the PATH variable when deactivating the environment
         mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

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
            git clone <https://github.com/aisingapore/PeekingDuck.git>
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

Refer to the `official guide <https://docs.wandb.ai/quickstart>`_ for setting up an account.

If you prefer private hosting instead, refer to the `official guide <https://docs.wandb.ai/guides/hosting>`_ to set up private hosting.

Also, if you prefer not to use weights & biases, you can disable it as such:

.. code-block:: bash
   :linenos:
   
   # Disable weights & biases
   wandb disable


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

View the results of each run at the specified output folder directory: :mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`

After running either of the commands without any errors, read on to the next page to see how you can configure the parameters for a more comprehensive training.