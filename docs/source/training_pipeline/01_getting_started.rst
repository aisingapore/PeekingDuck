.. include:: /include/substitution.rst

***************
Getting Started
***************

==============
 Introduction 
==============

**Low-code model training with custom dataset**

You can use your own dataset to train an image classification or object detection model without writing the boilerplate codes. Simply organize your data with the required format, place them at the designated directory, execute the terminal commands, and the training pipeline will take care of the rest.


**Target User**



Features
--------


**Custom dataset**

With the right data formats, the user can use their own dataset for various applications.


**Use Popular pre-trained models from TensorFlow and PyTorch**

Peekingduck supports popular pre-trained models for both PyTorch and TensorFlow for image classification, and PyTorch for object detection. The selection of the model can be easily customized in the configuration files.


**Analyze Runs**

PeekingDuck uses Weights & Biases for training tracking, model versioning and result visualization. The user is required to setup an account on Weights & Biases to upload and view the training logs online, or hosting a local server to view the training logs locally.  Refer to the setup guide for more details.


**Configure Training Parameters**

PeekingDuck allows you to configure your training parameters easily with configuration yaml files or command line arguments.


**Save the trained model**

You can save the model trained with PeekingDuck at the designated location for your own usage



How training pipeline works
===========================


**Overview Diagram**


**Sequence Diagram (Training Loop)**



----------

=====================================
Install PeekingDuck Training Pipeline
=====================================


OS
--

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

         $ conda create -n pkd python=3.8
         $ conda activate pkd
         $ git clone <PeekdingDuck repository>
         $ cd PeekdingDuck
         $ pip install -r peekingduck/training/requirements.txt

      **CUDA GPU**

      .. code-block:: bash

         $ conda create -n pkd python=3.8
         $ conda activate pkd
         $ git clone <PeekdingDuck repository>
         $ cd PeekdingDuck
         $ pip install -r peekingduck/training/requirements.txt
         $ conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
         $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
         && mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d \
         && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/'>\
         $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
         echo 'unset LD_LIBRARY_PATH'>\
         $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
         $ conda activate base && conda activate pkd

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

         $ conda create -n pkd python=3.8
         $ conda activate pkd
         $ git clone <PeekdingDuck repository>
         $ cd PeekdingDuck
         $ pip install -r peekingduck/training/requirements.txt

      **CUDA GPU**

      .. code-block:: bash

         $ conda create -n pkd python=3.8
         $ conda activate pkd
         $ git clone <PeekdingDuck repository>
         $ cd PeekdingDuck
         $ pip install -r peekingduck/training/requirements.txt
         $ conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
         $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d \
         && mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d \
         && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/'>\
         $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
         echo 'unset LD_LIBRARY_PATH'>\
         $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
         $ conda activate base && conda activate pkd

   .. tab:: Mac 
    
      **Intel Chip**

      **M1/M2 Chip**



Setting up Weights & Biases
---------------------------

We recommend using the cloud host from Weights and Biases as it's easier to get started. You may refer to the `official guide <https://docs.wandb.ai/quickstart>`_ for setting up an account.

In a nutshell, follow the 3 steps below:

1. `Sign up <https://wandb.ai/site>`_ for a free account and then login to your wandb account.
2. (If not yet installed via requirements.txt) Pip install the wandb library on your machine in a Python 3 environment.
4. Login to the wandb library on your machine. You will find your API key `here <https://wandb.ai/authorize>`_

----------

========================
Test Run
========================

To test the training pipeline with the default cifar10 dataset, use the following commands in terminal:

.. code-block:: bash
   
   # use the default configurations to test
   $ cd PeekdingDuck
   $ python ./peekingduck/training/main.py debug=True framework=tensorflow
   $ python ./peekingduck/training/main.py debug=True framework=pytorch
