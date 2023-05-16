.. include:: /include/substitution.rst

********
Overview
********

==============
 Introduction 
==============

Training custom computer vision models with custom datasets is a crucial aspect 
of building computer vision solutions. Peeking Duck simplifies this process by 
eliminating the need for writing boilerplate code. In this section, we will 
explore the key features of the training pipeline and provide a guide on getting 
started. Subsequent sections will demonstrate how to train your own model using 
your specific data by customizing the training parameters.

Please note that the current version of PeekingDuck focuses on image 
classification model training. While object detection capabilities are under 
development, they will be included in future releases.


Features
--------

.. raw:: html

   <h6>Train with TensorFlow or PyTorch</h6>

PeekingDuck supports official pre-trained models from both 
`TensorFlow 
<https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules>`_ 
and `PyTorch 
<https://pytorch.org/vision/stable/models.html#classification>`_.

.. raw:: html

   <h6>Train with your own data</h6>

PeekingDuck supports data loading and model training on your own dataset. 
Refer to :ref:`using_custom_dataset` for more details.

.. raw:: html

   <h6>Customize training parameters</h6>

With PeekingDuck, you can easily configure various training parameters via 
configuration yaml files or command line arguments. 
Refer to :ref:`configuring_training_parameters` for more details.

.. raw:: html

   <h6>Analyze right after training</h6>

PeekingDuck uses Weights & Biases to analyze and visualize the training process 
and performances of the saved models. 
Refer to the :ref:`setting-up-weights-and-biases` section below for more
details.


How the training pipeline works
===============================

PeekingDuck training pipeline is designed with cross-platform compatibility in 
mind. A high-level overview of the architecture is shown here:

.. image:: /assets/diagrams/C4Diagram-L4_SimplifiedOverview.png

The general workflow when using the training pipeline is:

#. Organize your data with the required format 
#. Place them at the designated directory
#. Personalize your training if necessary
#. Execute from the terminal

Refer to the next section to get started.

----------

================
 Getting Started
================

This guide explains how you can prepare your environment and test out the 
default pipeline using either TensorFlow or Pytorch.

It is highly recommended to create a new conda environment as instructed below. 
Installing PeekingDuck dependencies into existing conda environments may cause 
unexpected behaviors.

Install PeekingDuck Training Pipeline
-------------------------------------

.. tabs::

   .. tab:: Linux

      Platform:
         - Ubuntu 20.04 LTS / 22.04 LTS.
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
         - For CUDA GPU:
            - `Latest Compatible Nvidia Display Driver for your
              system <https://www.nvidia.com/download/index.aspx>`_
            - Do not install CUDA Toolkit and cuDNN at this stage. 
              They are bundled with PyTorch when installed by pip, 
              and will be installed via conda for TensorFlow.
      
      **Install for CPU**

      1. Create and activate the conda environment:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ 
            :green:`conda create -n pkd-training python=3.8` \
         | \ :blue:`[~user]` \ > \ :green:`conda activate pkd-training` \
         

      2. Clone PeekingDuck repository:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ 
            :green:`git clone https://github.com/aisingapore/PeekingDuck.git` \
         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \


      3. Install required packages:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`pip install -r peekingduck/training/requirements.txt` \

      **Install for CUDA GPU**

      1. Complete CPU installation steps.

      2. Refer to the `official Tensorflow step-by-step instructions 
         <https://www.tensorflow.org/install/pip#step-by-step_instructions>`_ 
         to install CUDA and CuDNN for TensorFlow. 
         For Ubuntu 22.04, do take note of the additional steps.

   .. tab:: Windows

      Platform:
         - Windows 11 / 10 version 2004 or higher.
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
         - For CUDA GPU:
            - `Latest Compatible Nvidia Display Driver for your system 
              <https://www.nvidia.com/download/index.aspx>`_
            - `WSL2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_
            - No need to install CUDA Toolkit and cuDNN at this stage. They are 
              bundled with PyTorch when installed by pip, and will be installed 
              via conda for TensorFlow.

      **Install for CPU**

      1. Create and activate the conda environment:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ 
            :green:`conda create -n pkd-training python=3.8` \
         | \ :blue:`[~user]` \ > \ 
            :green:`conda activate pkd-training` \
         

      2. Clone PeekingDuck repository:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ 
            :green:`git clone https://github.com/aisingapore/PeekingDuck.git` \
         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \


      3. Install required packages:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`pip install -r peekingduck/training/requirements.txt` \

      **Install for CUDA GPU**

      1. Complete CPU installation steps.

      2. Refer to the `official Tensorflow step-by-step instructions 
         <https://www.tensorflow.org/install/pip#step-by-step_instructions>`_ 
         to install CUDA and CuDNN for TensorFlow.

   .. tab:: Mac 
    
      Platform:
         - MacOS Intel Chip or MacOS M1/M2 chip
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniforge <https://github.com/conda-forge/miniforge#homebrew>`_

         On macOS, you can install miniforge with Homebrew by running

         .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`brew install miniforge` \

      **Install requirements**

      1. Clone PeekingDuck repository:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ 
            :green:`git clone https://github.com/aisingapore/PeekingDuck.git` \
         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \
      
      2. Create conda environment & activate:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ :green:
            `conda env create -f peekingduck/training/requirements_mac.yaml` \
         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`conda activate pkd-training` \

      3. Install TensorFlow

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ :green:
            `bash peekingduck/training/scripts/install_tensorflow_macos.sh` \

.. _setting-up-weights-and-biases:

Setting Up Weights & Biases
---------------------------

Follow the steps below to get configure weights & biases:

1. `Sign up <https://wandb.ai/site>`_ for a free account and then login to your 
   wandb account. 
   Refer to the `wandb official guide <https://docs.wandb.ai/quickstart>`_ for 
   setting up an account.
2. (If not yet installed via requirements.txt) Pip install the wandb library on 
   your machine in a Python 3 environment.

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`pip install wandb` \

3. Login to the wandb library on your machine with the API key. You will find 
   your API key `here <https://wandb.ai/authorize>`_

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`wandb login` \

Refer to the `wandb quick start guide <https://docs.wandb.ai/quickstart>`_ 
for more details.

If you prefer private hosting instead, refer to the `hosting guide 
<https://docs.wandb.ai/guides/hosting>`_ to set up private hosting.

Also, if you prefer not to use Weights & Biases, you can disable it as such:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`wandb disable` \

.. _getting-started-test-run:

========
Test Run
========

You can test the training pipeline with the default :mod:`cifar10` dataset using
 the following commands in terminal:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \

Test for Tensorflow:

.. admonition:: Terminal Session

   | \ :blue:`[~user/PeekingDuck]` \ > \ :green:
   `python ./peekingduck/training/main.py debug=True framework=tensorflow` \

Test for PyTorch:

.. admonition:: Terminal Session

   | \ :blue:`[~user/PeekingDuck]` \ > \ :green:
   `python ./peekingduck/training/main.py debug=True framework=pytorch` \


View the results of each run at the specified output folder directory 
:mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`, 
\ where the default value of the :mod:`<PROJECT_NAME>` is defined in the 
:ref:`config-files-mainconfig`.

After installation and test run, refer to :ref:`configuring_training_parameters`
 for training customizations.