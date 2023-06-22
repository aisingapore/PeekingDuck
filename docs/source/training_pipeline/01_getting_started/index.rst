.. include:: /include/substitution.rst

********
Overview
********

==============
Introduction
==============

Training custom computer vision models with custom datasets is a crucial aspect 
of building computer vision solutions. PeekingDuck simplifies this process by 
eliminating the need for writing boilerplate code. In this section, we will 
explore the key features of the training pipeline and provide a guide to get 
started. Subsequent sections will demonstrate how to train your own model using 
your specific data by customizing the training parameters.

Please note that the current version of PeekingDuck focuses on image 
classification and object detection model training only.


Features
========

* Image classification training with TensorFlow or PyTorch
   PeekingDuck supports official pre-trained models from both `TensorFlow 
   <https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules>`_ 
   and `PyTorch 
   <https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules>`_.

* Object Detection training with VOC or COCO format
   PeekingDuck supports `YOLOX <https://github.com/Megvii-BaseDetection/YOLOX>`_
   object detection training. Easily customize parameters using YAML configuration files.

* Train with your data
   PeekingDuck supports data loading and model training on your dataset. 
   Refer to :ref:`using_custom_dataset` for more details.

* Customize training parameters
   With PeekingDuck, you can easily configure various training parameters via 
   configuration YAML files or command line arguments. 
   Refer to :ref:`configuring_training_parameters` for more details.

* Analyze model performance after training
   PeekingDuck uses `Weights & Biases <https://wandb.ai/site>`_ to analyze and 
   visualize the training process and performances of the saved models.
   Refer to the :ref:`Setting Up Weights and Biases` section below for more
   details.


How the Training Pipeline Works
===============================

PeekingDuck's training pipeline is designed to be compatible with PyTorch and 
TensorFlow. The architecture diagram is shown below:

.. image:: /assets/diagrams/training_pipeline_c4_classification_detection_overview_diagram.png

The general workflow when using the training pipeline is:

#. Organize the training dataset with the required format in the designated 
   directory.
#. Customize training parameters, if necessary.
#. Start model training from the terminal.

----------

===============
Getting Started
===============

This guide explains how you can prepare your virtual environment and test out 
the default training pipeline using either TensorFlow or PyTorch.

It is highly recommended to create a new conda environment as instructed below. 
This is to avoid package conflicts and unexpected behaviors when installing the 
training pipeline into and running from an existing conda environment.


Install PeekingDuck Training Pipeline
=====================================

Take note that PeekingDuck training pipeline for **object detection** requires 
**CUDA GPU** to perform the training.

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
              They are bundled with PyTorch when installed by pip 
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
            :green:`pip install -r peekingduck/training/requirements_linux.txt` \

      **Install for CUDA GPU**

      1. Complete CPU installation steps.

      2. Refer to the `official TensorFlow step-by-step instructions 
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
              bundled with PyTorch when installed by pip and will be installed 
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
            :green:`pip install -r peekingduck/training/requirements_windows.txt` \

      **Install for CUDA GPU**

      1. Complete CPU installation step 1 and 2 (before :green:`pip install`).

      2. Install PyTorch with the 
         `official guide <https://pytorch.org/get-started/locally/>`_

      3. Complete CPU installation step 3 (:green:`pip install`)

      4. Refer to the `official Tensorflow step-by-step instructions 
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
      
      
      2. Create conda environment and install required packages:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`bash peekingduck/training/scripts/install_training_pipeline_macos.sh` \


.. _Setting Up Weights and Biases:

Setting Up Weights & Biases
===========================

Follow the steps below to configure weights & biases:

1. `Sign up <https://wandb.ai/site>`_ for a free account and then login to your 
   wandb account. 
   Refer to the `wandb official guide <https://docs.wandb.ai/quickstart>`_ for 
   setting up an account.
2. (If not yet installed via requirements.txt) Pip install the wandb library on 
   your machine in a Python 3 environment with the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`pip install wandb` \

3. Login to the wandb library on your machine with the API key. You can find 
   your API key `here <https://wandb.ai/authorize>`_

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`wandb login` \
   | Logging into wandb.ai.
   | You can find your API key in your browser here: https://wandb.ai/authorize
   | Paste an API key from your profile and hit enter, or press ctrl+c to quit: 
   | Appending key for api.wandb.ai to your netrc file: /Users/.netrc

Refer to the `wandb quick start guide <https://docs.wandb.ai/quickstart>`_ 
for more details.

If you prefer private hosting instead, refer to the `hosting guide 
<https://docs.wandb.ai/guides/hosting>`_ to set up private hosting.

Also, if you prefer not to use Weights & Biases, you can disable it using the 
following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`wandb disable` \


.. _getting-started-test-run:

Test Run
========

Select a use case and follow the steps to test the training pipeline.

.. tabs::

   .. tab:: Image Classification

      Follow the commands below to train a PyTorch or TensorFlow image
      classification training pipeline using the default :mod:`cifar10` dataset:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \

      Test for TensorFlow:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py project_name=my_tensorflow_classification_project debug=True use_case=classification use_case.framework=tensorflow` \

      Test for PyTorch:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py project_name=my_pytorch_classification_project debug=True use_case=classification use_case.framework=pytorch` \


      View the result of each run at the specified output folder directory: 
      :mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`. 
      The default value of :mod:`<PROJECT_NAME>` is defined in the 
      :ref:`config-files-mainconfig`.


   .. tab:: Object Detection

      The object detection training pipeline supports both COCO and VOC format.
      Follow the commands below to train an object detection training pipeline 
      using the default :mod:`fashion` dataset:


      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \

      Test for COCO Format:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py project_name=my_detection_coco_project use_case=detection data_module.detection.dataset_format=coco data_module.dataset=fashion_coco_format` \

      Test for VOC Format:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py project_name=my_detection_voc_project use_case=detection data_module.detection.dataset_format=voc data_module.dataset=fashion_voc_format` \


      View the result of each run at the specified output folder directory: 
      :mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`. 
      The default value of :mod:`<PROJECT_NAME>` is defined in the 
      :ref:`config-files-mainconfig`.

After installation and test runs, refer to 
:ref:`configuring_training_parameters` for training customization.
