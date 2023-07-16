.. include:: /include/substitution.rst

********
Overview
********

============
Introduction
============

The PeekingDuck training pipeline allows you to train your own computer vision models
with custom datasets for image classification and object detection tasks, without
writing boilerplate codes.
This section describes the key features of the training pipeline and provides a guide to
get you started.
Subsequent sections describes how to customize the training pipeline parameters to suit
your custom dataset.


Features
========

* Image classification training with TensorFlow or PyTorch
   PeekingDuck supports official pre-trained models from both `TensorFlow 
   <https://www.tensorflow.org/api_docs/python/tf/keras/applications#modules>`_ 
   and `PyTorch <https://pytorch.org/vision/master/models.html#classification>`_.

* Object detection training with PyTorch
   PeekingDuck supports the `YOLOX <https://github.com/Megvii-BaseDetection/YOLOX>`_
   model for object detection training.
   The dataset can be in the `COCO <https://cocodataset.org/>`_ or `VOC
   <http://host.robots.ox.ac.uk/pascal/VOC/>`_ format,

* Custom dataset support
   PeekingDuck supports model training using your own dataset through customizable
   training parameters via YAML files or command line arguments.
   Refer to :ref:`using_custom_dataset` and :ref:`configuring_training_parameters` for
   more details.

* Weights & Biases support
   PeekingDuck supports the (optional) use of `Weights & Biases
   <https://wandb.ai/site>`_ to analyze and visualize the training process and
   performances of the trained models.  Refer to the :ref:`Setting Up Weights and
   Biases` section below for more details.


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

This section explains how you can prepare your virtual environment and test out 
the default training pipeline using either TensorFlow or PyTorch.

It is highly recommended to create a new conda environment as instructed below to avoid
package conflicts. Installing and running the training pipeline in an existing conda
environment can result in unexpected behaviors.


Install PeekingDuck Training Pipeline
=====================================

Take note that the PeekingDuck training pipeline for **object detection** requires 
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
            :green:`pip install -r peekingduck/training/requirements.txt` \

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
            :green:`pip install -r peekingduck/training/requirements.txt` \

      **Install for CUDA GPU**

      1. Complete CPU installation steps 1 and 2 (before :green:`pip install`).

      2. Install PyTorch with the 
         `official PyTorch guide <https://pytorch.org/get-started/locally/>`_.

      3. Install Tensorflow with the
         `official TensorFlow guide <https://www.tensorflow.org/install/pip>`_. 

      4. Complete CPU installation step 3 (:green:`pip install`).

   .. tab:: Mac 
    
      Platform:
         - MacOS Intel Chip or MacOS M1/M2 chip
      Prerequisite:
         - `Git <https://github.com/git-guides/install-git>`_
         - `Miniforge <https://github.com/conda-forge/miniforge#homebrew>`_

         On macOS, you can install MiniForge with Homebrew by running

         .. admonition:: Terminal Session

            | \ :blue:`[~user]` \ > \ :green:`brew install miniforge` \

      **Install requirements**

      1. Clone PeekingDuck repository:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ 
            :green:`git clone https://github.com/aisingapore/PeekingDuck.git` \
         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \
      
      
      2. Create Conda environment and install required packages:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`bash peekingduck/training/scripts/install_training_pipeline_macos.sh` \

|br|

.. _Setting Up Weights and Biases:


Setting Up Weights & Biases
===========================

Follow the steps below to configure Weights & Biases (W&B):

1. `Sign up <https://wandb.ai/site>`_ for a free W&B account and log in.  Refer to the
   `official W&B guide <https://docs.wandb.ai/quickstart>`_ on how to set up an account.
2. If not yet installed via ``requirements.txt``, pip install the W&B package on 
   your machine in a Python 3 environment with the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`pip install wandb` \

3. Log in to W&B on your machine with the API key. You can find your API
   key `here <https://wandb.ai/authorize>`_

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`wandb login` \
   | Logging into wandb.ai.
   | You can find your API key in your browser here: https://wandb.ai/authorize
   | Paste an API key from your profile and hit enter, or press ctrl+c to quit: 
   | Appending key for api.wandb.ai to your netrc file: /Users/.netrc

Refer to the `official W&B quick start guide <https://docs.wandb.ai/quickstart>`_ 
for more details.

If you prefer private hosting instead, refer to the `W&B hosting guide 
<https://docs.wandb.ai/guides/hosting>`_ to set up private hosting.

Also, if you prefer not to use W&B, you can disable it using the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`wandb disabled` \


.. _getting-started-test-run:

Test Run
========

Test the training pipeline with one of the following use cases.

.. tabs::

   .. tab:: Image Classification

      Follow the commands below to train a PyTorch or TensorFlow image classification
      model using the default :mod:`cifar10` dataset:

      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \

      Test for TensorFlow:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py \\` \
            :green:`project_name=my_tensorflow_classification_project \\` |br|
            :green:`debug=True \\` |br|
            :green:`use_case=classification \\` |br|
            :green:`use_case.framework=tensorflow`

      Test for PyTorch:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py \\` \
            :green:`project_name=my_pytorch_classification_project \\` |br|
            :green:`debug=True \\` |br|
            :green:`use_case=classification \\` |br|
            :green:`use_case.framework=pytorch`



   .. tab:: Object Detection

      Follow the commands below to train an object detection model using the default
      :mod:`fashion` dataset:
      The object detection training pipeline supports both COCO and VOC dataset formats.


      .. note::

         Object detection training pipeline requires an Nvidia GPU with CUDA support to run


      .. admonition:: Terminal Session

         | \ :blue:`[~user]` \ > \ :green:`cd PeekingDuck` \

      Test for COCO Format:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py \\` \
            :green:`project_name=my_detection_coco_project \\` |br|
            :green:`use_case=detection \\` |br|
            :green:`data_module.dataset_format=coco \\` |br|
            :green:`data_module/dataset=fashion_coco_format`

      Test for VOC Format:

      .. admonition:: Terminal Session

         | \ :blue:`[~user/PeekingDuck]` \ > \ 
            :green:`python ./peekingduck/training/main.py \\` \
            :green:`project_name=my_detection_voc_project \\` |br|
            :green:`use_case=detection \\` |br|
            :green:`data_module.dataset_format=voc \\` |br|
            :green:`data_module/dataset=fashion_voc_format`


View the result of your training in the specified output folder directory: |br|
``~user/PeekingDuck/outputs/<PROJECT_NAME>/<DATE_TIME>``

Refer to :ref:`configuring_training_parameters` for information on how to customize your
training parameters.