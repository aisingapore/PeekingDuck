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

      **CUDA GPU**

      **CPU**
        +------------------------------+
        | paragraph                    |
        |                              |
        +------------------------------+

   .. tab:: Windows

      **CUDA GPU**

      **CPU**
        


   .. tab:: Mac 
    
      **Intel Chip**

      **M1/M2 Chip**



Setting up Weights & Biases
---------------------------


----------

========================
Verify Training Pipeline
========================


PyTorch
-------

[what value the user should get]



TensorFlow
----------

[what value the user should get]