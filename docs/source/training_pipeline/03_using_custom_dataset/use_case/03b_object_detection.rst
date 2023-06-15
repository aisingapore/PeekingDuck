.. include:: /include/substitution.rst

.. _using_custom_dataset_object_detection:

****************
Object Detection
****************

Introduction
============

If you have already collected your dataset, the following section describes how 
you can prepare your dataset, connect it to the training pipeline and use it 
for training a model.

.. _custom_dataset:

Setup
=====

Formatting Dataset Folder
-------------------------



Coco format

VOC Format

Roboflow References:
- `Convert Pascal VOC XML to COCO JSON <https://roboflow.com/convert/pascal-voc-xml-to-coco-json>`


Setting up configuration
------------------------




Run
===

You can now test the training pipeline with your custom dataset using the 
following commands in the terminal:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`cd path-to-project-folder/PeekingDuck` \

Test for Tensorflow:

.. admonition:: Terminal Session

   | \ :blue:`[~user/PeekingDuck]` \ > \ 
      :green:`python ./peekingduck/training/main.py debug=True framework=tensorflow` \

Test for PyTorch:

.. admonition:: Terminal Session

   | \ :blue:`[~user/PeekingDuck]` \ > \ 
      :green:`python ./peekingduck/training/main.py debug=True framework=pytorch` \


View the results of each run at the specified output folder directory 
:mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`, \
where the default value of the :mod:`<PROJECT_NAME>` is defined in the 
:ref:`config-files-mainconfig`.

You can refer to :ref:`configuring_training_parameters` for more details on how 
to customize your training parameters.