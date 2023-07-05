.. include:: /include/substitution.rst

.. _using_custom_dataset_object_detection:

****************
Object Detection
****************

============
Introduction
============

If you have already collected your dataset, the following section describes how 
you can prepare your dataset, connect it to the training pipeline and use it 
for training a model.

.. _custom_dataset:

=====
Setup
=====

Formatting Dataset Folder
=========================

To work with the PeekingDuck object detection training pipeline, the training dataset needs to 
be arranged in either of the formats below:

COCO format
-----------
The `COCO (Common Objects in Context) <https://cocodataset.org/#format-data>`_ 
format is a commonly used and popular data format for object detection and segmentation.
This is how the folder structure would look like:

.. parsed-literal::

   \ :blue:`PeekingDuck/data/` \ |Blank|
      └── \ :blue:`<your_dataset_folder>/` \ |Blank|
               ├── \ :blue:`annotations/` \ |Blank|
               │      ├── instances_train2017.json
               │      └── instances_val2017.json
               ├── \ :blue:`train2017/` \ |Blank|
               │      ├── <train_image_001>.jpg
               │      ├── <train_image_002>.jpg
               │      ├── <train_image_003>.jpg
               │      └── ...
               └── \ :blue:`val2017/` \ |Blank|
                      ├── <val_image_001>.jpg
                      ├── <val_image_002>.jpg
                      ├── <val_image_003>.jpg
                      └── ...

Distribute your images into the training (train2017) and validation (val2017) folders, 
and prepare the train and validation JSON annotation files according to the `COCO (Common Objects in Context) format <https://cocodataset.org/#format-data>`_.

VOC Format
----------
Alternatively, PeekingDuck supports the `Pascal VOC(Visual Object Classes) <https://mlhive.com/2022/02/read-and-write-pascal-voc-xml-annotations-in-python>`_ format.
Unlike COCO where there is a single JSON file for each train and validation folder, 
this format requires each image to have its own XML annotation file.
This is how the folder structure would look like:

.. parsed-literal::

   \ :blue:`PeekingDuck/data/` \ |Blank|
      └── \ :blue:`<your_dataset_folder>/` \ |Blank|
               └── \ :blue:`VOC2007/` \ |Blank|
                        ├── \ :blue:`Annotations/` \ |Blank|
                        │      ├── <image_001>.xml
                        │      ├── <image_002>.xml
                        │      ├── <image_003>.xml
                        │      └── ...
                        ├── \ :blue:`ImageSets/` \ |Blank|
                        │      └── \ :blue:`Main/` \ |Blank|
                        │             ├── test.txt
                        │             └── trainval.txt
                        └── \ :blue:`JPEGImages/` \ |Blank|
                               ├── <image_001>.jpg
                               ├── <image_002>.jpg
                               ├── <image_003>.jpg
                               └── ...

You can use this tool to assist in converting your dataset to the above format:
`Convert Pascal VOC XML to COCO JSON <https://roboflow.com/convert/pascal-voc-xml-to-coco-json>`_



Dataset Classes Global Variable
-------------------------------

After preparing the dataset folder, you will need to prepare the dataset classes' global variable. 
Depending on the dataset format you are using, you will need to edit the following files:

.. note::

   The classes listed in the following sets have to follow the ID order in your
   dataset annotation file.

For COCO Format locate this file : :mod:`./peekingduck/training/src/model/yolox/data/datasets/coco_classes.py`
and edit the classes according to your dataset.

.. code-block:: bash
   :linenos:

      COCO_CLASSES = (
         "fashion",
         "bag",
         "dress",
         "hat",
         "jacket",
         "pants",
         "shirt",
         "shoe",
         "shorts",
         "skirt",
         "sunglass",
      )

For VOC Format locate this file : :mod:`./peekingduck/training/src/model/yolox/data/datasets/voc_classes.py`
and edit the classes according to your dataset.

.. code-block:: bash
   :linenos:

      VOC_CLASSES = (
         "sunglass",
         "hat",
         "jacket",
         "shirt",
         "pants",
         "shorts",
         "skirt",
         "dress",
         "bag",
         "shoe",
      )



.. _03b-config-folder-structure:

Configuration Folder Structure
------------------------------

Below shows the folder structure and the files related to object detection configuration.

.. parsed-literal::

    \ :blue:`peekingduck/training/configs/` \ |Blank|
          ├── \ :blue:`data_module/` \ |Blank|
          │      ├── \ :blue:`dataset/` \ |Blank|
          │      │      ├── fashion_coco_format.yaml
          │      │      └── fashion_voc_format.yaml
          │      └── detection.yaml
          ├── \ :blue:`model/` \ |Blank|
          │      └── detection.yaml
          ├── \ :blue:`model_analysis/` \ |Blank|
          │      └── detection.yaml
          ├── \ :blue:`stores/` \ |Blank|
          │      └── detection.yaml
          ├── \ :blue:`trainer/` \ |Blank|
          │      └── detection.yaml
          └── config.yaml


Setting up configuration
------------------------

| After preparing your data folder, you will need to create and edit the 
  configuration files to connect your dataset to the training pipeline.

| For a better understanding of which configuration files to change, you can 
  refer to the directory tree at :ref:`03b-config-folder-structure`.


1. Create a Dataset File

| Create a YAML file under the :mod:`data_module/dataset` folder directory.

.. parsed-literal::

   \ :blue:`PeekingDuck/peekingduck/training/configs/data_module/` \ |Blank|
            ├── \ :blue:`data_adapter/...` \ |Blank|
            ├── \ :blue:`dataset/` \ |Blank|
            │      └── <dataset_filename>.yaml
            ├── \ :blue:`resample/...` \ |Blank|
            └── \ :blue:`transform/...` \ |Blank|

| Add this code snippet and change the values where necessary:
| :mod:`.peekingduck/training/configs/data_module/dataset/<dataset_filename>.yaml`

.. code-block:: bash
   :linenos:

   download          : False
   url               : ""
   blob_file         : ""
   root_dir          : "data"
   data_dir          : "${.root_dir}/<folder-name-of-your-dataset>" # change this - your dataset folder name
   num_classes       : 11 # change this - number of labels in your dataset

   # FOR COCO FORMAT
   train_ann         : "instances_train2017.json" # change this - name of your train JSON annotation file + .json extension
   val_ann           : "instances_val2017.json" # change this - name of your val JSON annotation file + .json extension
   test_ann          : null
   image_sets        : null

   # FOR VOC FORMAT
   train_ann         : null
   val_ann           : null
   test_ann          : null
   image_sets        : [['2007', 'trainval']] # change this - year and image set name


2. Edit Data Module File

| Edit the YAML file under the :mod:`configs/data_module` folder directory.

.. parsed-literal::

   \ :blue:`PeekingDuck/peekingduck/training/configs/data_module/` \ |Blank|
            ├── \ :blue:`data_adapter/...` \ |Blank|
            ├── \ :blue:`dataset/...` \ |Blank|
            ├── \ :blue:`resample/...` \ |Blank|
            ├── \ :blue:`transform/...` \ |Blank|
            └── detection.yaml

| Change the values where necessary:
| :mod:`.peekingduck/training/configs/data_module/detection.yaml`

.. code-block:: bash
   :linenos:

   dataset_format    : "coco" # 'coco' | 'voc' # change this value depending on your dataset format
   defaults:
         - dataset: fashion_coco_format # Change this value to the file name from the previous step

3. Edit Main Config File

| Edit the project name and use_case parameter in :mod:`.peekingduck/training/configs/config.yaml` file.

.. code-block:: bash
   :linenos:

    project_name: "<my_project_name>" # Change this value to your project name

    defaults:
    - use_case: detection # < classification | detection >  # Ensure this value is set to detection

    # Do NOT change the following
    - data_module: ${use_case}
    - model: ${use_case}
    - trainer: ${use_case}
    - callbacks: ${use_case}
    - metrics: ${use_case}
    - model_analysis: ${use_case}
    - stores: ${use_case}
    - _self_
    - override hydra/job_logging: custom

4. (Optional)  Edit Trainer Config File

| Additionally you can customise your training by changing some of the values
 in :mod:`.peekingduck/training/configs/trainer/detection.yaml` file.

.. code-block:: bash
   :linenos:

      yolox:
         experiment_name: ${project_name}
         name: null
         resume: False 
         fp16: True # Turn mix precision training on/off
         occupy: null
         cache: null
         seed : ${use_case.random_state}

         # Change the number of GPUs and batch size (Recommended batch size: 8*GPUs)
         devices: 1    
         batch_size: 8

         # Change the number of epochs
         max_epoch: 10

         # Select the model to train
         model: "yolox_s" # "yolox_s" | "yolox_m" | "yolox_l" | "yolox_x" | "yolox_nano" | "yolox_tiny"

         # Add pretrained weights
         ckpt: null

         # Change the model analysis tool
         logger: "tensorboard" # "tensorboard" | "wandb"

         # Train on more than 1 machine
         dist_url: null
         num_machines: 1
         machine_rank: null
         dist_backend: null

         # Change the output directory
         output_dir: './outputs'


Run
===

Assuming you have followed through the above steps, you can now test the 
object detection training pipeline with your custom dataset using the 
following commands in the terminal:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`cd path-to-project-folder/PeekingDuck` \
   | \ :blue:`[~user/PeekingDuck]` \ > \ 
      :green:`python ./peekingduck/training/main.py` \


View the result of your training in the specified output folder directory: 
:mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`.

You can refer to this page :ref:`configuring_training_parameters_detection` for more details on how 
to customize your training parameters.