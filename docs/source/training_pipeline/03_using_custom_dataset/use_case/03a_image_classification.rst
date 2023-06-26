.. include:: /include/substitution.rst

.. _using_custom_dataset_image_classification:

********************
Image Classification
********************

Introduction
============

If you have already collected your dataset, the following section describes how 
you can prepare your dataset, connect it to the training pipeline and use it 
for training a model.

.. _custom_dataset:

Setup
=====

Data Folder Structure   
---------------------

To work with the PeekingDuck image classification training pipeline,
the training dataset needs to be arranged in the folder structure below:

.. parsed-literal::

   \ :blue:`PeekingDuck/data/` \ |Blank|
      ├── \ :blue:`cifar10/...` \ |Blank|
      ├── \ :blue:`rsna/...` \ |Blank|
      ├── \ :blue:`vegfru/...` \ |Blank|
      └── \ :blue:`<your_dataset_folder>/` \ |Blank|
               ├── \ :blue:`<class_1>/` \ |Blank|
               │      ├── <image_001>.jpg
               │      ├── <image_002>.jpg
               │      ├── <image_003>.jpg
               │      └── ...
               ├── \ :blue:`<class_2>/` \ |Blank|
               │      ├── <image_001>.jpg
               │      ├── <image_002>.jpg
               │      ├── <image_003>.jpg
               │      └── ...
               ├── \ :blue:`<class_3>/` \ |Blank|
               │      ├── <image_001>.jpg
               │      ├── <image_002>.jpg
               │      ├── <image_003>.jpg
               │      └── ...
               ├── \ :blue:`.../` \ |Blank|
               │      ├── <...>.jpg
               │      ├── <...>.jpg
               │      ├── <...>.jpg               
               │      └── ...
               └── <image_to_label_map>.csv


Image to Label Mapping
----------------------

| The table below describes the columns needed to create the CSV file which 
  will be used by the training pipeline to map your images to their labels.
| `image_path`, `class_id` and  `class_name` are required columns in the `CSV` 
  file.

+-------------------+------------------+---------------------------+
| Column            | Data type        | Description               |
+===================+==================+===========================+
| `image_path`      | `string`         | path to the image file    |
+-------------------+------------------+---------------------------+
| `class_id`        | `integer`        | image class               |
+-------------------+------------------+---------------------------+
| `class_name`      | `string`         | image class name          |
+-------------------+------------------+---------------------------+

The CSV file should be in the format below:

+----------------------------------------------------+--------------+------------+
| image_path                                         | class_id     | class_name |
+====================================================+==============+============+
| data/<your_dataset_folder>/<class_1>/<image_1>.jpg | <class_id_1> | <class_1>  |
+----------------------------------------------------+--------------+------------+
| data/<your_dataset_folder>/<class_1>/<image_2>.jpg | <class_id_1> | <class_1>  |
+----------------------------------------------------+--------------+------------+
| …                                                  | …            | …          |
+----------------------------------------------------+--------------+------------+
| data/<your_dataset_folder>/<class_2>/<image_1>.jpg | <class_id_2> | <class_2>  |
+----------------------------------------------------+--------------+------------+
| data/<your_dataset_folder>/<class_2>/<image_2>.jpg | <class_id_2> | <class_2>  |
+----------------------------------------------------+--------------+------------+
| …                                                  | …            | …          |
+----------------------------------------------------+--------------+------------+
| data/<your_dataset_folder>/<class_m>/<image_1>.jpg | <class_id_m> | <class_m>  |
+----------------------------------------------------+--------------+------------+
| …                                                  | …            | …          |
+----------------------------------------------------+--------------+------------+
| data/<your_dataset_folder>/<class_m>/<image_n>.jpg | <class_id_m> | <class_m>  |
+----------------------------------------------------+--------------+------------+

Below is a snippet example of the CSV file using used in the `vegfru <https://
github.com/ustc-vim/vegfru>`_ dataset:

+------------------------------------------------+----------+------------------------------+
| image_path                                     | class_id | class_name                   |
+================================================+==========+==============================+
| data/vegfru/Chinese_artichoke/v_14_01_0001.jpg | 13       | Chinese_artichoke            |
+------------------------------------------------+----------+------------------------------+
| data/vegfru/Chinese_artichoke/v_06_03_0007.jpg | 3        | Chinese_kale                 |
+------------------------------------------------+----------+------------------------------+
| data/vegfru/Chinese_artichoke/v_09_03_0031.jpg | 8        | Chinese_pumpkin              |
+------------------------------------------------+----------+------------------------------+


.. _03a-config-folder-structure:

Configuration Folder Structure
------------------------------

Below shows the folder structure and the files related to image classification configuration.

.. parsed-literal::

    \ :blue:`peekingduck/training/configs/` \ |Blank|
          ├── \ :blue:`callbacks/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`data_module/` \ |Blank|
          │      ├── \ :blue:`data_adapter/` \ |Blank|
          │      │      └── adapter.yaml
          │      ├── \ :blue:`dataset/` \ |Blank|
          │      │      ├── cifar10.yaml
          │      │      ├── rsna.yaml
          │      │      └── vegfru.yaml
          │      ├── \ :blue:`resample/` \ |Blank|
          │      │      └── train_test_split.yaml
          │      ├── \ :blue:`transform/` \ |Blank|
          │      │      ├── test.yaml
          │      │      └── train.yaml
          │      └── classification.yaml
          ├── \ :blue:`metrics/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`model/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`model_analysis/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`stores/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`trainer/` \ |Blank|
          │      └── classification.yaml
          └── config.yaml


Configuration Files
-------------------

| After preparing your data folder, you will need to create and edit the 
  configuration files to connect your dataset to the training pipeline.

| For a better understanding of which configuration files to change, you can 
  refer to the directory tree at :ref:`03a-config-folder-structure`.


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

   download: False
   url: ""
   blob_file: ""
   root_dir: "data"  # can be changed
   dataset: "<your_dataset_folder>"  # <your_dataset_folder> your dataset folder name
   train_dir: "./${.root_dir}/${.dataset}"  # your training data folder
   test_dir: "./${.root_dir}/${.dataset}"  # your testing data folder
   train_csv: "./${.root_dir}/${.dataset}/<your_csv_file>.csv"  # your training csv file
   image_path_col_name: "image_path"
   target_col_name: "class_name"
   target_col_id: "class_id"
   stratify_by: "${.target_col_name}"
   classification_type: "multiclass"
   image_size: 224
   num_classes: 10
   class_name_to_id:   # Change this to suit your dataset
      <class_1>: 0
      <class_2>: 1
      <class_3>: 2
   classes:            # Change this to suit your dataset
      - <class_1>
      - <class_2>
      - <class_3>

2. Edit Data Module File

| Edit the YAML file under the :mod:`configs/data_module` folder directory.

.. parsed-literal::

   \ :blue:`PeekingDuck/peekingduck/training/configs/data_module/` \ |Blank|
            ├── \ :blue:`data_adapter/...` \ |Blank|
            ├── \ :blue:`dataset/...` \ |Blank|
            ├── \ :blue:`resample/...` \ |Blank|
            ├── \ :blue:`transform/...` \ |Blank|
            └── classification.yaml

| Change the values where necessary:
| :mod:`.peekingduck/training/configs/data_module/classification.yaml`

.. code-block:: bash
   :linenos:

      defaults:
            - dataset: cifar10 # Change this value to the file name from the previous step


3. Edit Main Config File

| Edit the project name and use_case parameter in :mod:`.peekingduck/training/configs/config.yaml` file.


.. code-block:: bash
   :linenos:

    project_name: "<my_project_name>" # Change this value to your project name

    defaults:
    - use_case: classification # < classification | detection >  # Ensure this value is set to classification

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


Run
===

Assuming you have followed through the above steps, you can now test the 
image classification training pipeline with your custom dataset using the 
following commands in the terminal for either Tensorflow or PyTorch:

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


View the result of your training at the specified output folder directory: 
:mod:`\./PeekingDuck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`.

You can refer to this page :ref:`configuring_training_parameters_classification` for more details on how 
to customize your training parameters.
