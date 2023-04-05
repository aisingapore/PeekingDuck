.. include:: /include/substitution.rst

.. _using_custom_dataset:

********************
Using custom dataset
********************

If you have already collected your own dataset, the following section describes how you can prepare your dataset, connect it to the training pipeline and use it for training a model.

.. _custom_dataset:

Label File Template
===================

| The table below describe the columns needed to create the csv file which will be used by the training pipeline to map your images to their labels.
| `image_path`, `class_id` and  `class_name` are required columns in the `csv` file.

+-------------------+------------------+-----------------------+
| Columns           | Data type        | Description           |
+===================+==================+=======================+
| `image_path`      | `string`         | path to image file    |
+-------------------+------------------+-----------------------+
| `class_id`        | `integer`        | image class           |
+-------------------+------------------+-----------------------+
| `class_name`      | `string`         | image class name      |
+-------------------+------------------+-----------------------+

Below is an example of the csv file using the vegfru dataset:

+--------------------------------------------------------------+----------+-------------------+
| image_path                                                   | class_id | class_name        |
+==============================================================+==========+===================+
| data/vegfru/veg200_images/Chinese_artichoke/v_14_01_0001.jpg | 13       | Chinese_artichoke |
+--------------------------------------------------------------+----------+-------------------+
| data/vegfru/veg200_images/Chinese_artichoke/v_06_03_0007.jpg | 3        | Chinese_kale      |
+--------------------------------------------------------------+----------+-------------------+
| data/vegfru/veg200_images/Chinese_artichoke/v_09_03_0031.jpg | 8        | Chinese_pumpkin   |
+--------------------------------------------------------------+----------+-------------------+

Preparation & Integrating with Training Pipeline
================================================

Once you have prepared the csv file and have your images you can follow these 3 steps described below.


Step 1. Prepare the data folder as such
---------------------------------------

It is important to note that :mod:`<your_dataset_folder>` should be the same value as the :mod:`project_name` value defined in the :ref:`config-files-mainconfig`.


.. parsed-literal::

   \ :blue:`PeekingDuck/data/` \ |Blank|
      ├── \ :blue:`cifar10/...` \ |Blank|
      ├── \ :blue:`rsna/...` \ |Blank|
      ├── \ :blue:`vegfru/...` \ |Blank|
      └── \ :blue:`<your_dataset_folder>/` \ |Blank|
            ├── \ :blue:`<your_images_folder>/` \ |Blank|
            │      ├── \ :blue:`<class_1>/` \ |Blank|
            │      │      ├── <image_001>.jpg
            │      │      ├── <image_002>.jpg
            │      │      ├── <image_003>.jpg
            │      │      └── ...
            │      ├── \ :blue:`<class_2>/` \ |Blank|
            │      │      ├── <image_001>.jpg
            │      │      ├── <image_002>.jpg
            │      │      ├── <image_003>.jpg
            │      │      └── ...
            │      └── \ :blue:`<class_3>/` \ |Blank|
            │             ├── <image_001>.jpg
            │             ├── <image_002>.jpg
            │             ├── <image_003>.jpg
            │             └── ...
            └── <your_csv_file>.yaml



| After preparing your folder, you will need to create and edit the configuration files to connect your dataset to the training pipeline:
| For better understanding of which configuration files to change, you can refer to the directory tree at :ref:`config-files-overview`


Step 2. Create YAML files
-------------------------

| Create two yaml files under the :mod:`data_module` directory and :mod:`dataset` subdirectory.
| The name of the files must be the same as the :mod:`data_module` value defined in the :ref:`config-files-mainconfig`.

.. parsed-literal::

   \ :blue:`PeekingDuck/peekingduck/training/configs/data_module/` \ |Blank|
            ├── \ :blue:`data_adapter/...` \ |Blank|
            ├── \ :blue:`dataset/` \ |Blank|
            │      └── 1. <dataset_filename>.yaml
            ├── \ :blue:`resample/...` \ |Blank|
            ├── \ :blue:`transform/...` \ |Blank|
            └── 2. <dataset_filename>.yaml   


| Folder 1: :mod:`.peekingduck/training/configs/data_module/dataset/<dataset_filename>.yaml`
| Add this code snippet and change the values where necessary:

.. code-block:: bash
   :linenos:

   download: False
   url: ""
   blob_file: ""
   root_dir: "data"  # can be changed
   train_dir: "./${.root_dir}/${project_name}"  # <your_dataset_folder> should be the same as the project_name value
   test_dir: "./${.root_dir}/${project_name}"  # <your_dataset_folder> should be the same as the project_name value
   train_csv: "./${.root_dir}/${project_name}/<your_csv_file>.csv"  # <your_dataset_folder> should be the same as the project_name value
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


| Folder 2: :mod:`.peekingduck/training/configs/data_module/<dataset_filename>.yaml`
| Add this code snippet and change the values where necessary:

.. code-block:: bash
   :linenos:

   defaults:
      - dataset: <dataset_filename>    # Change this to your dataset file name
      - resample: train_test_split
      - transform:
               - train
               - test
      - data_adapter:
               - adapter

   module:
      _target_: src.data.data_module.ImageClassificationDataModule
      _recursive_: False

   framework: ${framework}
   debug: ${debug}
   num_debug_samples: 25 # can be changed

.. raw:: html

   <h6>Step 3. Edit config.yaml</h6>

| Edit the data_module parameter in :mod:`.peekingduck/training/configs/config.yaml` file.

.. code-block:: bash
   :linenos:

    device: "auto"
    project_name: "<your_dataset_folder>" # change this value
    debug: True
    framework: "tensorflow"
    random_state: 11
    view_only: False

    defaults:
    - use_case: classification
    - data_module: <dataset_filename> # change this value
    - model: ${use_case}
    - trainer: ${use_case}
    - callbacks: ${use_case}
    - metrics: ${use_case}
    - model_analysis: ${use_case}
    - stores: ${use_case}
    - _self_
    - override hydra/job_logging: custom
    

Testing the Pipeline
====================

Refer to :ref:`getting-started-test-run` to test out the pipeline.

This is the end of the documentation for the training pipeline. The next section will describe the PeekingDuck ecosystem.