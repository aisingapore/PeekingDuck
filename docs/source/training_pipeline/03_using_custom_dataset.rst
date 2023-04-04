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

.. raw:: html

   <h6>Step 1. Prepare the data folder as such</h6>

(It is impotant to note that <your_dataset_folder> should be the same value as your project_name value).

.. raw:: html
   
   <pre>
      PeekingDuck/data/
         ├── cifar10/...
         ├── rsna/...
         ├── vegfru/...
         └── &#60;your_dataset_folder&#62;/
               ├── &#60;your_images_folder&#62;/
               │      ├── &#60;class_1&#62;/
               │      │      ├── &#60;image_001&#62;.jpg
               │      │      ├── &#60;image_002&#62;.jpg
               │      │      ├── &#60;image_003&#62;.jpg
               │      │      └── ...
               │      ├── &#60;class_2&#62;/
               │      │      ├── &#60;image_001&#62;.jpg
               │      │      ├── &#60;image_002&#62;.jpg
               │      │      ├── &#60;image_003&#62;.jpg
               │      │      └── ...
               │      └── &#60;class_3&#62;/
               │             ├── &#60;image_001&#62;.jpg
               │             ├── &#60;image_002&#62;.jpg
               │             ├── &#60;image_003&#62;.jpg
               │             └── ...
               └── &#60;your_csv_file&#62;.yaml
   </pre>

| After preparing your folder, you will need to create and edit the configuration files to connect your dataset to the training pipeline:
| For better understanding of which configuration files to change, you can refer to the directory tree at :ref:`config-files-overview`

.. raw:: html

   <h6>Step 2. Create YAML files</h6>

| Create a yaml file and give it a unique name with no spacing inside these 2 folders:

.. raw:: html
   
   <pre>
      PeekingDuck/peekingduck/training/configs/data_module/
               ├── data_adapter/...
               ├── dataset/
               │      └── 1. &#60;dataset_filename&#62;.yaml
               ├── resample/...
               ├── transform/...
               └── 2. &#60;dataset_filename&#62;.yaml
   </pre>

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