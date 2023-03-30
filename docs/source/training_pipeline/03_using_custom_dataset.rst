.. include:: /include/substitution.rst

.. _using_custom_dataset:

********************
Using custom dataset
********************

.. _custom_dataset:

If you have already collected your own dataset, the following section describes how you can create the csv file needed, and use it within PeekingDuck for training.

`image_path`, `class_id` and  `class_name` are required columns in the `csv` file.

+-------------------+------------------+-----------------------+
| Columns           | Data type        | Description           |
+===================+==================+=======================+
| `image_path`      | `string`         | path to image file    |
+-------------------+------------------+-----------------------+
| `class_id`        | `integer`        | image class           |
+-------------------+------------------+-----------------------+
| `class_name`      | `string`         | image class name      |
+-------------------+------------------+-----------------------+

Example function to extract images path: :mod:`./PeekingDuck/general_utils.py` at feat-training (github.com)


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


| Once you have prepared the csv file and have your images, there are 2 steps to follow to connect your files to the training pipeline:

| For better understanding, it is best to refer to the directory tree at :ref:`config-files-overview`

.. raw:: html

   <h6>1. Create YAML file</h6>

| Create a yaml file and give it a unique name with no spacing inside these 2 folders:
| Folder 1: :mod:`.peekingduck/training/configs/data_module/dataset/<dataset_filename>.yaml`
| Add this code snippet and change the values where necessary:

.. code-block:: bash
   :linenos:

    download: False
    url: ""
    blob_file: ""
    root_dir: "data"  # can be changed
    train_dir: "./${.root_dir}/${project_name}"  # can be changed
    test_dir: "./${.root_dir}/${project_name}"  # can be changed
    train_csv: "./${.root_dir}/${project_name}/train.csv"  # can be changed
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

   <h6>2. Edit config.yaml</h6>

| Edit the data_module parameter in :mod:`.peekingduck/training/configs/config.yaml` file.

.. code-block:: bash
   :linenos:

    device: "auto"
    project_name: "project_name"
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
    

You can now test the training pipeline with the default training parameters using the following commands in terminal:

.. code-block:: bash
   :linenos:
   
   # Use the default configurations to test

   cd PeekingDuck

   # Tensorflow
   python ./peekingduck/training/main.py debug=True framework=tensorflow

   # Pytorch
   python ./peekingduck/training/main.py debug=True framework=pytorch

View the results of each run at the specified output folder directory: :mod:`\./Peekingduck/outputs/\<PROJECT_NAME\>/\<DATE_TIME\>`.
You can further configure your training by referring to the guide in section 2: :ref:`configuring_training_parameters`


This is the end of the documentation for the training pipeline. The next section will describe the PeekingDuck ecosystem.