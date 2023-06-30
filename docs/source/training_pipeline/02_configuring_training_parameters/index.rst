.. include:: /include/substitution.rst

.. _configuring_training_parameters:

*******************************
Configuring Training Parameters
*******************************

This section describes how you can configure the pipeline to suit your model 
training.


Documentation Convention
========================

.. _documentation_convention:

The following table shows the documentation convention for the training 
pipeline configurations. The color scheme is used to differentiate between the 
values that can be changed and the ones that should not:

+----------------+------------------------------------------------+
| Color          | Context                                        |
+================+================================================+
| :grey:`Grey`   | :grey:`Required Value (Don't change this)`     |
+----------------+------------------------------------------------+
| :mod:`Black`   | This value can be changed                      |
+----------------+------------------------------------------------+

The table below shows one example based on the above convention:

+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| Key                                               | Value                                               |   Action                            |
+===================================================+=====================================================+=====================================+
| :mod:`adapter`                                    |  "timm"                                             | Can be changed.                     |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :grey:`task`                                      |  :grey:`${use_case}`                                | :grey:`Should not be changed.`      |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :mod:`model_name`                                 |  "vgg16"                                            | Can be changed.                     |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :mod:`weights`                                    |  "DEFAULT"                                          | Can be changed.                     |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :mod:`pretrained`                                 |  True                                               | Can be changed.                     |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :mod:`fine_tune`                                  |  True                                               | Can be changed.                     |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :mod:`fine_tune_all`                              |  True                                               | Can be changed.                     |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :mod:`fine_tune_modules`                          |                                                     | Can be changed.                     |
+------------------------------+--------------------+-----------------------------------------------------+-------------------------------------+
|                              | :mod:`features`    | 7                                                   | Can be changed.                     |
+------------------------------+--------------------+-----------------------------------------------------+-------------------------------------+
|                              | :mod:`pre_logits`  | [ "fc1", "act1", "drop", "fc2", "act2" ]            | Can be changed.                     |
+------------------------------+--------------------+-----------------------------------------------------+-------------------------------------+
| :grey:`num_classes`                               | :grey:`${data_module.dataset.num_classes}`          | :grey:`Should not be changed.`      |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+
| :grey:`device`                                    | :grey:`${device}`                                   | :grey:`Should not be changed.`      |
+---------------------------------------------------+-----------------------------------------------------+-------------------------------------+


Configuration Structure
=======================

We use the YAML syntax for the config file. Below shows the folder structure 
and describes how users can understand and navigate the config structure. 
Configuration files that are not user-customizable are not included in the 
view below.

.. parsed-literal::

    \ :blue:`peekingduck/training/configs/` \ |Blank|
          ├── \ :blue:`callbacks/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`data_module/` \ |Blank|
          │      ├── \ :blue:`data_adapter/` \ |Blank|
          │      │      └── adapter.yaml
          │      ├── \ :blue:`dataset/` \ |Blank|
          │      │      ├── cifar10.yaml
          │      │      ├── fashion_coco_format.yaml
          │      │      ├── fashion_voc_format.yaml
          │      │      ├── rsna.yaml
          │      │      └── vegfru.yaml
          │      ├── \ :blue:`resample/` \ |Blank|
          │      │      └── train_test_split.yaml
          │      ├── \ :blue:`transform/` \ |Blank|
          │      │      ├── test.yaml
          │      │      └── train.yaml
          │      ├── classification.yaml
          │      └── detection.yaml
          ├── \ :blue:`metrics/` \ |Blank|
          │      └── classification.yaml
          ├── \ :blue:`model/` \ |Blank|
          │      ├── classification.yaml
          │      └── detection.yaml
          ├── \ :blue:`model_analysis/` \ |Blank|
          │      └── detection.yaml
          ├── \ :blue:`trainer/` \ |Blank|
          │      ├── classification.yaml
          │      └── detection.yaml
          ├── \ :blue:`use_case/` \ |Blank|
          │      ├── classification.yaml
          │      └── detection.yaml
          └── config.yaml

+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Folder/file under configs folder | Description                                                                                                                                                                         |
+==================================+=====================================================================================================================================================================================+
| config.yaml                      | Main configuration file that defines the project name and use case (classification / objection / ...).                                                                              |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| callbacks                        | Specify callbacks for TensorFlow or PyTorch. Applicable to classification only.                                                                                                     |
|  └── classification.yaml         |                                                                                                                                                                                     |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data_module                      | Main configuration files for data module.                                                                                                                                           |
|  ├── classification.yaml         |                                                                                                                                                                                     |
|                                  |                                                                                                                                                                                     |
|  └── detection.yaml              |                                                                                                                                                                                     |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data_module                      | Adapter for TensorFlow and PyTorch configurations that controls the batch size for train/test dataset, shuffling control, etc.                                                      |
|  └── data_adapter                |                                                                                                                                                                                     |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data_module                      | | Configurations for training dataset in separate .yaml files, including options for downloading the dataset, image size, number of classes, mapping of the class names to id, etc. |
|  └── dataset                     |                                                                                                                                                                                     |
|                                  | A separate YAML file needs to be created for each custom dataset.                                                                                                                   |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data_module                      | Controls train/test split and shuffling.                                                                                                                                            |
|  └── resample                    |                                                                                                                                                                                     |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| data_module                      | Controls image augmentations and transformations for train and test, such as cropping, flipping, etc.                                                                               |
|  └── transform                   |                                                                                                                                                                                     |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| metrics                          | Choose training metrics to monitor during training                                                                                                                                  |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model                            | Controls selection of pre-trained models and fine-tuning model settings.                                                                                                            |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| model_analysis                   | Configure model analysis parameters like weights and biases.                                                                                                                        |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| trainer                          | Control training related parameters including number of epochs, learning rate, loss function, metric and patience for early stopping                                                |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| use_case                         | Define high-level configurations for each use case (classification / detection / ...)                                                                                               |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Changing the Configurations
===========================

There are two ways to change the default configuration:

1. Update the parameter values inside YAML file for the respective 
   configuration.

2. Pass the argument in the command line.

For the second option, you can pass the arguments explicitly stated in the main 
`config.yaml` file directly in the command line, such as follows:

.. code-block:: bash
   :linenos: 

   cd PeekingDuck
   python ./peekingduck/training/main.py debug=True framework=tensorflow project_name=my_project view_only=True

To change the arguments in other configuration files such as `model`, 
`trainer`, etc., the user would need to chain up the arguments based on the 
hierarchy in the yaml files. Here is one example:

.. code-block:: bash
   :linenos: 

   cd PeekingDuck
   python ./peekingduck/training/main.py debug=True framework=pytorch model.pytorch.model_name=mobilenetv3_small_050 trainer.pytorch.global_train_params.debug_epochs=5


Supported Use Cases
===================

Refer to the following sections for the detailed configurations for each use case:

.. toctree::
   :maxdepth: 1

   /training_pipeline/02_configuring_training_parameters/use_case/02a_image_classification
   /training_pipeline/02_configuring_training_parameters/use_case/02b_object_detection
