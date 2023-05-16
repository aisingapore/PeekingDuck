.. include:: /include/substitution.rst

.. _configuring_training_parameters:

*******************************
Configuring Training Parameters
*******************************

This section describes how you can configure the pipeline to suite your model training.
\ The following parts will describe the image classification section and the documentation conventions used.


Image Classification
====================

.. toctree::
   :maxdepth: 3

   /training_pipeline/02_configuring_training_parameters/use_case/02a_image_classification


Documentation Convention
========================

.. _documentation_convention:

| There will be multiple instances talking about the configuration files that you will be interacting with to change the pipeline settings.

| These configuration files will be displayed in a table format, illustrated in the following table.
| The text color scheme shown is used to illustrate which of the values can be changed, and the ones that should not:


+----------------+------------------------------------------------+
| Color          | Context                                        |
+================+================================================+
| :grey:`Grey`   | :grey:`Required Value (Don't change this)`     |
+----------------+------------------------------------------------+
| :mod:`Black`   | This value can be changed                      |
+----------------+------------------------------------------------+


Example
*******


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

