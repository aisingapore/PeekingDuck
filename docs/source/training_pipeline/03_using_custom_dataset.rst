.. include:: /include/substitution.rst

********************
Using custom dataset
********************

.. _custom_dataset:

If you have already collected your own dataset, the following section describes how you can create the csv file needed, and use it within PeekingDuck for training.

+--------------------------------------------------------------+----------+-------------------+
| image_path                                                   | class_id | class_name        |
+==============================================================+==========+===================+
| data/vegfru/veg200_images/Chinese_artichoke/v_14_01_0001.jpg | 13       | Chinese_artichoke |
+--------------------------------------------------------------+----------+-------------------+
| data/vegfru/veg200_images/Chinese_artichoke/v_06_03_0007.jpg | 3        | Chinese_kale      |
+--------------------------------------------------------------+----------+-------------------+
| data/vegfru/veg200_images/Chinese_artichoke/v_09_03_0031.jpg | 8        | Chinese_pumpkin   |
+--------------------------------------------------------------+----------+-------------------+

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

Example function to extract images path: PeekingDuck/general_utils.py at feat-training · (github.com)