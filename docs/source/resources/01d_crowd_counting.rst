***********************
Crowd Counting Models
***********************

List of Crowd Counting Models
===============================

The table below shows the crowd counting models available.

+------------------------+------------------------------------------------------------+
| Model                  | Documentation                                              |
+========================+============================================================+
| CSRNet                 | :mod:`peekingduck.pipeline.nodes.model.csrnet`             |
+------------------------+------------------------------------------------------------+

Benchmarks
==========


Model Accuracy
--------------

The table below shows the performance of CSRNet obtained from the original 
`GitHub repo <https://github.com/Neerajj9/CSRNet-keras>`__, using Mean Absolute Error (MAE) as the 
metric. The reported metrics are close to the results from the
`CSRNet paper <https://arxiv.org/pdf/1802.10062.pdf>`__. 

+--------------+--------+---------------------+-------+
| Model        | Type   | Dataset             | MAE   |
+==============+========+=====================+=======+
|              | dense  | ShanghaiTech Part A | 65.92 |
|              +--------+---------------------+-------+
| CSRNet       | sparse | ShanghaiTech Part B | 11.01 |
+--------------+--------+---------------------+-------+

Dataset
^^^^^^^

The `ShanghaiTech <https://www.kaggle.com/tthien/shanghaitech>`__ dataset was used. It contains 1,198 annotated
images split into 2 parts: Part A contains 482 images with highly congested scenes, while Part B contains 716 
images with relatively sparse crowd scenes. 