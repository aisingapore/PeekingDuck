**********************
Pose Estimation Models
**********************

List of Pose Estimation Models
==============================

The table below shows the pose estimation models available for each task category.

+---------------+---------+-------------------------------------------------+
| Category      | Model   | Documentation                                   |
+===============+=========+=================================================+
|               | HRNet   | :mod:`peekingduck.pipeline.nodes.model.hrnet`   |
+               +---------+-------------------------------------------------+
| Whole body    | PoseNet | :mod:`peekingduck.pipeline.nodes.model.posenet` |
+               +---------+-------------------------------------------------+
|               | MoveNet | :mod:`peekingduck.pipeline.nodes.model.movenet` |
+---------------+---------+-------------------------------------------------+

Benchmarks
==========

Inference Speed
---------------

The table below shows the frames per second (FPS) of each model type.

+--------------+----------+-----------------+-------------------+-------------------+
|              |          |                 | CPU               | GPU               |
|              |          |                 +--------+----------+--------+----------+
| Model        | Type     | Size            | single | multiple | single | multiple |
+==============+==========+=================+========+==========+========+==========+
| PoseNet      | 50       | 225             |  79.81 |   59.97  | 136.31 |   89.37  |
+--------------+----------+-----------------+--------+----------+--------+----------+
| PoseNet      | 75       | 225             |  67.96 |   50.98  | 132.84 |   83.73  |
+--------------+----------+-----------------+--------+----------+--------+----------+
| PoseNet      | 100      | 225             |  50.84 |   41.02  | 132.73 |   81.24  |
+--------------+----------+-----------------+--------+----------+--------+----------+
| PoseNet      | resnet   | 225             |  15.27 |   14.32  |  73.15 |   51.65  |
+--------------+----------+-----------------+--------+----------+--------+----------+
| HRNet (YOLO) | (v4tiny) | 256 × 192 (416) |  5.86  |   1.09   |  21.91 |   13.86  |
+--------------+----------+-----------------+--------+----------+--------+----------+

Hardware
^^^^^^^^
The following hardware were used to conduct the FPS benchmarks:
 | - ``CPU``: MacBook Pro 2017, with 2.9 GHz Quad-Core Intel Core i7 and 16GB RAM
 | - ``GPU``: NVIDIA A100, paired with 2.2 GHz 6-Core Intel Xeon CPU and 85GB RAM

Test Conditions
^^^^^^^^^^^^^^^
The following test conditions were followed:
 | - :class:`input.recorded <peekingduck.pipeline.nodes.input.recorded.Node>`, the model of
    interest, and :class:`dabble.fps <peekingduck.pipeline.nodes.dabble.fps.Node>` nodes were
    used to perform inference on videos
 | - 2 videos were used to benchmark each model, one with only 1 human (``single``), and the other
    with multiple humans (``multiple``)
 | - Both videos are about 1 minute each, recorded at ~30 FPS, which translates to about 1,800
    frames to process per video
 | - 1280×720 (HD ready) resolution was used, as a bridge between 640×480 (VGA) of poorer quality
    webcams, and 1920×1080 (Full HD) of CCTVs
 | - All unnecessary processes, such as browsers, were closed to prevent IO/resource contention

Model Accuracy
--------------

The table below shows the performance of our pose estimation models using the keypoint evaluation
metrics from COCO. Description of these metrics can be found `here <https://cocodataset.org/#keypoints-eval>`__.

+--------------+----------+-----------------+------+----------------------+----------------------+---------------------+---------------------+--------------------+---------------------+----------------------+---------------------+--------------------+
| Model        | Type     | Size            | AP   | AP :sup:`OKS=.50`    | AP :sup:`OKS=.75`    | AP :sup:`medium`    | AP :sup:`large`     | AR                 | AR :sup:`OKS=.50`   | AR :sup:`OKS=.75`    | AR :sup:`medium`    | AR :sup:`large`    |
+==============+==========+=================+======+======================+======================+=====================+=====================+====================+=====================+======================+=====================+====================+
| PoseNet      | 50       | 225             | 5.2  | 15.4                 | 2.7                  | 0.8                 | 11.9                | 9.6                | 22.7                | 7.1                  | 1.4                 | 20.7               |
+--------------+----------+-----------------+------+----------------------+----------------------+---------------------+---------------------+--------------------+---------------------+----------------------+---------------------+--------------------+
| PoseNet      | 75       | 225             | 7.2  | 19.7                 | 3.6                  | 1.3                 | 16.0                | 12.0               | 26.5                | 9.3                  | 2.2                 | 25.4               |
+--------------+----------+-----------------+------+----------------------+----------------------+---------------------+---------------------+--------------------+---------------------+----------------------+---------------------+--------------------+
| PoseNet      | 100      | 225             | 7.8  | 20.8                 | 4.4                  | 1.5                 | 17.1                | 12.6               | 27.7                | 10.1                 | 2.4                 | 26.6               |
+--------------+----------+-----------------+------+----------------------+----------------------+---------------------+---------------------+--------------------+---------------------+----------------------+---------------------+--------------------+
| PoseNet      | resnet   | 225             | 11.9 | 27.4                 | 8.2                  | 2.2                 | 25.3                | 17.3               | 32.5                | 15.8                 | 2.9                 | 36.8               |
+--------------+----------+-----------------+------+----------------------+----------------------+---------------------+---------------------+--------------------+---------------------+----------------------+---------------------+--------------------+
| HRNet (YOLO) | (v4tiny) | 256 × 192 (416) | 33.3 | 56.0                 | 35.1                 | 27.1                | 42.0                | 37.3               | 58.0                | 39.6                 | 29.6                | 47.9               |
+--------------+----------+-----------------+------+----------------------+----------------------+---------------------+---------------------+--------------------+---------------------+----------------------+---------------------+--------------------+


Dataset
^^^^^^^

The `MS COCO <https://cocodataset.org/#download>`__ (val 2017) dataset is used. We integrated the
COCO API into the PeekingDuck pipeline for loading the annotations and evaluating the outputs from
the models. All values are reported in percentage.

All images from the "person" category in the MS COCO (val 2017) dataset were processed.


Keypoint IDs
============

.. _whole-body-keypoint-ids:

Whole Body
----------

+----------------+----+-------------+----+
| Keypoint       | ID | Keypoint    | ID |
+================+====+=============+====+
| nose           | 0  | left wrist  | 9  |
+----------------+----+-------------+----+
| left eye       | 1  | right wrist | 10 |
+----------------+----+-------------+----+
| right eye      | 2  | left hip    | 11 |
+----------------+----+-------------+----+
| leftEar        | 3  | right hip   | 12 |
+----------------+----+-------------+----+
| right ear      | 4  | left knee   | 13 |
+----------------+----+-------------+----+
| left shoulder  | 5  | right knee  | 14 |
+----------------+----+-------------+----+
| right shoulder | 6  | left ankle  | 15 |
+----------------+----+-------------+----+
| left elbow     | 7  | right ankle | 16 |
+----------------+----+-------------+----+
| right elbow    | 8  |             |    |
+----------------+----+-------------+----+
