**********************
Object Tracking Models
**********************

List of Object Tracking Models
==============================

The table below shows the object tracking models available for each task category.

+---------------+----------------------+---------------------------------------------------+
| Category      | Model                | Documentation                                     |
+===============+======================+===================================================+
|               | IoU Tracker          | :mod:`peekingduck.pipeline.nodes.dabble.tracking` |
+               +----------------------+---------------------------------------------------+
| General       | OpenCV MOSSE Tracker | :mod:`peekingduck.pipeline.nodes.dabble.tracking` |
+---------------+----------------------+---------------------------------------------------+
| Human         | JDE                  | :mod:`peekingduck.pipeline.nodes.model.jde`       |
+---------------+----------------------+---------------------------------------------------+

Benchmarks
==========

.. _object-tracking-benchmarks:

Model Accuracy
--------------

The table below shows the performance of our object tracking models using multiple object tracker
(MOT) metrics from MOT Challenge. Description of these metrics can be found
`here <https://motchallenge.net/results/MOT16/#metrics>`__.


+---------------------------------+----------------------+-------+-------+--------+-------+--------+
| Model                           | Object Detector Type | MOTA  | IDF1  | ID Sw. | FP    | FN     |
+=================================+======================+=======+=======+========+=======+========+
| IoU Tracker with YOLOX          | yolox-m              | 34.1  | 40.9  | 960    | 8997  | 62830  |
+---------------------------------+----------------------+-------+-------+--------+-------+--------+
| OpenCV MOSSE Tracker with YOLOX | yolox-m              | 32    | 37.6  | 2124   | 7695  | 65268  |
+---------------------------------+----------------------+-------+-------+--------+-------+--------+
| JDE                             | --                   | 70.1  | 65.1  | 1321   | 6412  | 25292  |
+---------------------------------+----------------------+-------+-------+--------+-------+--------+

Dataset
^^^^^^^

The `MOT16 <https://motchallenge.net/data/MOT16/>`__ (train) dataset is used. We integrated the
MOT Challenge API into the PeekingDuck pipeline for loading the annotations and evaluating the
outputs from the models. `MOTA` and `IDF1` are reported in percentage while `IDS`, `FP`, and `FN`
are raw numbers.

Only the "pedestrian" category in MOT16 (train) was processed.
