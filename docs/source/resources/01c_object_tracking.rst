**********************
Object Tracking Models
**********************

List of Object Tracking Models
==============================

The table below shows the object tracking models available for each task category.

+---------------+----------------------+------------------------+
| Category      | Model                | Documentation          |
+===============+======================+========================+
|               | IoU Tracker          | :mod:`dabble.tracking` |
+               +----------------------+------------------------+
| General       | OpenCV MOSSE Tracker | :mod:`dabble.tracking` |
+---------------+----------------------+------------------------+
|               | JDE                  | :mod:`model.jde`       |
+               +----------------------+------------------------+
| Human         | FairMOT              | :mod:`model.fairmot`   |
+---------------+----------------------+------------------------+

Benchmarks
==========

.. _object-tracking-benchmarks:


Inference Speed
---------------

The table below shows the frames per second (FPS) of each model type.

+---------------------------------+----------------------+------------+-------+--------+
| Model                           | Object Detector Type | Input Size | CPU   | GPU    |
+=================================+======================+============+=======+========+
| IoU Tracker with YOLOX          | yolox-m              | --         | 7.87  | 36.18  |
+---------------------------------+----------------------+------------+-------+--------+
| OpenCV MOSSE Tracker with YOLOX | yolox-m              | --         | 6.74  | 21.45  |
+---------------------------------+----------------------+------------+-------+--------+
| JDE                             | --                   | --         | 1.86  | 26.32  |
+---------------------------------+----------------------+------------+-------+--------+
| FairMOT                         | --                   | 864 × 480  | 0.30  | 22.60  |
+---------------------------------+----------------------+------------+-------+--------+


Hardware
^^^^^^^^

The following hardware were used to conduct the FPS benchmarks:
 | - ``CPU``: 2.8 GHz 4-Core Intel Xeon (Cascade Lake) CPU and 16GB RAM
 | - ``GPU``: NVIDIA A100, paired with 2.2 GHz 6-Core Intel Xeon CPU and 85GB RAM

Test Conditions
^^^^^^^^^^^^^^^

The following test conditions were followed:
 | - :mod:`input.visual`, the model of interest, and :mod:`dabble.fps` nodes were used to perform
     inference on videos
 | - A video sequence from the MOT Challenge dataset (MOT16-04) was used
 | - The video sequence has 1050 frames and is encoded at 30 FPS, which translates to about 35 seconds
 | - 1280×720 (HD ready) resolution was used, as a bridge between 640×480 (VGA) of poorer quality
     webcams, and 1920×1080 (Full HD) of CCTVs

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
| OpenCV MOSSE Tracker with YOLOX | yolox-m              | 32.8  | 38    | 2349   | 7695  | 65268  |
+---------------------------------+----------------------+-------+-------+--------+-------+--------+
| JDE                             | --                   | 70.1  | 65.1  | 1321   | 6412  | 25292  |
+---------------------------------+----------------------+-------+-------+--------+-------+--------+
| FairMOT                         | --                   | 81.8  | 80.9  | 536    | 3663  | 15903  |
+---------------------------------+----------------------+-------+-------+--------+-------+--------+

Dataset
^^^^^^^

The `MOT16 <https://motchallenge.net/data/MOT16/>`__ (train) dataset is used. We integrated the
MOT Challenge API into the PeekingDuck pipeline for loading the annotations and evaluating the
outputs from the models. `MOTA` and `IDF1` are reported in percentages while `IDS`, `FP`, and `FN`
are raw numbers.

Only the "pedestrian" category in MOT16 (train) was processed.
