*******************
CV Model Benchmarks
*******************

At PeekingDuck, we aim to give users as much flexibility as possible to cater to every possible scenario.
From real-time object detection to high-accuracy pose estimation, PeekingDuck houses a wide range of models.
The full list of models can be found :mod:`here <peekingduck.pipeline.nodes.model>`.

Choosing the right model, given hardware constraints, is usually about making a trade-off between inference speed
(FPS, or Frames Per Second, in the case of CV), and model performance (mAP, or mean average precision for some CV tasks).
We would like to use our best performing models all the time, but in the real world, we are often limited by factors such
as project costs and power/size requirements of the hardware.

We hope that these benchmarks would help you choose the most suitable model for your application.


Inference Speed
===============

Test Hardware
-------------
We ran FPS benchmarks on the following devices to compare CPU vs GPU performance:
 | - MacBook Pro 2017, with 2.9 GHz Quad-Core Intel Core i7 and 16GB RAM
 | - NVIDIA A100 GPU, paired with 2.2 GHz 6-Core Intel Xeon CPU and 85GB RAM

Test Conditions
---------------
The following test conditions were followed:
 | - ``input.recorded``, the model of interest, and ``dabble.fps`` nodes were used to perform inference on videos
 | - 2 videos were used to benchmark each model, one with only 1 human (``single``), and the other with multiple humans (``multiple``)
 | - Both videos are about 1 minute each, recorded at ~30 FPS, which translates to about 1,800 frames to process for each
 | - 1280x720 (HD ready) resolution was used, as a bridge between 640x480 (VGA) of poorer quality webcams, and 1920x1080 (Full HD) of CCTVs
 | - All unnecessary processes such as browsers were closed to prevent IO/resource contention

Results
-------
The results below show the FPS (Frames Per Second) of each model type.
It is quite clear that the FPS of pose estimation models significantly drops as the number of humans in the videos increase.

+------------------------------------------+-------------------+-------------------+
|                                          |  MacBook Pro 2017 |    NVIDIA A100    |
+------------------+--------------+--------+--------+----------+--------+----------+
|       Task       |     Model    |  Type  | single | multiple | single | multiple |
+------------------+--------------+--------+--------+----------+--------+----------+
| Object Detection |     YOLO     | v4tiny |  22.42 |   21.71  |  65.24 |   57.50  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  |     YOLO     |   v4   |  2.62  |   2.59   |  30.40 |   28.71  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  | EfficientDet |    0   |  5.24  |   5.25   |  29.51 |   29.39  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  | EfficientDet |    1   |  2.53  |   2.49   |  23.79 |   24.44  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  | EfficientDet |    2   |  1.54  |   1.50   |  19.86 |   20.51  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  | EfficientDet |    3   |  0.78  |   0.75   |  14.69 |   14.84  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  | EfficientDet |    4   |  0.43  |   0.42   |  11.74 |   11.88  |
+------------------+--------------+--------+--------+----------+--------+----------+
|  Pose Estimation |    PoseNet   |   50   |  79.81 |   59.97  | 136.31 |   89.37  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  |    PoseNet   |   75   |  67.96 |   50.98  | 132.84 |   83.73  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  |    PoseNet   |   101  |  50.50 |   41.06  | 132.52 |   81.81  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  |    PoseNet   | resnet |  15.27 |   14.32  |  73.15 |   51.65  |
|                  +--------------+--------+--------+----------+--------+----------+
|                  | HRNet + YOLO | v4tiny |  5.86  |   1.09   |  21.91 |   13.86  |
+------------------+--------------+--------+--------+----------+--------+----------+

Model Performance
=================

We are currently running model performance benchmarks and will update this section soon.