*******************
CV Model Benchmarks
*******************

PeekingDuck is equipped with a wide range of models for different scenarios, from real-time object detection to high-accuracy
pose estimation. The full list of models can be found :mod:`here <peekingduck.pipeline.nodes.model>`.

Given various constraints, such as hardware and costs, users are often faced with a tough trade-off between 
inference speed and model accuracy. Hence, we prepared the following benchmarks to help users choose the most suitable model for 
their application. These benchmarks are obtained using PeekingDuck's ``model`` nodes, and are measured in Frames Per Second (FPS) for 
inference speed and mean average precision (mAP) for model accuracy. 


Inference Speed
===============

Test Hardware
-------------
We ran FPS benchmarks on the following devices to compare CPU vs GPU performance:
 | - CPU: MacBook Pro 2017, with 2.9 GHz Quad-Core Intel Core i7 and 16GB RAM
 | - GPU: NVIDIA A100, paired with 2.2 GHz 6-Core Intel Xeon CPU and 85GB RAM

Test Conditions
---------------
The following test conditions were followed:
 | - ``input.recorded``, the model of interest, and ``dabble.fps`` nodes were used to perform inference on videos
 | - 2 videos were used to benchmark each model, one with only 1 human (``single``), and the other with multiple humans (``multiple``)
 | - Both videos are about 1 minute each, recorded at ~30 FPS, which translates to about 1,800 frames to process per video
 | - 1280x720 (HD ready) resolution was used, as a bridge between 640x480 (VGA) of poorer quality webcams, and 1920x1080 (Full HD) of CCTVs
 | - All unnecessary processes, such as browsers, were closed to prevent IO/resource contention

Results
-------
The results below show the FPS (Frames Per Second) of each model type.

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