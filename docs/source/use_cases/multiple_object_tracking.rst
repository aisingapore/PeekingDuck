************************
Multiple Object Tracking
************************

Overview
========

Multiple Object Tracking (MOT) is the task of detecting unique objects and tracking them as they
move across frames in a video.

AI Singapore has developed a vision-based `multiple object tracker <https://aisingapore.org/2021/05/covid-19-stay-vigilant-with-group-size-checker>`_
that tracks multiple moving objects. This tracking capability is to be used in tandem with an
object detection model. Objects to track can be, for example, pedestrians on the street, vehicles
in the road, sport players on the court, groups of animals and more.

.. image:: /assets/use_cases/multiple_object_tracking.gif
   :class: no-scaled-link
   :width: 70 %

Currently, there are two types of trackers available with PeekingDuck: MOSSE (using OpenCV), and
Intersection Over Union (IoU). This is explained in the `How it Works`_ section.

Demo
====

.. |pipeline_config| replace:: multiple_object_tracking.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/dev/use_cases/multiple_object_tracking.yml

To try our solution on your own computer, :doc:`install </getting_started/02_basic_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. parsed-literal::

    > peekingduck run --config_path <path/to/\ |pipeline_config|\ >

How it Works
============

There are three main components to obtain the track of an object:

#. Object detection using AI,
#. Measuring similarity between objects in frames, and
#. Recovering the identity information based on the similarity measurement between objects across
   frames.

**1. Object Detection**

The MOT node requires a detected bounding box from an object detector model. To achieve this with
PeekingDuck, you may use our open source models such as YOLOv4, EfficientDet, and `PoseNet <https://arxiv.org/abs/1505.07427>`_
(for human detection only) which return detected bounding boxes. This allows the application to
identify where each object is located within the video feed. The location is returned as two `x, y`
coordinates in the form :math:`[x_1, y_1, x_2, y_2]`, where :math:`(x_1, y_1)` is the top left
corner of the bounding box, and :math:`(x_2, y_2)` is the bottom right. These are used to form the
bounding box of each object detected which will then be used to determine a track for each object.

.. image:: /assets/use_cases/yolo_demo.gif
   :class: no-scaled-link
   :width: 70 %

**2. MOSSE (Using OpenCV)**

Minimum Output Sum of Squared Error (MOSSE) uses an adaptive correlation for object tracking which
produces stable correlation filters when initialized using a single frame. MOSSE tracker is robust
to variations in lighting, scale, pose, and non-rigid deformations. It also detects occlusion based
upon the Peak to Sidelobe Ratio (PSR), which enables the tracker to pause and resume where it left
off when the object reappears. MOSSE tracker also operates at a higher FPS. It is much faster than
other models but not as accurate.

The bounding boxes detected in the first frame are used to initialize a single tracker instance for
each detection. The tracker for each bounding box is then updated per frame and is deleted when the
tracker fails to find a match over time.

To account for new detections in a frame, which do not have an associated tracker, we perform an
IoU of the new bounding box with previous tracked bounding boxes. Should the IoU exceed a
threshold, it is then associated with a current track, otherwise a new instance of a track is
initialized for the new bounding box.

**3. Intersection Over Union**

With ever increasing performances of object detectors, the basis for a tracker becomes much more
reliable. This enables the deployment of much simpler tracking algorithms which can compete with
more sophisticated approaches at a fraction of the computational cost. Check out the
`original paper <http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf>`_ for more details.

This method is based on the assumption that the detector produces a detection per frame for every
object to be tracked, i.e., there are none or only few "gaps" in the detections. Furthermore, it is
assumed that detections of an object in consecutive frames have an unmistakably high overlap IoU
which is commonly the case when using sufficiently high frame rates.

The authors proposed a simple IoU tracker which essentially continues a track by associating the
detection with the highest IoU to the last detection in the previous frame if a certain IoU
threshold is met. All detections not assigned to an existing track will start a new one.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.recorded:   
       input_dir: <path/to/input_video>
   - model.yolo:
       model_type: "v4tiny"
   - dabble.fps
   - dabble.tracking:
       tracking_type: "iou"
   - draw.tag
   - draw.bbox
   - draw.legend
   - output.media_writer:
       output_dir: <path/to/output_folder>

**1. Object Detection Node**

By default, the node uses the YOLOv4-tiny model for object detection, set to detect people
(``detect_ids: [0]``). To use more accurate models, you can try the :mod:`YOLOv4 model <model.yolo>`
or the :mod:`model.efficientdet` that is included in our repo.

**2. Adjusting Nodes**

Some common node behaviors that you might need to adjust are:

* ``model_type``: ``v4``, or ``v4tiny`` for :mod:`model.yolo`. ``0``, ``1``, ``2``, ``3``, or ``4``
  for :mod:`model.efficientdet` node. Either of these models can be used for object detection.
* ``detect_ids``: Object class IDs to be detected. Refer to :ref:`Object Detection IDs table <general-object-detection-ids>`
  for the class IDs for each model.
* ``tracking_type``: The type of tracking to be used, choose one of: ``["iou", "mosse"]``.

For more adjustable node behaviors not listed here, check out the :ref:`API Documentation <api_doc>`.