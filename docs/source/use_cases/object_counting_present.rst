*************************
Object Counting (Present)
*************************

.. include:: /include/substitution.rst

Overview
========

Object counting (present) is a solution within PeekingDuck's suite of 
:ref:`smart monitoring <smart_monitoring_use_cases>` use cases. It counts the number of objects
detected by PeekingDuck's object detection models at the present point in time, and calculates 
statistics such as the cumulative average, maximum and minimum for further analytics. Up to
:ref:`80 types <general-object-detection-ids>` of objects can be counted, including humans,
vehicles, animals and even household objects. Thus, this can be applied to a wide variety of
scenarios, from traffic control to counting livestock.

.. seealso::

   For advanced counting tasks such as counting tracked objects over time or counting within
   specific zones, refer to PeekingDuck's other :ref:`smart monitoring <smart_monitoring_use_cases>`
   use cases.

.. image:: /assets/use_cases/object_counting_present.gif
   :class: no-scaled-link
   :width: 50 %

In the GIF above, the count and statistics change as the number of detected persons change. 
This is explained in the `How It Works`_ section.

Demo
====

.. |pipeline_config| replace:: object_counting_present.yml
.. _pipeline_config: https://github.com/aimakerspace/PeekingDuck/blob/main/use_cases/object_counting_present.yml

To try our solution on your own computer, :doc:`install </getting_started/02_standard_install>` and run
PeekingDuck with the configuration file |pipeline_config|_ as shown:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`peekingduck run -\-config_path <path/to/`\ |pipeline_config|\ :green:`>`

How It Works
============

There are 3 main components to this solution:

#. Object detection,
#. Count detections, and
#. Calculate statistics.

**1. Object Detection**

We use an open source object detection model known as `YOLOv4 <https://arxiv.org/abs/2004.10934>`_
and its smaller and faster variant known as YOLOv4-tiny to identify the bounding boxes of chosen
objects we want to detect. This allows the application to identify where objects are located within
the video feed. The location is returned as two `x, y` coordinates in the form
:math:`[x_1, y_1, x_2, y_2]`, where :math:`(x_1, y_1)` is the top left corner of the bounding box,
and :math:`(x_2, y_2)` is the bottom right. These are used to form the bounding box of each object
detected. For more information on how to adjust the ``yolo`` node, check out its
:doc:`configurable parameters </nodes/model.yolo>`.

.. image:: /assets/use_cases/yolo_demo.gif
   :class: no-scaled-link
   :width: 50 %

**2. Count Detections**

To count the number of objects detected, we simply take the sum of the number of bounding boxes
detected for the object category.

**3. Calculate Statistics**

The cumulative average, minimum and maximum over time is calculated from the count from each frame.

Nodes Used
==========

These are the nodes used in the earlier demo (also in |pipeline_config|_):

.. code-block:: yaml

   nodes:
   - input.visual:
       source: 0
   - model.yolo:
       detect: ["person"]
   - dabble.bbox_count
   - dabble.statistics:
       identity: count
   - draw.bbox
   - draw.legend:
       show: ["count", "cum_avg", "cum_max", "cum_min"]
   - output.screen


**1. Object Detection Node**

By default, the node uses the YOLOv4-tiny model for object detection, set to detect people. Please
take a look at the :doc:`benchmarks </resources/01a_object_detection>` of object detection models
that are included in PeekingDuck if you would like to use a different model or model type better
suited to your use case.

**2. Object Counting Node**

:mod:`dabble.bbox_count` takes the detected bounding boxes and outputs the total count of bounding boxes.
This node has no configurable parameters.

**3. Statistics Node**

The :mod:`dabble.statistics` node calculates the :term:`cum_avg`, :term:`cum_max` and :term:`cum_min`
from the output of the object counting node.

**4. Adjusting Nodes**

For the object detection model used in this demo, please see the :doc:`documentation </nodes/model.yolo>`
for adjustable behaviors that can influence the result of the object counting node.

For more adjustable node behaviors not listed here, check out the :ref:`API Documentation <api_doc>`.