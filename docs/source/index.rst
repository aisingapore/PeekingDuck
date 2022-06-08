************
Introduction
************

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg
   :target: https://pypi.org/project/peekingduck

.. image:: https://badge.fury.io/py/peekingduck.svg
   :target: https://pypi.org/project/peekingduck

.. image:: https://img.shields.io/pypi/dm/peekingduck
   :target: https://pypi.org/project/peekingduck
    
.. image:: https://img.shields.io/badge/license-Apache%202.0-blue.svg
   :target: https://github.com/aimakerspace/PeekingDuck/blob/main/LICENSE


What is PeekingDuck?
====================

PeekingDuck is an open-source, modular framework in Python, built for Computer Vision (CV)
inference. The name "PeekingDuck" is a play on: "Peeking" in a nod to CV; and "Duck" in
`duck typing <https://en.wikipedia.org/wiki/Duck_typing>`_ used in Python.


Features
========


Build realtime CV pipelines
---------------------------

PeekingDuck enables you to build powerful CV pipelines with minimal lines of code.


Leverage on SOTA models
-----------------------

PeekingDuck comes with various state of the art (SOTA)
:doc:`object detection </resources/01a_object_detection>`,
:doc:`pose estimation </resources/01b_pose_estimation>`,
:doc:`object tracking <resources/01c_object_tracking>`, and
:doc:`crowd counting </resources/01d_crowd_counting>` models. Mix and match different nodes to
construct solutions for various :doc:`use cases </use_cases/index>`.


Create custom nodes
-------------------

You can create :ref:`custom nodes <tutorial_custom_nodes>` to meet your own project's requirements.
PeekingDuck can also be :doc:`imported as a library </tutorials/05_calling_peekingduck_in_python>`
to fit into your existing workflows.


.. _how_peekingduck_works:

How PeekingDuck Works
=====================

**Nodes** are the building blocks of PeekingDuck. Each node is a wrapper for a pipeline function, and
contains information on how other PeekingDuck nodes may interact with it.

PeekingDuck has 6 types of nodes:

.. image:: /assets/diagrams/node_types.drawio.svg

A **pipeline** governs the behavior of a chain of nodes. The diagram below shows a sample pipeline.
Nodes in a pipeline are called in sequential order, and the output of one
node will be the input to another. For example, :mod:`input.visual` produces :term:`img`, which is taken
in by :mod:`model.yolo`, and :mod:`model.yolo` produces :term:`bboxes`, which is taken in by
:mod:`draw.bbox`. For ease of visualization, not all the inputs and outputs of these nodes are
included in this diagram.

.. image:: /assets/diagrams/yolo_demo.drawio.svg


Acknowledgements
================

This project is supported by the National Research Foundation, Singapore under its AI Singapore
Programme (AISG-RP-2019-050). Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the author(s) and do not reflect the views of the National Research
Foundation, Singapore.


License
=======

PeekingDuck is under the open source `Apache License 2.0 <https://github.com/aimakerspace/PeekingDuck/blob/main/LICENSE>`_ (:

Even so, your organization may require legal proof of its right to use PeekingDuck, due to
circumstances such as the following:

* Your organization is using PeekingDuck in a jurisdiction that does not recognize this license
* Your legal department requires a license to be purchased
* Your organization wants to hold a tangible legal document as evidence of the legal right to use
  and distribute PeekingDuck

`Contact us <https://aisingapore.org/home/contact>`_ if any of these circumstances apply to you.


Communities
===========

* `AI Singapore community forum <https://community.aisingapore.org/groups/computer-vision/forum/>`_
* `Discuss on GitHub <https://github.com/aimakerspace/PeekingDuck/discussions>`_