***********
Duck Confit
***********

.. |br| raw:: html

   <br />

.. role:: red

.. role:: blue

.. role:: green

This tutorial presents intermediate recipes for cooking up new PeekingDuck
pipelines by modifying the nodes and their configs.
It will also show how to create custom nodes to implement custom user functions.


Nodes and Configs
=================

PeekingDuck comes with a rich collection of nodes that you can use to create
your own CV pipelines. Each node can be customised by changing its
configurations or settings.

To get a quick overview of PeekingDuck's nodes, run the following command::

   ~ > peekingduck nodes

.. image:: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/tutorials/ss_pkd_nodes.png
   :width: 1575
   :height: 360
   :alt: PeekingDuck screenshot : nodes output

You will see a comprehensive list of all PeekingDuck's nodes with links to their
``readthedocs`` pages for more information.


Pipelines, Nodes, Configs
-------------------------

PeekingDuck supports 6 types of nodes:

+------------+-----------------------------------------------------------------+
| Node Type  | Node Description                                                |
+------------+-----------------------------------------------------------------+
| Input      | Reads a video file from disk or captures images from the webcam |
+------------+-----------------------------------------------------------------+
| Model      | CV model does the "heaving lifting" here, like object detection |
+------------+-----------------------------------------------------------------+
| Dabble     | Does the "smaller" computations, like counting number of bboxes |
+------------+-----------------------------------------------------------------+
| Draw       | Draws things/text onto an image, like bboxes or FPS             |
+------------+-----------------------------------------------------------------+
| Output     | Shows an image on screen or saves to a video file on disk       |
+------------+-----------------------------------------------------------------+
| Preprocess | Applies effects onto an image                                   |
+------------+-----------------------------------------------------------------+

A PeekingDuck pipeline is created by stringing together a series of nodes that 
perform a logical sequence of operations.
Each node has its own set of configurable settings that can be modified to
change its behaviour.


.. _configure_nodes:

More Object Detection
---------------------

This section will demonstrate how to change the settings of PeekingDuck's nodes 
to vary their functionalities.

To follow this guide, first download this `demo video
<http://orchard.dnsalias.com:8100/computers_800.mp4>`_
**([Todo]: update to GCP URL)** and save it in a local folder.

Next, create a PeekingDuck project as shown :ref:`earlier <verify_installation>`.

We will modify this project to perform object detection on the ``demo video``.
Edit the ``run_config.yml`` file as follows:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
         input_dir: /folder/containing/demo_video.mp4    # replace this with actual path
      - model.yolo:
         detect_ids: ["cup", "cat", "laptop", "keyboard", "mouse"]
      - draw.bbox:
         show_labels: True
      - output.screen

Here is an explanation of what has been done above:

   2. ``input.recorded``: tells PeekingDuck to load the ``demo video``. |br|
   4. ``model.yolo``: by default, the Yolo model detects ``person`` only.
   The ``demo video`` contains other classes of objects like cup, cat, laptop, etc. 
   So we have to change the model settings to detect the other object classes. |br|
   6. ``draw.bbox``: reconfigure this node to display the detected object class label.

   .. note::
      The Yolo model can detect 80 different :ref:`object classes
      <general-object-detection-ids>`.

Run the above with ``peekingduck run``. |br|
You should see a display of the ``demo video`` with the various objects being
highlighted by PeekingDuck in bounding boxes. 
The 30-second video will auto-close at the end, or you can press ``q`` to end early.


Record and Save Video File with FPS
-----------------------------------

This section demonstrates how to record PeekingDuck's output into a video file. |br|
In addition, we will modify the pipeline by adding new nodes to calculate the
frames per second (FPS) and to show the FPS.

Edit ``run_config.yml`` and *add the four new lines* as shown here:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
         input_dir: /folder/containing/demo_video.mp4    # replace this with actual path
      - model.yolo:
         detect_ids: ["cup", "cat", "laptop", "keyboard", "mouse"]
      - draw.bbox:
         show_labels: True
      - dabble.fps                           # line 1: add new dabble node
      - draw.legend                          # line 2: show fps
      - output.screen
      - output.media_writer:                 # line 3: add new output node
         output_dir: /folder/to/save/video   # line 4: this is a folder name

The additions are explained below:

#. ``dabble.fps``: adds new ``dabble`` node to the pipeline. 
   This node calculates the FPS.

#. ``draw.legend``: adds new ``draw`` node to display the FPS.

#. ``output.media_writer``: adds new ``output`` node to save PeekingDuck's
   output to a local video file. It requires a local folder path. If the folder
   is not available, PeekingDuck will create the folder automatically. The
   filename is auto-generated by PeekingDuck based on the input source.

Run the above with ``peekingduck run``. |br|
You will see the same video being played, but now it has the FPS counter.
When the video ends, an ``mp4`` video file will be created and saved in the
specified folder.


   .. note::
      You can view all the available nodes and their respective configurable
      settings in PeekingDuck's :ref:`API documentation <api_doc>`.



.. _create_custom_nodes:

Custom Nodes
============

This tutorial will show you how to create your own custom nodes to run with
PeekingDuck. 
Perhaps you'd like to take a snapshot of a video frame, and post it to your API
endpoint; 
or perhaps you have a model trained on a custom dataset, and would like to use
PeekingDuck's :ref:`input <api_doc>`, :ref:`draw <api_doc>`, and :ref:`output
<api_doc>` nodes. 
PeekingDuck is designed to be very flexible --- you can create your own nodes
and use them with ours.

Let's start by creating a new PeekingDuck project::

    ~ > mkdir custom_project
    ~ > cd custom_project
    ~/custom_project > peekingduck init

This creates the following ``custom_project`` folder structure::

   custom_project/
   ├── run_config.yml
   └── src/
      └── custom_nodes/
         └── configs/

The sub-folders ``src``, ``custom_nodes`` and ``configs`` are empty: they serve 
as placeholders for contents to be added.


Custom Node to Show Object Detection Score
------------------------------------------

When the Yolo object detection model detects an object in the image, it assigns 
a bounding box and a score to it.
This score is the "confidence score" which reflects how likely the box contains 
an object and how accurate is the bounding box.
It is a decimal number that ranges from 0.0 to 1.0 (or 100%).
This number is internal and not readily viewable.

We will create a custom node to retrieve this score and display it on screen.

Use the following command to create a custom node::

    ~/custom_project > peekingduck create-node

It will prompt you to answer several questions.
Press ``<enter>`` to accept the default ``custom_nodes`` folder name, then enter 
``draw`` for node type and ``score`` for node name.
Finally, press ``<enter>`` to answer ``Y`` when asked to proceed.
The entire interaction is shown here::

    ~/custom_project > peekingduck create-node
   Creating new custom node...
   Enter node directory relative to ~/custom_project [src/custom_nodes]: 
   Select node type (input, model, draw, dabble, output): draw
   Enter node name [my_custom_node]: score

   Node directory:	~/custom_project/src/custom_nodes
   Node type:	draw
   Node name:	score

   Creating the following files:
      Config file: ~/custom_project/src/custom_nodes/configs/draw/score.yml
      Script file: ~/custom_project/src/custom_nodes/draw/score.py
   Proceed? [Y/n]: 
   Created node!
 
This will update the ``custom_project`` folder structure to become like this::

   custom_project/
   ├── run_config.yml
   └── src/
      └── custom_nodes/
         ├── configs/
         │   └── draw/
         │       └── score.yml
         └── draw/
               └── score.py

``custom_project`` now contains three files that we need to modify to implement 
our custom function to display the score:

1. **run_config.yml** (default content):

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.live
      - model.yolo
      - draw.bbox
      - output.screen

2. **src/custom_nodes/configs/draw/score.yml** (default content):

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["in1", "in2"]             # replace values
      output: ["out1", "out2", "out3"]  # replace values

      # Optional configs depending on node
      threshold: 0.5                    # example

3. **src/custom_nodes/draw/score.py** (default content):

   .. code-block:: python
      :linenos:

      """
      Node template for creating custom nodes.
      """

      from typing import Any, Dict

      from peekingduck.pipeline.nodes.node import AbstractNode


      class Node(AbstractNode):
         """This is a template class of how to write a node for PeekingDuck.

         Args:
            config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
         """

         def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
            super().__init__(config, node_path=__name__, **kwargs)

            # initialize/load any configs and models here
            # configs can be called by self.<config_name> e.g. self.filepath
            # self.logger.info(f"model loaded with configs: config")

         def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            """This node does ___.

            Args:
                  inputs (dict): Dictionary with keys "__", "__".

            Returns:
                  outputs (dict): Dictionary with keys "__".
            """

            # result = do_something(inputs["in1"], inputs["in2"])
            # outputs = {"out1": result}
            # return outputs






``Todo`` tutorial on drawing bbox coords and confidence score



Temp Placeholders
=================

Royalty free video of computer hardware from:
https://www.youtube.com/watch?v=-C1TEGZavko





