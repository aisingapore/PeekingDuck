************
Custom Nodes
************

.. include:: /include/substitution.rst

.. _tutorial_custom_nodes:

This tutorial will show you how to create your own custom nodes to run with
PeekingDuck. 
Perhaps you'd like to take a snapshot of a video frame, and post it to your API
endpoint; 
or perhaps you have a model trained on a custom dataset, and would like to use
PeekingDuck's :ref:`input <api_doc>`, :ref:`draw <api_doc>`, and :ref:`output
<api_doc>` nodes. 
PeekingDuck is designed to be very flexible --- you can create your own nodes
and use them with ours.

Let's start by creating a new PeekingDuck project:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir custom_project` \ 
   | \ :blue:`[~user]` \ > \ :green:`cd custom_project` \ 
   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck init` \ 

This creates the following ``custom_project`` folder structure:

.. parsed-literal::

   \ :blue:`custom_project/` \ |Blank|
   ├── pipeline_config.yml
   └── \ :blue:`src/` \ |Blank|
       └── \ :blue:`custom_nodes/` \ |Blank|
           └── \ :blue:`configs/` \ |Blank|

The sub-folders ``src``, ``custom_nodes``, and ``configs`` are empty: they serve 
as placeholders for contents to be added.


.. _tutorial_object_detection_score:

Recipe 1: Object Detection Score
================================

When the Yolo object detection model detects an object in the image, it assigns 
a bounding box and a score to it.
This score is the "confidence score" which reflects how likely the box contains 
an object and how accurate is the bounding box.
It is a decimal number that ranges from 0.0 to 1.0 (or 100%).
This number is internal and not readily viewable.

We will create a custom node to retrieve this score and display it on screen.
This tutorial will use the `cat_and_computer.mp4  
<https://storage.googleapis.com/peekingduck/videos/cat_and_computer.mp4>`_ video from
the earlier :ref:`object detection tutorial <tutorial_more_object_detection>`.
Copy it into the ``custom_project`` folder.

Use the following command to create a custom node: :greenbox:`peekingduck create-node` |br|
It will prompt you to answer several questions.
Press :greenbox:`<Enter>` to accept the default ``custom_nodes`` folder name, then key
in :greenbox:`draw` for node type and :greenbox:`score` for node name.
Finally, press :greenbox:`<Enter>` to answer ``Y`` when asked to proceed.

The entire interaction is shown here, the answers you type are in shown in 
:green:`green text`:

.. _tutorial_wave_project_custom_node:

.. admonition:: Terminal Session

   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck create-node` \ 
   | Creating new custom node...
   | Enter node directory relative to ~user/custom_project [src/custom_nodes]: \ :green:`⏎` \
   | Select node type (input, augment, model, draw, dabble, output): \ :green:`draw` \
   | Enter node name [my_custom_node]: \ :green:`score` \
   | 
   | Node directory:	~user/custom_project/src/custom_nodes
   | Node type:	draw
   | Node name:	score
   | 
   | Creating the following files:
   |    Config file: ~user/custom_project/src/custom_nodes/configs/draw/score.yml
   |    Script file: ~user/custom_project/src/custom_nodes/draw/score.py
   | Proceed? [Y/n]: \ :green:`⏎` \
   | Created node!

The ``custom_project`` folder structure should look like this:

.. parsed-literal::

   \ :blue:`custom_project/` \ |Blank|
   ├── cat_and_computer.mp4
   ├── pipeline_config.yml
   └── \ :blue:`src/` \ |Blank|
       └── \ :blue:`custom_nodes/` \ |Blank|
           ├── \ :blue:`configs/` \ |Blank|
           │   └── \ :blue:`draw/` \ |Blank|
           │       └── score.yml
           └── \ :blue:`draw/` \ |Blank|
               └── score.py

``custom_project`` now contains **three files** that we need to modify to
implement our custom node function.

#. **src/custom_nodes/configs/draw/score.yml**:

   ``score.yml`` initial content:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      # Receive bounding boxes and their respective labels as input. Replace with
      # other data types as required. List of built-in data types for PeekingDuck can
      # be found at https://peekingduck.readthedocs.io/en/stable/glossary.html.
      input: ["bboxes", "bbox_labels"]
      # Example:
      # Output `obj_attrs` for visualization with `draw.tag` node and `custom_key` for
      # use with other custom nodes. Replace as required.
      output: ["obj_attrs", "custom_key"]

      # Optional configs depending on node
      threshold: 0.5 # example

   The first file ``score.yml`` defines the properties of the custom node. |br|
   Lines 5 and 9 show the mandatory configs ``input`` and ``output``.

   ``input`` specifies the data types the node would consume, to be read from the pipeline. |br|
   ``output`` specifies the data types the node would produce, to be put into the pipeline.

   To display the bounding box confidence score, our node requires three pieces of input
   data: the bounding box, the score to display, and the image to draw on.  These are
   defined as the data types :term:`bboxes`, :term:`bbox_scores`, and :term:`img`
   respectively in the :ref:`API docs <api_doc>`.

   Our custom node only displays the score on screen and does not produce any
   outputs for the pipeline, so the output is ":term:`none <(output) none>`".

   There are also no optional configs, so lines 11 - 12 can be removed.

   ``score.yml`` updated content:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["img", "bboxes", "bbox_scores"]
      output: ["none"]

      # No optional configs

   .. note::

      Comments in yaml files start with ``#`` |br|
      It is possible for a node to have ``input: ["none"]``

#. **src/custom_nodes/draw/score.py**:

   The second file ``score.py`` contains the boilerplate code for creating a
   custom node. Update the code to implement the desired behavior for the node.

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for score.py**

      .. code-block:: python
         :linenos:
   
         """
         Custom node to show object detection scores
         """
   
         from typing import Any, Dict, List, Tuple
         import cv2
         from peekingduck.pipeline.nodes.node import AbstractNode
   
         YELLOW = (0, 255, 255)        # in BGR format, per opencv's convention
   
   
         def map_bbox_to_image_coords(
            bbox: List[float], image_size: Tuple[int, int]
         ) -> List[int]:
            """This is a helper function to map bounding box coords (relative) to 
            image coords (absolute).
            Bounding box coords ranges from 0 to 1
            where (0, 0) = image top-left, (1, 1) = image bottom-right.
   
            Args:
               bbox (List[float]): List of 4 floats x1, y1, x2, y2
               image_size (Tuple[int, int]): Width, Height of image
   
            Returns:
               List[int]: x1, y1, x2, y2 in integer image coords
            """
            width, height = image_size[0], image_size[1]
            x1, y1, x2, y2 = bbox
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            return int(x1), int(y1), int(x2), int(y2)
   
   
         class Node(AbstractNode):
            """This is a template class of how to write a node for PeekingDuck,
               using AbstractNode as the parent class.
               This node draws scores on objects detected.

            Args:
               config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
            """
   
            def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
               """Node initializer

               Since we do not require any special setup, it only calls the __init__
               method of its parent class.
               """
               super().__init__(config, node_path=__name__, **kwargs)
   
            def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
               """This method implements the display score function. 
               As PeekingDuck iterates through the CV pipeline, this 'run' method 
               is called at each iteration.
   
               Args:
                     inputs (dict): Dictionary with keys "img", "bboxes", "bbox_scores"
   
               Returns:
                     outputs (dict): Empty dictionary
               """

               # extract pipeline inputs and compute image size in (width, height)
               img = inputs["img"]
               bboxes = inputs["bboxes"]
               scores = inputs["bbox_scores"]
               img_size = (img.shape[1], img.shape[0])  # width, height
   
               for i, bbox in enumerate(bboxes):
                  # for each bounding box:
                  #   - compute (x1, y1) left-top, (x2, y2) right-bottom coordinates
                  #   - convert score into a two decimal place numeric string
                  #   - draw score string onto image using opencv's putText()
                  #     (see opencv's API docs for more info)
                  x1, y1, x2, y2 = map_bbox_to_image_coords(bbox, img_size)
                  score = scores[i]
                  score_str = f"{score:0.2f}"
                  cv2.putText(
                     img=img,
                     text=score_str,
                     org=(x1, y2),
                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                     fontScale=1.0,
                     color=YELLOW,
                     thickness=3,
                  )
   
               return {}               # node has no outputs

   The updated node code defines a helper function ``map_bbox_to_image_coords`` to map
   the bounding box coordinates to the image coordinates, as explained in :ref:`this
   section <tutorial_coordinate_systems>`.

   The ``run`` method implements the main logic which processes every bounding box to 
   compute its on-screen coordinates and to draw the bounding box confidence score at 
   its left-bottom position.


#. **pipeline_config.yml**:

   ``pipeline_config.yml`` initial content:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.live
      - model.yolo
      - draw.bbox
      - output.screen

   This file implements the pipeline.  Modify the default pipeline to the one shown below:

   ``pipeline_config.yml`` updated content:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
          input_dir: cat_and_computer.mp4
      - model.yolo:
          detect_ids: ["cup", "cat", "laptop", "keyboard", "mouse"]
      - draw.bbox:
          show_labels: True
      - custom_nodes.draw.score
      - output.screen

   Line 8 adds our custom node into the pipeline where it will be ``run`` by 
   PeekingDuck during each pipeline iteration.

Execute :greenbox:`peekingduck run` to see your custom node in action.

   .. figure:: /assets/tutorials/ss_custom_nodes_1.png
      :width: 416
      :alt: Custom node screenshot - show object detection scores

      Custom Node Showing Object Detection Scores

   .. note::

      Royalty free video of cat and computer from: https://www.youtube.com/watch?v=-C1TEGZavko


.. _tutorial_count_hand_wave:

Recipe 2: Keypoints, Count Hand Waves
=====================================

This tutorial will create a custom node to analyze the skeletal keypoints of the
person from the `wave.mp4 <https://storage.googleapis.com/peekingduck/videos/wave.mp4>`_
video in the :ref:`pose estimation tutorial <tutorial_pose_estimation>` and to
count the number of times the person waves his hand.

The PoseNet pose estimation model outputs seventeen keypoints for the person 
corresponding to the different body parts as documented :ref:`here
<whole-body-keypoint-ids>`.
Each keypoint is a pair of ``(x, y)`` coordinates, where ``x`` and ``y`` are
real numbers ranging from 0.0 to 1.0 (using relative coordinates).

Starting with a newly initialized PeekingDuck folder, call :greenbox:`peekingduck
create-node` to create a new ``dabble.wave`` custom node as shown below:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir wave_project` \
   | \ :blue:`[~user]` \ > \ :green:`cd wave_project` \
   | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck init` \
   | Welcome to PeekingDuck! 
   | 2022-02-11 18:17:31 peekingduck.cli  INFO:  Creating custom nodes folder in ~user/wave_project/src/custom_nodes 
   | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck create-node` \ 
   | Creating new custom node...
   | Enter node directory relative to ~user/wave_project [src/custom_nodes]: \ :green:`⏎` \
   | Select node type (input, augment, model, draw, dabble, output): \ :green:`dabble` \
   | Enter node name [my_custom_node]: \ :green:`wave` \
   | 
   | Node directory:	~user/wave_project/src/custom_nodes
   | Node type:	dabble
   | Node name:	wave
   | 
   | Creating the following files:
   |    Config file: ~user/wave_project/src/custom_nodes/configs/dabble/wave.yml
   |    Script file: ~user/wave_project/src/custom_nodes/dabble/wave.py
   | Proceed? [Y/n]: \ :green:`⏎` \
   | Created node!


Also, copy `wave.mp4 <https://storage.googleapis.com/peekingduck/videos/wave.mp4>`_ into
the above folder.  You should end up with the following folder structure:

.. parsed-literal::

   \ :blue:`wave_project/` \ |Blank|
   ├── pipeline_config.yml
   ├── \ :blue:`src/` \ |Blank|
   │   └── \ :blue:`custom_nodes/` \ |Blank|
   │       ├── \ :blue:`configs/` \ |Blank|
   │       │   └── \ :blue:`dabble/` \ |Blank|
   │       │       └── wave.yml
   │       └── \ :blue:`dabble/` \ |Blank|
   │           └── wave.py
   └── wave.mp4

To implement this tutorial, the **three files** ``wave.yml``, ``wave.py`` and
``pipeline_config.yml`` are to be edited as follows:

1. **src/custom_nodes/configs/dabble/wave.yml**:

   .. code-block:: yaml
      :linenos:

      # Dabble node has both input and output
      input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]
      output: ["none"]

      # No optional configs

We will implement this tutorial using a custom :mod:`dabble` node, which will take the
inputs :term:`img`, :term:`bboxes`, :term:`bbox_scores`, :term:`keypoints`, and
:term:`keypoint_scores` from the pipeline. The node has no output.

2. **src/custom_nodes/dabble/wave.py**:

   The ``dabble.wave`` code structure is similar to the ``draw.score`` code structure in
   the other custom node tutorial.

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for wave.py**

      .. code-block:: python
         :linenos:
   
         """
         Custom node to show keypoints and count the number of times the person's hand is waved
         """
   
         from typing import Any, Dict, List, Tuple
         import cv2
         from peekingduck.pipeline.nodes.node import AbstractNode
   
         # setup global constants
         FONT = cv2.FONT_HERSHEY_SIMPLEX
         WHITE = (255, 255, 255)       # opencv loads file in BGR format
         YELLOW = (0, 255, 255)
         THRESHOLD = 0.6               # ignore keypoints below this threshold
         KP_RIGHT_SHOULDER = 6         # PoseNet's skeletal keypoints
         KP_RIGHT_WRIST = 10
   
   
         def map_bbox_to_image_coords(
            bbox: List[float], image_size: Tuple[int, int]
         ) -> List[int]:
            """First helper function to convert relative bounding box coordinates to
            absolute image coordinates.
            Bounding box coords ranges from 0 to 1
            where (0, 0) = image top-left, (1, 1) = image bottom-right.
   
            Args:
               bbox (List[float]): List of 4 floats x1, y1, x2, y2
               image_size (Tuple[int, int]): Width, Height of image
   
            Returns:
               List[int]: x1, y1, x2, y2 in integer image coords
            """
            width, height = image_size[0], image_size[1]
            x1, y1, x2, y2 = bbox
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
            return int(x1), int(y1), int(x2), int(y2)
   
   
         def map_keypoint_to_image_coords(
            keypoint: List[float], image_size: Tuple[int, int]
         ) -> List[int]:
            """Second helper function to convert relative keypoint coordinates to
            absolute image coordinates.
            Keypoint coords ranges from 0 to 1
            where (0, 0) = image top-left, (1, 1) = image bottom-right.
   
            Args:
               bbox (List[float]): List of 2 floats x, y (relative)
               image_size (Tuple[int, int]): Width, Height of image
   
            Returns:
               List[int]: x, y in integer image coords
            """
            width, height = image_size[0], image_size[1]
            x, y = keypoint
            x *= width
            y *= height
            return int(x), int(y)
   
   
         def draw_text(img, x, y, text_str: str, color_code):
            """Helper function to call opencv's drawing function, 
            to improve code readability in node's run() method.
            """
            cv2.putText(
               img=img,
               text=text_str,
               org=(x, y),
               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
               fontScale=0.4,
               color=color_code,
               thickness=2,
            )
   
   
         class Node(AbstractNode):
            """Custom node to display keypoints and count number of hand waves
   
            Args:
               config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
            """
   
            def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
               super().__init__(config, node_path=__name__, **kwargs)
               # setup object working variables
               self.right_wrist = None
               self.direction = None
               self.num_direction_changes = 0
               self.num_waves = 0
   
            def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
               """This node draws keypoints and count hand waves.
   
               Args:
                     inputs (dict): Dictionary with keys
                        "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".
   
               Returns:
                     outputs (dict): Empty dictionary.
               """
   
               # get required inputs from pipeline
               img = inputs["img"]
               bboxes = inputs["bboxes"]
               bbox_scores = inputs["bbox_scores"]
               keypoints = inputs["keypoints"]
               keypoint_scores = inputs["keypoint_scores"]
   
               img_size = (img.shape[1], img.shape[0])  # image width, height
   
               # get bounding box confidence score and draw it at the 
               # left-bottom (x1, y2) corner of the bounding box (offset by 30 pixels)
               the_bbox = bboxes[0]             # image only has one person
               the_bbox_score = bbox_scores[0]  # only one set of scores
   
               x1, y1, x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
               score_str = f"BBox {the_bbox_score:0.2f}"
               cv2.putText(
                  img=img,
                  text=score_str,
                  org=(x1, y2 - 30),            # offset by 30 pixels
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=1.0,
                  color=WHITE,
                  thickness=3,
               )
   
               # hand wave detection using a simple heuristic of tracking the 
               # right wrist movement
               the_keypoints = keypoints[0]              # image only has one person
               the_keypoint_scores = keypoint_scores[0]  # only one set of scores
               right_wrist = None
               right_shoulder = None
   
               for i, keypoints in enumerate(the_keypoints):
                  keypoint_score = the_keypoint_scores[i]

                  if keypoint_score >= THRESHOLD:
                     x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
                     x_y_str = f"({x}, {y})"

                     if i == KP_RIGHT_SHOULDER:
                        right_shoulder = keypoints
                        the_color = YELLOW
                     elif i == KP_RIGHT_WRIST:
                        right_wrist = keypoints
                        the_color = YELLOW
                     else:                   # generic keypoint
                        the_color = WHITE

                     draw_text(img, x, y, x_y_str, the_color)
   
               if right_wrist is not None and right_shoulder is not None:
                  # only count number of hand waves after we have gotten the 
                  # skeletal poses for the right wrist and right shoulder
                  if self.right_wrist is None:
                     self.right_wrist = right_wrist            # first wrist data point
                  else:
                     # wait for wrist to be above shoulder to count hand wave
                     if right_wrist[1] > right_shoulder[1]:
                        pass
                     else:
                        if right_wrist[0] < self.right_wrist[0]:
                           direction = "left"
                        else:
                           direction = "right"

                        if self.direction is None:
                           self.direction = direction          # first direction data point
                        else:
                           # check if hand changes direction
                           if direction != self.direction:
                              self.num_direction_changes += 1
                           # every two hand direction changes == one wave
                           if self.num_direction_changes >= 2:
                              self.num_waves += 1
                              self.num_direction_changes = 0   # reset direction count

                        self.right_wrist = right_wrist         # save last position
                        self.direction = direction
   
                  wave_str = f"#waves = {self.num_waves}"
                  draw_text(img, 20, 30, wave_str, YELLOW)
   
               return {}

   This (long) piece of code implements our custom :mod:`dabble` node. 
   It defines three helper functions to convert relative to absolute coordinates 
   and to draw text on-screen.
   The number of hand waves is displayed at the left-top corner of the screen.

   A simple heuristic is used to count the number of times the person waves his hand. 
   It tracks the direction the right wrist is moving in and notes when the wrist changes
   direction. 
   Upon encountering two direction changes, e.g., left -> right -> left, one wave is
   counted.

   The heuristic also waits until the right wrist has been lifted above the right 
   should before it starts tracking hand direction and counting waves.

3. **pipeline_config.yml**:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
          input_dir: wave.mp4
      - model.yolo
      - model.posenet
      - dabble.fps
      - custom_nodes.dabble.wave
      - draw.poses
      - draw.legend:
          show: ["fps"]
      - output.screen

   We modify ``pipeline_config.yml`` to run both the object detection and pose estimation
   models to obtain the required inputs for our custom :mod:`dabble` node.

Execute :greenbox:`peekingduck run` to see your custom node in action.

   .. figure:: /assets/tutorials/ss_custom_nodes_2.png
      :width: 389
      :alt: Custom node screenshot - count hand waves

      Custom Node Counting Hand Waves

   .. note::

      Royalty free video of man waving from: https://www.youtube.com/watch?v=IKj_z2hgYUM


.. _tutorial_debugging:

Recipe 3: Debugging
===================

When working with PeekingDuck's pipeline, you may sometimes wonder what is available in
the :ref:`data pool <tutorial_pipeline_data_pool>`, or whether a particular data object
has been correctly computed.
This tutorial will show you how to use a custom node to help with troubleshooting and 
debugging PeekingDuck's pipeline.

Continuing from the above tutorial, create a new ``dabble.debug`` custom node:

.. admonition:: Terminal Session

   | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck create-node` \ 
   | Creating new custom node...
   | Enter node directory relative to ~user/wave_project [src/custom_nodes]: \ :green:`⏎` \
   | Select node type (input, augment, model, draw, dabble, output): \ :green:`dabble` \
   | Enter node name [my_custom_node]: \ :green:`debug` \
   | 
   | Node directory:	~user/wave_project/src/custom_nodes
   | Node type:	dabble
   | Node name:	debug
   | 
   | Creating the following files:
   |    Config file: ~user/wave_project/src/custom_nodes/configs/dabble/debug.yml
   |    Script file: ~user/wave_project/src/custom_nodes/dabble/debug.py
   | Proceed? [Y/n]: \ :green:`⏎` \
   | Created node!

The updated folder structure is:

.. parsed-literal::

   \ :blue:`wave_project/` \ |Blank|
   ├── pipeline_config.yml
   ├── \ :blue:`src` \ |Blank|
   │   └── \ :blue:`custom_nodes` \ |Blank|
   │       ├── \ :blue:`configs` \ |Blank|
   │       │   └── \ :blue:`dabble` \ |Blank|
   │       │       ├── debug.yml
   │       │       └── wave.yml
   │       └── \ :blue:`dabble` \ |Blank|
   │           ├── debug.py
   │           └── wave.py
   └── wave.mp4

Then, make the following **three** changes:

1. Specify ``debug.yml`` to receive everything ":term:`all <(input) all>`" from
   the pipeline, as follows:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["all"]
      output: ["none"]

      # No optional configs

2. Update ``debug.py`` as shown below:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for debug.py**

      .. code-block:: python
         :linenos:

         """
         A custom node for debugging
         """

         from typing import Any, Dict

         from peekingduck.pipeline.nodes.node import AbstractNode


         class Node(AbstractNode):
            """This is a simple example of creating a custom node to help with debugging.

            Args:
               config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
            """

            def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
               super().__init__(config, node_path=__name__, **kwargs)

            def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
               """A simple debugging custom node

               Args:
                     inputs (dict): "all", to view everything in data pool

               Returns:
                     outputs (dict): "none"
               """

               self.logger.info("-- debug --")
               # show what is available in PeekingDuck's data pool
               self.logger.info(f"input.keys={list(inputs.keys())}")
               # debug specific data: bboxes
               bboxes = inputs["bboxes"]
               bbox_labels = inputs["bbox_labels"]
               bbox_scores = inputs["bbox_scores"]
               self.logger.info(f"num bboxes={len(bboxes)}")
               for i, bbox in enumerate(bboxes):
                     label, score = bbox_labels[i], bbox_scores[i]
                     self.logger.info(f"bbox {i}:")
                     self.logger.info(f"  label={label}, score={score:0.2f}")
                     self.logger.info(f"  coords={bbox}")

               return {}  # no outputs

   The custom node code shows how to see what is available in PeekingDuck's pipeline 
   data pool by printing the input dictionary keys.
   It also demonstrates how to debug a specific data object, such as :term:`bboxes`, by
   printing relevant information for each item within the data.

3. Update **pipeline_config.yml**:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
          input_dir: wave.mp4
      - model.yolo
      - model.posenet
      - dabble.fps
      - custom_nodes.dabble.wave
      - custom_nodes.dabble.debug
      - draw.poses
      - draw.legend:
          show: ["fps"]
      - output.screen

Now, do a :greenbox:`peekingduck run` and you should see a sample debug output like the
one below:

.. admonition:: Terminal Session

   | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck run` \ 
   | 2022-03-02 18:42:51 peekingduck.declarative_loader  INFO:  Successfully loaded pipeline_config file. 
   | 2022-03-02 18:42:51 peekingduck.declarative_loader  INFO:  Initialising input.recorded node.\.\. 
   | 2022-03-02 18:42:51 peekingduck.declarative_loader  INFO:  Config for node input.recorded is updated to: 'input_dir': wave.mp4 
   | 2022-03-02 18:42:51 peekingduck.pipeline.nodes.input.recorded  INFO:  Video/Image size: 710 by 540 
   | 2022-03-02 18:42:51 peekingduck.pipeline.nodes.input.recorded  INFO:  Filepath used: wave.mp4 
   | 2022-03-02 18:42:51 peekingduck.declarative_loader  INFO:  Initialising model.yolo node.\.\. 
   |                     [ .\.\. many lines of output deleted here .\.\. ]
   | 2022-03-02 18:42:53 peekingduck.declarative_loader  INFO:  Initialising custom_nodes.dabble.debug node.\.\. 
   | 2022-03-02 18:42:53 peekingduck.declarative_loader  INFO:  Initialising draw.poses node.\.\. 
   | 2022-03-02 18:42:53 peekingduck.declarative_loader  INFO:  Initialising draw.legend node.\.\. 
   | 2022-03-02 18:42:53 peekingduck.declarative_loader  INFO:  Initialising output.screen node.\.\. 
   | 2022-03-02 18:42:55 custom_nodes.dabble.debug  INFO:  -- debug -- 
   | 2022-03-02 18:42:55 custom_nodes.dabble.debug  INFO:  input.keys=['img', 'pipeline_end', 'filename', 'saved_video_fps', 'bboxes', 'bbox_labels', 'bbox_scores', 'keypoints', 'keypoint_scores', 'keypoint_conns', 'hand_direction', 'num_waves', 'fps'] 
   | 2022-03-02 18:42:55 custom_nodes.dabble.debug  INFO:  num bboxes=1 
   | 2022-03-02 18:42:55 custom_nodes.dabble.debug  INFO:  bbox 0: 
   | 2022-03-02 18:42:55 custom_nodes.dabble.debug  INFO:  |nbsp| |nbsp| |nbsp| label=Person, score=0.91 
   | 2022-03-02 18:42:55 custom_nodes.dabble.debug  INFO:  |nbsp| |nbsp| |nbsp| coords=[0.40047657 0.21553655 0.85199741 1.02150181] 


Other Recipes to Create Custom Nodes
====================================

This section describes two faster ways to create custom nodes for users who are familiar
with PeekingDuck.


CLI Recipe
----------

You skip the step-by-step prompts from :green:`peekingduck create-node` by specifying all the
options on the command line, for instance:

.. admonition:: Terminal Session

   | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck create-node -\-node_subdir src/custom_nodes -\-node_type dabble -\-node_name wave` \

The above is the equivalent of the tutorial *Recipe 1: Object Detection Score*
:ref:`custom node creation <tutorial_wave_project_custom_node>`.
For more information, see :green:`peekingduck create-node --help`.


Pipeline Recipe
---------------

PeekingDuck can also create custom nodes by parsing your pipeline configuration file.
Starting with the basic folder structure from :green:`peekingduck init`:

   .. parsed-literal::

      \ :blue:`wave_project/` \ |Blank|
      ├── pipeline_config.yml
      ├── \ :blue:`src` \ |Blank|
      │   └── \ :blue:`custom_nodes` \ |Blank|
      │       └── \ :blue:`configs` \ |Blank|
      └── wave.mp4
 
 and the following modified ``pipeline_config.yml`` file:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
          input_dir: wave.mp4
      - model.yolo
      - model.posenet
      - dabble.fps
      - custom_nodes.dabble.wave
      - custom_nodes.dabble.debug
      - draw.poses
      - draw.legend:
          show: ["fps"]
      - output.screen

You can tell PeekingDuck to parse your pipeline file with :green:`peekingduck create-node --config_path pipeline_config.yml`:

   .. admonition:: Terminal Session

      | \ :blue:`[~user/wave_project]` \ > \ :green:`peekingduck create-node --config_path pipeline_config.yml` \
      | 2022-03-14 11:21:21 peekingduck.cli  INFO:  Creating custom nodes declared in ~user/wave_project/pipeline_config.yml. 
      | 2022-03-14 11:21:21 peekingduck.declarative_loader  INFO:  Successfully loaded pipeline file. 
      | 2022-03-14 11:21:21 peekingduck.cli  INFO:  Creating files for custom_nodes.dabble.wave:
      |         Config file: ~user/wave_project/src/custom_nodes/configs/dabble/wave.yml
      |         Script file: ~user/wave_project/src/custom_nodes/dabble/wave.py 
      | 2022-03-14 11:21:21 peekingduck.cli  INFO:  Creating files for custom_nodes.dabble.debug:
      |         Config file: ~user/wave_project/src/custom_nodes/configs/dabble/debug.yml
      |         Script file: ~user/wave_project/src/custom_nodes/dabble/debug.py 

PeekingDuck will read ``pipeline_config.yml`` and create the two specified custom nodes 
``custom_nodes.dabble.wave`` and ``custom_nodes.dabble.debug``.
Your folder structure will now look like this:

   .. parsed-literal::

      \ :blue:`wave_project/` \ |Blank|
      ├── pipeline_config.yml
      ├── \ :blue:`src` \ |Blank|
      │   └── \ :blue:`custom_nodes` \ |Blank|
      │       ├── \ :blue:`configs` \ |Blank|
      │       │   └── \ :blue:`dabble` \ |Blank|
      │       │       ├── debug.yml
      │       │       └── wave.yml
      │       └── \ :blue:`dabble` \ |Blank|
      │           ├── debug.py
      │           └── wave.py
      └── wave.mp4

From here, you can proceed to edit the custom node configs and source files.

