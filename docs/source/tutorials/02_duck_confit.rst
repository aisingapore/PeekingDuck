***********
Duck Confit
***********

.. |br| raw:: html

   <br />

.. role:: red

.. role:: blue

.. role:: green

.. |Blank| unicode:: U+2800 .. Invisible character

.. |nbsp| unicode:: U+00A0 .. Non-breaking space
   :trim:

This tutorial presents intermediate recipes for cooking up new PeekingDuck
pipelines by modifying the nodes and their configs.
It will also show how to create custom nodes to implement custom user functions.


.. _nodes_config_intro:

Nodes and Configs
=================

PeekingDuck comes with a rich collection of nodes that you can use to create
your own CV pipelines. Each node can be customized by changing its
configurations or settings.

To get a quick overview of PeekingDuck's nodes, run the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`peekingduck nodes` \


.. url: https://raw.githubusercontent.com/aimakerspace/PeekingDuck/dev/images/tutorials/ss_pkd_nodes.png
.. image:: /assets/tutorials/ss_pkd_nodes.png
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
change its behavior.


.. _configure_nodes:

More Object Detection
---------------------

This section will demonstrate how to change the settings of PeekingDuck's nodes 
to vary their functionalities.

First, create a PeekingDuck project as shown below:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`mkdir pkd_project` \
    | \ :blue:`[~user]` \ > \ :green:`cd pkd_project` \
    | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck init` \

Then, download this `demo video
<http://orchard.dnsalias.com:8100/computers_800.mp4>`_
(**TODO: to update demo_video.mp4 link to GCP URL**)
and save it into the created ``pkd_project`` folder.

The folder should contain the following:

.. parsed-literal::

   \ :blue:`pkd_project` \ |Blank|
   ├── run_config.yml
   ├── \ :blue:`src/` \ |Blank|
   └── demo_video.mp4

To perform object detection on the ``demo video.mp4`` file.  edit the
``run_config.yml`` file as follows:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
         input_dir: demo_video.mp4
      - model.yolo:
         detect_ids: ["cup", "cat", "laptop", "keyboard", "mouse"]
      - draw.bbox:
         show_labels: True
      - output.screen

Here is an explanation of what has been done above:

   | Line 2 ``input.recorded``: tells PeekingDuck to load the ``demo video``.
   | Line 4 ``model.yolo``: by default, the Yolo model detects ``person`` only.
   |        The ``demo video`` contains other classes of objects like cup, cat, laptop, etc. 
   |        So we have to change the model settings to detect the other object classes.
   | Line 6 ``draw.bbox``: reconfigure this node to display the detected object class label.

   .. note::
      The Yolo model can detect 80 different :ref:`object classes
      <general-object-detection-ids>`.

Run the above with the command ``peekingduck run``. |br|
You should see a display of the ``demo video`` with the various objects being
highlighted by PeekingDuck in bounding boxes. |br|
The 30-second video will auto-close at the end, or you can press ``q`` to end early.


Record and Save Video File with FPS
-----------------------------------

This section demonstrates how to record PeekingDuck's output into a video file.
In addition, we will modify the pipeline by adding new nodes to calculate the
frames per second (FPS) and to show the FPS.

Edit ``run_config.yml`` as shown below:

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

The additions are:

   | Line 8 ``dabble.fps``: adds new ``dabble`` node to the pipeline. This node calculates the FPS.
   | Line 9 ``draw.legend``: adds new ``draw`` node to display the FPS.
   | Line 11 ``output.media_writer``: adds new ``output`` node to save PeekingDuck's
            output to a local video file. It requires a local folder path. If the folder
            is not available, PeekingDuck will create the folder automatically. The
            filename is auto-generated by PeekingDuck based on the input source.

Run the above with the command ``peekingduck run``. |br|
You will see the same video being played, but now it has the FPS counter.
When the video ends, an ``mp4`` video file will be created and saved in the
specified folder.


   .. note::
      You can view all the available nodes and their respective configurable
      settings in PeekingDuck's :ref:`API documentation <api_doc>`.



.. _coordinate_systems:

Bounding Box vs Image Coordinates
=================================

PeekingDuck has two coordinate systems, with top-left corner as origin (0, 0):

   .. figure:: /assets/tutorials/bbox_image_coords.png
      :alt: Image vs Bounding Box Coordinates

      PeekingDuck's Image vs Bounding Box Coordinates

* Absolute image coordinates
   For an image of width W and height H, the absolute image coordinates are 
   integers from (0, |nbsp| 0) to (W-1, |nbsp| H-1). |br|
   E.g. For a 720 x 480 image, the absolute coordinates range from 
   (0, |nbsp| 0) to (719, |nbsp| 479)

* Relative bounding box coordinates
   For an image of width W and height H, the relative image coordinates are 
   real numbers from (0.0, |nbsp| 0.0) to (1.0, |nbsp| 1.0). |br|
   E.g. For a 720 x 480 image, the relative coordinates range from 
   (0.0, |nbsp| 0.0) to (1.0, |nbsp| 1.0)

This means that in order to draw a bounding box onto an image, the bounding box 
relative coordinates would have to be converted to the image absolute coordinates.

Using the above figure as an illustration, the bounding box coordinates are
given as ( 0.18, 0.10 ) left-top and ( 0.52, 0.88 ) right-bottom.
To convert them to image coordinates, multiply the x-coordinates by the image 
width and the y-coordinates by the image height, and round the results into 
integers.

.. math::

   0.18 -> 0.18 * 720 = 129.6 = 130 \: (int) 

   0.10 -> 0.10 * 720 = 72.0 = 72 \: (int)

.. math::

   0.52 -> 0.52 * 720 = 374.4 = 374 \: (int) 
   
   0.88 -> 0.88 * 720 = 633.6 = 634 \: (int)

Thus, the image coordinates are ( 130, 72 ) left-top and ( 374, 634 ) right-bottom.

   .. note::
      The ``model`` nodes return results in relative coordinates.



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

Let's start by creating a new PeekingDuck project:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir custom_project` \ 
   | \ :blue:`[~user]` \ > \ :green:`cd custom_project` \ 
   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck init` \ 


This creates the following ``custom_project`` folder structure:

.. parsed-literal::

   \ :blue:`custom_project/` \ |Blank|
   ├── run_config.yml
   └── \ :blue:`src/` \ |Blank|
       └── \ :blue:`custom_nodes/` \ |Blank|
           └── \ :blue:`configs/` \ |Blank|


The sub-folders ``src``, ``custom_nodes`` and ``configs`` are empty: they serve 
as placeholders for contents to be added.


Custom Node 1: Show Object Detection Score
------------------------------------------

When the Yolo object detection model detects an object in the image, it assigns 
a bounding box and a score to it.
This score is the "confidence score" which reflects how likely the box contains 
an object and how accurate is the bounding box.
It is a decimal number that ranges from 0.0 to 1.0 (or 100%).
This number is internal and not readily viewable.

We will create a custom node to retrieve this score and display it on screen.

Use the following command to create a custom node: ``peekingduck create-node``

It will prompt you to answer several questions.
Press ``<enter>`` to accept the default ``custom_nodes`` folder name, then enter 
``draw`` for node type and ``score`` for node name.
Finally, press ``<enter>`` to answer ``Y`` when asked to proceed.

The entire interaction is shown here, the answers you type are in shown in 
:green:`green text`:


.. admonition:: Terminal Session

   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck create-node` \ 
   | Creating new custom node...
   | Enter node directory relative to ~user/custom_project [src/custom_nodes]: ⏎
   | Select node type (input, model, draw, dabble, output): \ :green:`draw` \
   | Enter node name [my_custom_node]: \ :green:`score` \
   | 
   | Node directory:	~user/custom_project/src/custom_nodes
   | Node type:	draw
   | Node name:	score
   | 
   | Creating the following files:
   |    Config file: ~user/custom_project/src/custom_nodes/configs/draw/score.yml
   |    Script file: ~user/custom_project/src/custom_nodes/draw/score.py
   | Proceed? [Y/n]: ⏎
   | Created node!


This will update the ``custom_project`` folder structure to become like this:

.. parsed-literal::

   \ :blue:`custom_project/` \ |Blank|
   ├── run_config.yml
   └── \ :blue:`src/` \ |Blank|
       └── \ :blue:`custom_nodes/` \ |Blank|
           ├── \ :blue:`configs/` \ |Blank|
           │   └── \ :blue:`draw/` \ |Blank|
           │       └── score.yml
           └── \ :blue:`draw/` \ |Blank|
               └── score.py

``custom_project`` now contains **three files** that we need to modify to
implement our custom node function.

1. **src/custom_nodes/configs/draw/score.yml** (default content):

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["in1", "in2"]             # replace values
      output: ["out1", "out2", "out3"]  # replace values

      # Optional configs depending on node
      threshold: 0.5                    # example

   The first file ``score.yml`` defines the properties of the custom node. |br|
   Lines 2-3 show the mandatory configs ``input`` and ``output``.

   ``input`` defines the data the node would consume, to be read from the pipeline. |br|
   ``output`` defines the data the node would produce, to be put into the pipeline.

   To display the bounding box confidence score, our node requires three pieces
   of input data: the bounding box, the score to display, and the image to draw on.
   These are defined as ``img``, ``bboxes``, ``bbox_scores`` respectively in the 
   :ref:`API docs <api_doc>`.

   Our custom node only displays the score on screen and does not produce any
   outputs for the pipeline, so the output is ``none``.

   There are also no optional configs, so lines 5-6 can be removed.
   The updated ``score.yml`` is:

   .. code-block:: yaml
      :linenos:

      # Mandatory configs
      input: ["img", "bboxes", "bbox_scores"]
      output: ["none"]

      # No optional configs

   .. note::
      Comments in yaml files start with ``#`` |br|
      It is possible for a node to have input: [ \``none`` ]



2. **src/custom_nodes/draw/score.py** (default content):

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

   The second file ``score.py`` contains the boilerplate code for creating a
   custom node. Update the code to implement the desired behavior for the node.

   We will show the modified ``score.py`` below and explain what has been done:

   .. code-block:: python
      :linenos:

      """
      Custom node to show object detection scores
      """

      from typing import Any, Dict, List, Tuple
      import cv2
      from peekingduck.pipeline.nodes.node import AbstractNode

      YELLOW = (0, 255, 255)  # opencv loads file in BGR format


      def map_bbox_to_image_coords(
         bbox: List[float], image_size: Tuple[int, int]
      ) -> List[int]:
         """Convert relative bounding box coords to absolute image coords.
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
         """This is a template class of how to write a node for PeekingDuck.

         Args:
            config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
         """

         def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
            super().__init__(config, node_path=__name__, **kwargs)

         def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
            """This node draws scores on objects detected

            Args:
                  inputs (dict): Dictionary with keys "img", "bboxes", "bbox_scores"

            Returns:
                  outputs (dict): Empty dictionary
            """
            img = inputs["img"]
            bboxes = inputs["bboxes"]
            scores = inputs["bbox_scores"]
            img_size = (img.shape[1], img.shape[0])  # width, height

            for i, bbox in enumerate(bboxes):
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

            return {}

   Line 6 imports the `opencv <https://opencv.org>`_ library which we will use
   to display the score. ``opencv`` would have been installed alongside
   PeekingDuck as it is a dependency.

   Line 7 imports the ``AbstractNode`` class from PeekingDuck which will serve 
   as the parent class for our custom node.

   Line 9 defines the ``YELLOW`` color code for the score. Note that ``opencv`` 
   uses the BGR-format instead of the common RGB-format.

   Lines 12-32 implement a helper function ``map_bbox_to_image_coords`` to map
   the bounding box coordinates to the image coordinates, as explained
   :ref:`above <coordinate_systems>`.

   Line 42 is the node object initializer. We do not require any special setup,
   so it simply calls the ``__init__`` method of its parent class.

   Lines 45-71 implement the display score function in the node's ``run``
   method, which is called by PeekingDuck as it iterates through the pipeline.

   Lines 54-57 extract the inputs from the pipeline and computes the image size
   in ( width, height ).

   Line 59 onwards iterates through all the bounding boxes, whereby it computes
   the (x1, y1) left-top and (x2, y2) right-bottom bounding box coordinates. 
   It also converts the score into a numeric string with two decimal places.

   Line 63 uses the ``opencv`` ``putText`` function to draw the score string
   onto the image at the left-bottom ``org=(x1, y2)`` of the bounding box.
   For more info on the various parameters, please refer to ``opencv``'s API
   documentation.

   Line 73 returns an empty dictionary ``{}`` to tell PeekingDuck that the node
   has no outputs.


3. **run_config.yml** (default content):

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.live
      - model.yolo
      - draw.bbox
      - output.screen

   Finally, the ``run_config.yml`` file implements the pipeline. 
   Modify the default pipeline to the one shown below:

   .. code-block:: yaml
      :linenos:

      nodes:
      - input.recorded:
          input_dir: /folder/containing/demo_video.mp4
      - model.yolo:
         detect_ids: ["cup", "cat", "laptop", "keyboard", "mouse"]
      - draw.bbox:
         show_labels: True
      - custom_nodes.draw.score
      - output.screen

   Line 8 adds our custom node into the pipeline where it will be ``run`` by 
   PeekingDuck during each pipeline iteration.

Execute ``peekingduck run`` to see your custom node in action.

   .. figure:: /assets/tutorials/ss_custom_nodes_1.png
      :alt: Custom node screenshot - show object detection scores

      Custom Node Showing Object Detection Scores

   .. note::

      Royalty free video of computer hardware from:
      https://www.youtube.com/watch?v=-C1TEGZavko



.. _count_hand_wave:

Custom Node 2: Show Keypoints and Count Hand Waves
--------------------------------------------------

This tutorial will create a custom node to analyze the skeletal keypoints of the
person from the ``wave.mp4`` video in the :ref:`pose estimation tutorial
<tutorial_pose_estimation>` and to count the number of times he waves.

The PoseNet pose estimation model outputs seventeen keypoints for the person 
corresponding to the different body parts as documented :ref:`here
<whole-body-keypoint-ids>`.
Each keypoint is a pair of ``(x, y)`` coordinates, where ``x`` and ``y`` are
real numbers ranging from 0.0 to 1.0 (using the relative coordinate system).

Starting with a newly initialised PeekingDuck folder, call ``peekingduck
create-node`` to create a new ``dabble`` custom node ``wave`` as shown below:


.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir custom_project` \
   | \ :blue:`[~user]` \ > \ :green:`cd custom_project` \
   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck init` \
   | Welcome to PeekingDuck! 
   | 2022-02-11 18:17:31 peekingduck.cli  INFO:  Creating custom nodes folder in ~user/custom_project/src/custom_nodes 
   | \ :blue:`[~user/custom_project]` \ > \ :green:`peekingduck create-node` \ 
   | Creating new custom node...
   | Enter node directory relative to ~user/custom_project [src/custom_nodes]: \ :green:`⏎` \
   | Select node type (input, model, draw, dabble, output): \ :green:`dabble` \
   | Enter node name [my_custom_node]: \ :green:`wave` \
   | 
   | Node directory:	~user/custom_project/src/custom_nodes
   | Node type:	dabble
   | Node name:	wave
   | 
   | Creating the following files:
   |    Config file: ~user/custom_project/src/custom_nodes/configs/dabble/wave.yml
   |    Script file: ~user/custom_project/src/custom_nodes/dabble/wave.py
   | Proceed? [Y/n]: \ :green:`⏎` \
   | Created node!


Also, copy ``wave.mp4`` into the above folder.  You should end up with the
following folder structure:


.. parsed-literal::

   \ :blue:`custom_project/` \ |Blank|
   ├── run_config.yml
   ├── \ :blue:`src/` \ |Blank|
   │   └── \ :blue:`custom_nodes/` \ |Blank|
   │       ├── \ :blue:`configs/` \ |Blank|
   │       │   └── \ :blue:`dabble/` \ |Blank|
   │       │       └── wave.yml
   │       └── \ :blue:`dabble/` \ |Blank|
   │           └── wave.py
   └── wave.mp4

To implement this tutorial, the **three files** ``wave.yml``, ``wave.py`` and
``run_config.yml`` are to be edited as follows:

1. **src/custom_nodes/configs/dabble/wave.yml**:

   .. code-block:: yaml
      :linenos:

      # Dabble node has both input and output
      input: ["img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores"]
      output: ["none"]

      # No optional configs

We will implement this tutorial using a ``dabble`` node, which will take the 
inputs ``img``, ``bboxes``, ``bbox_scores``, ``keypoints``, ``keypoint_scores`` 
from the pipeline. The node has no output.

2. **src/custom_nodes/dabble/wave.py**:

   .. code-block:: python
      :linenos:

      """
      Custom node to show keypoints and count the number of times the person's hand is waved
      """

      from typing import Any, Dict, List, Tuple
      import cv2
      from peekingduck.pipeline.nodes.node import AbstractNode

      FONT = cv2.FONT_HERSHEY_SIMPLEX
      WHITE = (255, 255, 255)  # opencv loads file in BGR format
      YELLOW = (0, 255, 255)
      THRESHOLD = 0.6  # ignore keypoints below this threshold


      def map_bbox_to_image_coords(
         bbox: List[float], image_size: Tuple[int, int]
      ) -> List[int]:
         """Convert relative bounding box coords to absolute image coords.
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
         """Convert relative keypoint coords to absolute image coords.
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

            img = inputs["img"]
            bboxes = inputs["bboxes"]
            bbox_scores = inputs["bbox_scores"]
            keypoints = inputs["keypoints"]
            keypoint_scores = inputs["keypoint_scores"]

            img_size = (img.shape[1], img.shape[0])  # image width, height

            # bounding box confidence score
            the_bbox = bboxes[0]  # image only has one person
            the_bbox_score = bbox_scores[0]  # only one set of scores

            x1, y1, x2, y2 = map_bbox_to_image_coords(the_bbox, img_size)
            score_str = f"BBox {the_bbox_score:0.2f}"
            cv2.putText(
                  img=img,
                  text=score_str,
                  org=(x1, y2 - 30),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=1.0,
                  color=WHITE,
                  thickness=3,
            )

            # hand wave detection
            the_keypoints = keypoints[0]  # image only has one person
            the_keypoint_scores = keypoint_scores[0]  # only one set of scores
            right_wrist = None
            right_shoulder = None

            for i, keypoints in enumerate(the_keypoints):
                  keypoint_score = the_keypoint_scores[i]

                  if keypoint_score >= THRESHOLD:
                     x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
                     x_y_str = f"({x}, {y})"

                     if 6 == i:     # right shoulder
                        right_shoulder = keypoints
                        the_color = YELLOW
                     elif i == 10:  # right wrist
                        right_wrist = keypoints
                        the_color = YELLOW
                     else:          # generic keypoint
                        the_color = WHITE

                     draw_text(img, x, y, x_y_str, the_color)

            if right_wrist is not None and right_shoulder is not None:
                  if self.right_wrist is None:
                     self.right_wrist = right_wrist  # first wrist data point
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
                              self.direction = direction  # first direction data point
                        else:
                              # check if hand changes direction
                              if direction != self.direction:
                                 self.num_direction_changes += 1
                              # every three hand direction changes == one wave
                              if self.num_direction_changes >= 2:
                                 self.num_waves += 1
                                 self.num_direction_changes = 0  # reset direction count

                        self.right_wrist = right_wrist  # save last position
                        self.direction = direction

                  wave_str = f"#waves = {self.num_waves}"
                  draw_text(img, 20, 30, wave_str, YELLOW)

            return {}

This long piece of code implements our custom ``dabble`` node. As can be seen, 
this ``dabble.wave`` code structure is very similar to the other custom 
node tutorial ``draw.score`` code structure.

Line 6 imports the ``opencv`` library which we will use for drawing onto the 
image.

Line 7 imports the PeekingDuck's ``AbstractNode`` class which is required for 
all custom node implementation.

Lines 9-12 set up some working global constants.

Lines 15-35 define a helper function ``map_bbox_to_image_coords`` to convert 
relative bounding box coordinates to absolute image coordinates.

Lines 38-56 define a second helper function ``map_keypoint_to_image_coords`` to 
convert relative keypoint coordinates to absolute image coordinates.

Lines 59-68 define another helper function ``draw_text`` to call the ``opencv`` 
drawing function to improve code readability.

Line 71 onwards implements the custom ``dabble`` node logic.

Lines 96-100 get the required inputs from the pipeline.

Lines 105-118 get the bounding box confidence score and draw it at the
left-bottom (x1, y2) corner of the bounding box.

Lines 121-172 implement a simple heuristic to count the number of times the 
person waves his hand. It tracks the direction the right wrist is moving in and 
notes when the wrist changes direction. Upon encountering two direction changes, 
e.g. left -> right -> left, one wave is counted.
The heuristic also waits until the right wrist has been lifted above the right 
should before it starts tracking hand direction and counting waves.
The number of waves is displayed at the left-top corner of the screen.



3. **run_config.yml**:

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
      - draw.legend
      - output.screen

We modify the pipeline file ``run_config.yml`` to run both the object detection 
and pose estimation models to obtain the required inputs for our custom
``dabble`` node.

Execute ``peekingduck run`` to see your custom node in action.

   .. figure:: /assets/tutorials/ss_custom_nodes_2.png
      :alt: Custom node screenshot - count hand waves

      Custom Node Counting Hand Waves



