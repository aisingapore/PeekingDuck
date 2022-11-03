.. include:: /include/substitution.rst

*********************
Using Model Hub Nodes
*********************

PeekingDuck has support for models from external model hubs. You can leverage the model
hub nodes to use these external models with the PeekingDuck pipeline. The model
hub nodes differ in design from traditional model nodes as the external models
perform a variety of computer vision tasks. This tutorial demonstrates how
to use these nodes through some sample pipelines.


List of Model Hub Nodes
=======================

The table below shows the model hub nodes available.

+---------------------------------+-----------------------------------------------------------------+
| Documentation                   | Model Hubs                                                      |
+=================================+=================================================================+
| :mod:`model.huggingface_hub`    | `Hugging Face Hub <https://huggingface.co/docs/hub/index>`_     |
+---------------------------------+-----------------------------------------------------------------+
| :mod:`model.mediapipe_hub`      | `MediaPipe Solutions <https://google.github.io/mediapipe/>`_    |
+---------------------------------+-----------------------------------------------------------------+


Hugging Face Hub
================

The :mod:`model.huggingface_hub` node supports transformer models which perform
the following computer vision tasks:

   #. Object detection
   #. Instance segmentation

You can use the following command to get a list of supported computer vision tasks:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface tasks` \
      | Supported computer vision tasks: ['instance segmentation', 'object detection']


Object Detection
----------------

This example shows how Hugging Face Hub's object detection models can be used
to blur computer related objects from `cat_and_computer.mp4 
<https://storage.googleapis.com/peekingduck/videos/cat_and_computer.mp4>`_.

Supported Models
^^^^^^^^^^^^^^^^

Use the following command to get a list of supported object detection models:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface models -\-task \'object detection\'` \
      | Supported Hugging Face \`object_detection\` models:
      | facebook/detr-resnet-50
      | facebook/detr-resnet-50-dc5
      | hustvl/yolos-base
      | hustvl/yolos-small
      |
      | <long list truncated>

Class Labels
^^^^^^^^^^^^

The models are trained on a variety of datasets which may not share the same set
of class labels, e.g., the `PubTables-1M <https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3>`_
dataset contains the label "table column" which is not found in the `COCO <https://cocodataset.org/#home>`_
dataset. As such, it may be necessary to look through the class labels supported
by the model. You can do so with the following command:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface detect-ids  -\-model_type \'hustvl/yolos-small\'` \
      | The detect ID-to-label mapping for \`facebook/hustvl/yolos-small\` can be found at https://huggingface.co/hustvl/yolos-small/blob/main/config.json under the \`id2label\` key.

Pipeline
^^^^^^^^

The following ``pipeline_config.yml`` can be used to blur computer related objects:

   .. code-block:: yaml
      :linenos:
   
      nodes:
      - input.visual:
          source: /path/to/cat_and_computer.mp4
      - model.huggingface_hub:
          task: object_detection 
          model_type: hustvl/yolos-small
          detect: ["laptop", "keyboard", "mouse"]
      - draw.blur_bbox
      - output.screen

Here is a step-by-step explanation of what has been done:

   | Line 2 :mod:`input.visual` tells PeekingDuck to load ``cat_and_computer.mp4``.
   | Line 4 :mod:`model.huggingface_hub` is set to perform the ``object_detection``
   |        task and load the ``hustvl/yolos-small`` model. We also
   |        set the model to detect only laptops, keyboards, and mice.
   | Line 8 :mod:`draw.blur_bbox` is used to apply a blur effect on the bounding
   |        boxes detected by the model.

Run the above with the command :greenbox:`peekingduck run`. 

   .. figure:: /assets/tutorials/ss_huggingface_object_detection.png
      :width: 416
      :alt: Cat and Computer Screenshot with Computer Related Objects Blurred
   
      Cat and Computer Screenshot with Computer Related Objects Blurred


You should see a display of the ``cat_and_computer.mp4`` with the computer related
objects blurred. The 30-second video will auto-close at the end, or you can press
:greenbox:`q` to end early.


Instance Segmentation
---------------------

This example also demonstrates how to blur computer related objects but with Hugging
Face Hub's instance segmentation models instead.

Supported Models
^^^^^^^^^^^^^^^^

To get a list of supported instance segmentation models, use the following CLI command:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface models -\-task \'instance segmentation\'` \
      | Supported hugging face \`instance_segmentation\` models:
      | facebook/detr-resnet-101-panoptic
      | facebook/detr-resnet-50-dc5-panoptic
      | facebook/detr-resnet-50-panoptic
      | facebook/maskformer-swin-tiny-coco
      |
      | <long list truncated>

Class Labels
^^^^^^^^^^^^

Similar to the object detection models, the instance segmentation models are also
trained on a variety of datasets, the same ``detect-ids`` command can be used to
retrieve the file containing the model's class labels:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface detect-ids  -\-model_type \'facebook/maskformer-swin-tiny-coco\'` \
      | The detect ID-to-label mapping for \`facebook/maskformer-swin-tiny-coco\` can be found at https://huggingface.co/facebook/maskformer-swin-tiny-coco/blob/main/config.json under the \`id2label\` key.

Pipeline
^^^^^^^^

The task of blurring computer related objects can be achieved with the following
``pipeline_config.yml``:

   .. code-block:: yaml
      :linenos:
   
      nodes:
      - input.visual:
          source: /path/to/cat_and_computer.mp4
      - model.huggingface_hub:
          task: instance_segmentation
          model_type: facebook/maskformer-swin-tiny-coco
          detect: ["laptop", "keyboard", "mouse"]
      - draw.instance_mask:
          effect:
            blur: 50
      - output.screen

Here is a step-by-step explanation of what has been done:

   | Line 2 :mod:`input.visual` tells PeekingDuck to load ``cat_and_computer.mp4``.
   | Line 4 :mod:`model.huggingface_hub` is set to perform the ``instance_segmentation``
   |        task and load the ``facebook/maskformer-swin-tiny-coco`` model. We also
   |        set the model to detect only laptops, keyboards, and mice.
   | Line 8 :mod:`draw.instance_mask` is used to visualize the output of the
   |        Hugging Face Hub model. A blur effect is applied on the instance mask
   |        outputs.

Run the above with the command :greenbox:`peekingduck run`. 

   .. figure:: /assets/tutorials/ss_huggingface_instance_segmentation.png
      :width: 416
      :alt: Cat and Computer Screenshot with Computer Related Objects Blurred
   
      Cat and Computer Screenshot with Computer Related Objects Blurred


You should see a display of the ``cat_and_computer.mp4`` with the computer related
objects blurred. The 30-second video will auto-close at the end, or you can press
:greenbox:`q` to end early.

MediaPipe Solutions
===================

The :mod:`model.mediapipe_hub` node supports MediaPipe solutions which perform the
following computer vision tasks:

   #. Object detection
   #. Pose estimation

*Note: The object detection model only detects faces as the generic object detection
model is not available in Python at the time of writing.*

You can use the following command to get a list of supported computer vision tasks
and their respective subtasks:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub mediapipe tasks` \
      | Supported computer vision tasks and respective subtasks:
      | pose estimation
      |     body
      | object detection
      |     face


Object Detection (Face)
-----------------------

This example demonstrates the usage of MediaPipe object detection (face) solution.

Supported Model Types
^^^^^^^^^^^^^^^^^^^^^

To get a list of supported model types for the pose estimation (body) task, use
the following command:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub mediapipe model-types -\-task \'object detection\' -\-subtask \'face\'` \
      | Supported model types for 'object detection/face'
      | 0: A short-range model that works best for faces within 2 meters from the camera.
      | 1: A full-range model best for faces within 5 meters.


Pipeline
^^^^^^^^

   .. code-block:: yaml
      :linenos:
   
       nodes:
        - input.visual:
            source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
        - model.mediapipe_hub:
            task: object_detection
            subtask: face 
            model_type: 0
        - draw.bbox
        - output.screen

Here is a step-by-step explanation of what has been done:

   | Line 2 :mod:`input.visual` tells PeekingDuck to load ``wave.mp4``.
   | Line 4 :mod:`model.mediapipe_hub` is set to perform the ``object_detection``
   |        task and ``face`` subtask. Model type 0 is selected.
   | Line 8 :mod:`draw.bbox` is used to visualize the output of the
   |        MediaPipe model.

Run the above with the command :greenbox:`peekingduck run`.

   .. figure:: /assets/tutorials/ss_mediapipe_object_detection_face.png
      :width: 416
      :alt: Object Detection Screenshot
   
      Object Detection Screenshot

You should see a display of the ``wave.mp4`` with a bounding box drawn around the
person's face. The video will auto-close at the end, or you can press :greenbox:`q`
to end early.


Pose Estimation (Body)
----------------------

This example demonstrates the usage of MediaPipe pose estimation (body) solution.

Supported Model Types
^^^^^^^^^^^^^^^^^^^^^

To get a list of supported model types for the pose estimation (body) task, use
the following command:

   .. admonition:: Terminal Session
   
      | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub mediapipe model-types -\-task \'pose estimation\' -\-subtask \'body\'` \
      | Supported model types for 'pose estimation/body'
      | 0: BlazePose GHUM Lite, lower inference latency at the cost of landmark accuracy.
      | 1: BlazePose GHUM Full.
      | 2: BlazePose GHUM Heavy, higher landmark accuracy at the cost of inference latency.

Pipeline
^^^^^^^^

   .. code-block:: yaml
      :linenos:
   
       nodes:
        - input.visual:
            source: https://storage.googleapis.com/peekingduck/videos/wave.mp4
        - model.mediapipe_hub:
            task: pose_estimation
            subtask: body
            model_type: 1
        - draw.poses
        - output.screen

Here is a step-by-step explanation of what has been done:

   | Line 2 :mod:`input.visual` tells PeekingDuck to load ``wave.mp4``.
   | Line 4 :mod:`model.mediapipe_hub` is set to perform the ``pose_estimation``
   |        task and ``body`` subtask. Model type 1 is selected.
   | Line 8 :mod:`draw.poses` is used to visualize the output of the
   |        MediaPipe model.

Run the above with the command :greenbox:`peekingduck run`.

   .. figure:: /assets/tutorials/ss_mediapipe_pose_estimation_body.png
      :width: 416
      :alt: Pose Estimation Screenshot
   
      Pose Estimation Screenshot

You should see a display of the ``wave.mp4`` with the skeletal poses drawn on the
person. The video will auto-close at the end, or you can press :greenbox:`q` to
end early.
