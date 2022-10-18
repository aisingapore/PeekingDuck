*********************
Using Model Hub Nodes 
*********************

.. include:: /include/substitution.rst

PeekingDuck has support for external model hub models. You can leverage the model
hub nodes to use these external models with the PeekingDuck pipeline. The model
hub nodes differ in design from traditional model nodes as the model hub models
perform a variety of computer vision tasks. This tutorial will demonstrate how
to use these nodes through some sample pipelines.


Hugging Face Hub
================

The :mod:`model.huggingface_hub` node supports transformer models which perform
the following computer vision tasks:

   #. Instance segmentation
   #. Object detection

You can use the following command to get a list of supported computer vision tasks:

.. admonition:: Terminal Session

   | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface tasks` \
   | Supported computer vision tasks: ['instance segmentation', 'object detection']

Instance segmentation
_____________________

This examples show how Hugging Face Hub's instance segmentation models can be used
to blur computer related objects from `cat_and_computer.mp4 
<https://storage.googleapis.com/peekingduck/videos/cat_and_computer.mp4>`_.

Supported models
****************

To get a list of supported instance segmentation models, use the following CLI command:

.. admonition:: Terminal Session

   | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface models -\-task 'instance segmentation'` \
   | Supported hugging face \`instance_segmentation\` models:
   | facebook/detr-resnet-101-panoptic
   | facebook/detr-resnet-50-dc5-panoptic
   | facebook/detr-resnet-50-panoptic
   | facebook/maskformer-swin-base-ade
   |
   | <omitted>

Class labels
************

The models are trained on a variety of datasets which may not share the same set
of class labels, e.g., the `ADE20K <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_
dataset contains the label "sky" which is not found in the `COCO <https://cocodataset.org/#home>`_
dataset. As such, it may be necessary to look through the class labels supported
by the model. You can do so with the following command:

.. admonition:: Terminal Session

   | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface detect-ids  -\-model_type 'facebook/detr-resnet-50-panoptic'` \
   | The detect ID-to-label mapping for \`facebook/detr-resnet-50-panoptic\` can be found at https://huggingface.co/facebook/detr-resnet-50-panoptic/blob/main/config.json under the \`id2label\` key.


Pipeline
********

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

Object detection
________________


.. admonition:: Terminal Session

   | \ :blue:`[~user/project_dir]` \ > \ :green:`peekingduck model-hub huggingface models -\-task 'object detection'` \
   | Supported Hugging Face \`object_detection\` models:
   | facebook/detr-resnet-50
   | facebook/detr-resnet-50-dc5
   | hustvl/yolos-base
   | hustvl/yolos-small
   |
   | <omitted>

