*****************************
Calling PeekingDuck in Python
*****************************

.. include:: /include/substitution.rst


.. _tutorial_calling_peekingduck_in_python:

Using PeekingDuck's Pipeline
============================

As an alternative to running PeekingDuck using the command-line interface (CLI), users
can also import PeekingDuck as a Python module and run it in a Python script. This demo
corresponds to the :ref:`Record Video File with FPS <tutorial_media_writer>` Section of
the :doc:`Duck Confit </tutorials/02_duck_confit>` tutorial.

In addition, we will demonstrate basic debugging techniques which users can employ when
troubleshooting PeekingDuck projects.


Setting Up
----------

Create a PeekingDuck project using:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir pkd_project` \
   | \ :blue:`[~user]` \ > \ :green:`cd pkd_project` \
   | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck init` \

Then, download the `cat and computer video <https://storage.googleapis.com/peekingduck/videos/cat_and_computer.mp4>`_
to the ``pkd_project`` folder and create a Python script ``demo_debug.py`` in the same folder.

You should have the following directory structure at this point:

.. parsed-literal::

   \ :blue:`pkd_project/` \ |Blank|
   ├── cat_and_computer.mp4
   ├── demo_debug.py
   ├── pipeline_config.yml
   └── \ :blue:`src/` \ |Blank|


Creating a Custom Node for Debugging
------------------------------------

Run the following to create a :mod:`dabble` node for debugging:

.. admonition:: Terminal Session

   | \ :blue:`[~user/pkd_project]` \ > \ :green:`peekingduck create-node -\-node_subdir src/custom_nodes -\-node_type dabble -\-node_name debug` \

The command will create the ``debug.py`` and ``debug.yml`` files in your project directory as
shown:

.. parsed-literal::

   \ :blue:`pkd_project/` \ |Blank|
   ├── cat_and_computer.mp4
   ├── demo_debug.py
   ├── pipeline_config.yml
   └── \ :blue:`src/` \ |Blank|
       └── \ :blue:`custom_nodes/` \ |Blank|
           ├── \ :blue:`configs/` \ |Blank|
           │   └── \ :blue:`dabble/` \ |Blank|
           │       └── \ debug.yml
           └── \ :blue:`dabble/` \ |Blank|
               └── debug.py

Change the content of ``debug.yml`` to:

.. code-block:: yaml
   :linenos:

   input: ["all"]
   output: ["none"]

Line 1: The data type ``all`` allows the node to receive all outputs from the previous
nodes as its input. Please see the :doc:`Glossary </glossary>` for a list of available
data types.

Change the content of ``debug.py`` to:

.. container:: toggle

   .. container:: header

      **Show/Hide Code**
    
   .. code-block:: python
      :linenos:
  
      from typing import Any, Dict
  
      import numpy as np
  
      from peekingduck.pipeline.nodes.abstract_node import AbstractNode
  
  
      class Node(AbstractNode):
          def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
              super().__init__(config, node_path=__name__, **kwargs)
              self.frame = 0
  
          def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
              if "cat" in inputs["bbox_labels"]:
                  print(
                      f"{self.frame} {inputs['bbox_scores'][np.where(inputs['bbox_labels'] == 'cat')]}"
                  )
              self.frame += 1
              return {}

Lines 14 - 17: Print out the frame number and the confidence scores of bounding boxes which are
detected as "cat".

Line 18: Increment the frame number each time ``run()`` is called.


Creating the Python Script
--------------------------

Copy over the following code to ``demo_debug.py``:

.. container:: toggle

   .. container:: header

      **Show/Hide Code**

   .. code-block:: python
      :linenos:
  
      from pathlib import Path
  
      from peekingduck.pipeline.nodes.dabble import fps
      from peekingduck.pipeline.nodes.draw import bbox, legend
      from peekingduck.pipeline.nodes.input import visual
      from peekingduck.pipeline.nodes.model import yolo
      from peekingduck.pipeline.nodes.output import media_writer, screen
      from peekingduck.runner import Runner
      from src.custom_nodes.dabble import debug
  
  
      def main():
          debug_node = debug.Node(pkd_base_dir=Path.cwd() / "src" / "custom_nodes")
      
          visual_node = visual.Node(source=str(Path.cwd() / "cat_and_computer.mp4"))
          yolo_node = yolo.Node(detect=["cup", "cat", "laptop", "keyboard", "mouse"])
          bbox_node = bbox.Node(show_labels=True)
      
          fps_node = fps.Node()
          legend_node = legend.Node(show=["fps"])
          screen_node = screen.Node()
      
          media_writer_node = media_writer.Node(output_dir=str(Path.cwd() / "results"))
      
          runner = Runner(
              nodes=[
                  visual_node,
                  yolo_node,
                  debug_node,
                  bbox_node,
                  fps_node,
                  legend_node,
                  screen_node,
                  media_writer_node,
              ]
          )
          runner.run()
  
  
      if __name__ == "__main__":
          main()

Lines 9, 13: Import and initialize the ``debug`` custom node. Pass in the 
``path/to/project_dir/src/custom_nodes`` via ``pkd_base_dir`` for the configuration YAML file of
the custom node to be loaded properly.

Lines 15 - 23: Create the PeekingDuck nodes necessary to replicate the demo shown in the
:ref:`Record Video File with FPS <tutorial_media_writer>` tutorial. To change the node
configuration, you can pass the new values to the ``Node()`` constructor as keyword arguments.


Lines 25 - 37: Initialize the PeekingDuck ``Runner`` from
`runner.py <https://github.com/aimakerspace/PeekingDuck/blob/main/peekingduck/runner.py>`_ with the
list of nodes passed in via the ``nodes`` argument.

.. note::

   A PeekingDuck node can be created in Python code by passing a dictionary of config keyword -
   config value pairs to the ``Node()`` constructor.


Running the Python Script
-------------------------

Run the ``demo_debug.py`` script using:

.. admonition:: Terminal Session

   | \ :blue:`[~user/pkd_project]` \ > \ :green:`python demo_debug.py` \

You should see the following output in your terminal:

.. code-block:: text
   :linenos:

   2022-02-24 16:33:06 peekingduck.pipeline.nodes.input.visual  INFO:  Config for node input.visual is updated to: 'source': ~user/pkd_project/cat_and_computer.mp4 
   2022-02-24 16:33:06 peekingduck.pipeline.nodes.input.visual  INFO:  Video/Image size: 720 by 480 
   2022-02-24 16:33:06 peekingduck.pipeline.nodes.input.visual  INFO:  Filepath used: ~user/pkd_project/cat_and_computer.mp4 
   2022-02-24 16:33:06 peekingduck.pipeline.nodes.model.yolo  INFO:  Config for node model.yolo is updated to: 'detect': [41, 15, 63, 66, 64] 
   2022-02-24 16:33:06 peekingduck.pipeline.nodes.model.yolov4.yolo_files.detector  INFO:  Yolo model loaded with following configs: 
       Model type: v4tiny, 
       Input resolution: 416, 
       IDs being detected: [41, 15, 63, 66, 64] 
       Max Detections per class: 50, 
       Max Total Detections: 50, 
       IOU threshold: 0.5, 
       Score threshold: 0.2 
   2022-02-24 16:33:07 peekingduck.pipeline.nodes.draw.bbox  INFO:  Config for node draw.bbox is updated to: 'show_labels': True 
   2022-02-24 16:33:07 peekingduck.pipeline.nodes.dabble.fps  INFO:  Moving average of FPS will be logged every: 100 frames 
   2022-02-24 16:33:07 peekingduck.pipeline.nodes.output.media_writer  INFO:  Config for node output.media_writer is updated to: 'output_dir': ~user/pkd_project/results 
   2022-02-24 16:33:07 peekingduck.pipeline.nodes.output.media_writer  INFO:  Output directory used is: ~user/pkd_project/results 
   0 [0.90861976]
   1 [0.9082737]
   2 [0.90818006]
   3 [0.8888804]
   4 [0.8877487]
   5 [0.9071386]
   6 [0.870267]

   [Truncated]

Lines 17 - 23: The debugging output showing the frame number and the confidence score of
bounding boxes predicted as "cat".


Integrating with Your Workflow
==============================

The modular design of PeekingDuck allows users to pick and choose the nodes they want to use. Users
are also able to use PeekingDuck nodes with external packages when designing their pipeline.

In this demo, we will show how users can construct a custom PeekingDuck pipeline using:

    * Data loaders such as `tf.keras.preprocessing.image_dataset_from_directory
      <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory>`_
      (available in ``tensorflow>=2.3.0``),
    * External packages (not implemented as PeekingDuck nodes) such as `easyocr
      <https://pypi.org/project/easyocr/>`_, and
    * Visualization packages such as `matplotlib <https://pypi.org/project/matplotlib/>`_.

The notebook corresponding to this tutorial, ``calling_peekingduck_in_python.ipynb``, can be found in the
`notebooks <https://github.com/aimakerspace/PeekingDuck/tree/main/notebooks>`_ folder of the
PeekingDuck repository and is also available as a
`Colab notebook <https://colab.research.google.com/drive/1NwQKrnY_3ia2mBEaUinkvUqbrjjT3ssq#scrollTo=l2MCyh5Hgp5O>`_.


.. raw:: html

    <h3>Running Locally</h3>


.. raw:: html

    <h4>Install Prerequisites</h4>

.. container:: toggle

   .. container:: header

      **Show/Hide Instructions for Linux/Mac (Intel)/Windows**

   .. raw:: html

      <p>
        <div class="highlight"><pre
          ><span class="pkd-raw-terminal">pip install easyocr</span
          ><br /><span class="pkd-raw-terminal">pip uninstall -y opencv-python-headless opencv-contrib-python</span
          ><br /><span class="pkd-raw-terminal">pip install "tensorflow&lt;2.8.0,&gt;=2.3.0" opencv-contrib-python==4.5.4.60 matplotlib oidv6 lap==0.4.0</span
          ><br /><span class="pkd-raw-terminal">pip install colorama==0.4.4</span
        ></pre></div>

   .. note::
       
      The uninstallation step is necessary to ensure that the proper version of OpenCV is installed.
  
      You may receive an error message about the incompatibility between ``awscli`` and
      ``colorama==0.4.4``. ``awscli`` is conservative about pinning versions to maintain backward
      compatibility. The code presented in this tutorial has been tested to work and we have chosen
      to prioritize PeekingDuck's dependency requirements.

.. container:: toggle

   .. container:: header

      **Show/Hide Instructions for Mac (Apple Silicon)**

   .. raw:: html

        <div class="highlight"><pre
          ><span class="pkd-raw-terminal">conda create -n pkd_notebook python=3.8</span
          ><br /><span class="pkd-raw-terminal">conda activate pkd_notebook</span
          ><br /><span></span
          ><br /><span class="pkd-raw-terminal">conda install jupyterlab matplotlib click colorama opencv openblas pyyaml \</span
          ><br /><span>    requests scipy shapely tqdm pillow scikit-image python-bidi pandas awscli progressbar2</span
          ><br /><span class="pkd-raw-terminal">pip install easyocr oidv6 lap</span
          ><br /><span class="pkd-raw-terminal">pip uninstall opencv-contrib-python opencv-python-headless</span
          ><br /><span></span
          ><br /><span class="pkd-raw-terminal"># Pick one:</span
          ><br /><span class="pkd-raw-terminal"># for macOS Big Sur</span
          ><br /><span class="pkd-raw-terminal">conda install -c apple tensorflow-deps=2.6.0</span
          ><br /><span class="pkd-raw-terminal">pip install tensorflow-estimator==2.6.0 tensorflow-macos==2.6.0</span
          ><br /><span class="pkd-raw-terminal">pip install tensorflow-metal==0.2.0</span
          ><br /><span class="pkd-raw-terminal"># for macOS Monterey</span
          ><br /><span class="pkd-raw-terminal">conda install -c apple tensorflow-deps</span
          ><br /><span class="pkd-raw-terminal">pip install tensorflow-macos tensorflow-metal</span
          ><br /><span></span
          ><br /><span class="pkd-raw-terminal">pip install torch torchvision</span
          ><br /><span class="pkd-raw-terminal">pip install 'peekingduck==1.2.0' --no-dependencies</span
        ></pre></div>
      </p>

   .. note::

      We install the problematic packages ``easyocr`` and ``oidv6`` first and then uninstall the
      ``pip``-related OpenCV packages which were installed as dependencies. Mac (Apple silicon)
      needs ``conda``'s OpenCV.

      There will be a warning that ``easyocr`` needs some version of Pillow which can be ignored.


.. raw:: html

    <br />
    <h3>Download Demo Data</h3>

We are using `Open Images Dataset V6 <https://storage.googleapis.com/openimages/web/index.html>`_
as the dataset for this demo. We recommend using the third-party
`oidv6 PyPI package <https://pypi.org/project/oidv6/>`_ to download the images necessary for this
demo.

Run the following command after installing the prerequisites:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir pkd_project` \
   | \ :blue:`[~user]` \ > \ :green:`cd pkd_project` \
   | \ :blue:`[~user/pkd_project]` \ > \ :green:`oidv6 downloader en -\-dataset data/oidv6 -\-type_data train -\-classes car -\-limit 10 -\-yes` \

Copy ``calling_peekingduck_in_python.ipynb`` to the ``pkd_project`` folder and you should have the
following directory structure at this point:

.. parsed-literal::

   \ :blue:`pkd_project/` \ |Blank|
   ├── calling_peekingduck_in_python.ipynb
   └── \ :blue:`data/` \ |Blank|
       └── \ :blue:`oidv6/` \ |Blank|
           ├── \ :blue:`boxes/` \ |Blank|
           ├── \ :blue:`metadata/` \ |Blank|
           └── \ :blue:`train/` \ |Blank|
               └── \ :blue:`car/` \ |Blank|


Import the Modules
------------------

.. container:: toggle

   .. container:: header

      **Show/Hide Code** 
      
   .. code-block:: python
      :linenos:
  
      import os
      from pathlib import Path
  
      import cv2
      import easyocr
      import matplotlib.pyplot as plt
      import numpy as np
      import tensorflow as tf
      from peekingduck.pipeline.nodes.draw import bbox
      from peekingduck.pipeline.nodes.model import yolo_license_plate
  
      %matplotlib inline

Lines 9 - 10: You can also do::

   from peekingduck.pipeline.nodes.draw import bbox as pkd_bbox
   from peekingduck.pipeline.nodes.model import yolo_license_plate as pkd_yolo_license_plate
    
   bbox_node = pkd_bbox.Node()
   yolo_license_plate_node = pkd_yolo_license_plate.Node()

to avoid potential name conflicts.


Initialize PeekingDuck Nodes
----------------------------

.. container:: toggle

   .. container:: header 

      **Show/Hide Code**

   .. code-block:: python
      :linenos:
  
      yolo_lp_node = yolo_license_plate.Node()
  
      bbox_node = bbox.Node(show_labels=True)

Lines 3: To change the node configuration, you can pass the new values to the ``Node()``
constructor as keyword arguments.

Refer to the :ref:`API Documentation <api_doc>` for the configurable settings for each node.


Create a Dataset Loader
-----------------------

.. container:: toggle

   .. container:: header

      **Show/Hide Code**

   .. code-block:: python
      :linenos:
  
      data_dir = Path.cwd().resolve() / "data" / "oidv6" / "train"
      dataset = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir, batch_size=1, shuffle=False
      )

Lines 2 - 4: We create the data loader using ``tf.keras.preprocessing.image_dataset_from_directory()``;
you can also create your own data loader class.


Create a License Plate Parser Class
-----------------------------------

.. container:: toggle

   .. container:: header

      **Show/Hide Code** 

   .. code-block:: python
      :linenos:
  
      class LPReader:
          def __init__(self, use_gpu):
              self.reader = easyocr.Reader(["en"], gpu=use_gpu)
  
          def read(self, image):
              """Reads text from the image and joins multiple strings to a
              single string.
              """
              return " ".join(self.reader.readtext(image, detail=0))
      
      reader = LPReader(False)

We create the license plate parser class in a Python class using ``easyocr`` to demonstrate how
users can integrate the PeekingDuck pipeline with external packages.

Alternatively, users can create a custom node for parsing license plates and run the pipeline
through the CLI instead. Refer to the :ref:`custom nodes <tutorial_custom_nodes>`
tutorial for more information.


The Inference Loop
------------------

.. container:: toggle

   .. container:: header

      **Show/Hide Code** 

   .. code-block:: python
      :linenos:
  
      def get_best_license_plate(frame, bboxes, bbox_scores, width, height):
          """Returns the image region enclosed by the bounding box with the highest
          confidence score.
          """
          best_idx = np.argmax(bbox_scores)
          best_bbox = bboxes[best_idx].astype(np.float32).reshape((-1, 2))
          best_bbox[:, 0] *= width
          best_bbox[:, 1] *= height
          best_bbox = np.round(best_bbox).astype(int)
  
          return frame[slice(*best_bbox[:, 1]), slice(*best_bbox[:, 0])]
      
      num_col = 3
      # For visualization, we plot 3 columns, 1) the original image, 2) image with
      # bounding box, and 3) the detected license plate region with license plate
      # number prediction shown as the plot title 
      fig, ax = plt.subplots(
          len(dataset), num_col, figsize=(num_col * 3, len(dataset) * 3)
      )
      for i, (element, path) in enumerate(zip(dataset, dataset.file_paths)):
          image_orig = cv2.imread(path)
          image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
          height, width = image_orig.shape[:2]
  
          image = element[0].numpy().astype("uint8")[0].copy()
  
          yolo_lp_input = {"img": image}
          yolo_lp_output = yolo_lp_node.run(yolo_lp_input)
  
          bbox_input = {
              "img": image,
              "bboxes": yolo_lp_output["bboxes"],
              "bbox_labels": yolo_lp_output["bbox_labels"],
          }
          _ = bbox_node.run(bbox_input)
  
          ax[i][0].imshow(image_orig)
          ax[i][1].imshow(image)
          # If there are any license plates detected, try to predict the license
          # plate number
          if len(yolo_lp_output["bboxes"]) > 0:
              lp_image = get_best_license_plate(
                  image_orig, yolo_lp_output["bboxes"],
                  yolo_lp_output["bbox_scores"],
                  width,
                  height,
              )
              lp_pred = reader.read(lp_image)
              ax[i][2].imshow(lp_image)
              ax[i][2].title.set_text(f"Pred: {lp_pred}")

Lines 1 - 11: We define a utility function for retrieving the image region of the license
plate with the highest confidence score to improve code clarity. For more information on
how to convert between bounding box and image coordinates, please refer to the
:ref:`Bounding Box vs Image Coordinates <tutorial_coordinate_systems>` tutorial.

Lines 27 - 35: By carefully constructing the input for each of the nodes, we can perform the
inference loop within a custom workflow.

Lines 37 - 38: We plot the data using ``matplotlib`` for debugging and visualization purposes.

Lines 41 - 48: We integrate the inference loop with external packages such as the license plate
parser we have created earlier using ``easyocr``.
