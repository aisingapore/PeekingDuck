*************************************
Import PeekingDuck as a Python Module
*************************************

.. |br| raw:: html

   <br />

.. |Blank| unicode:: U+2800 .. Invisible character

.. role:: red

.. role:: blue

.. role:: green

**TODO: Update links**

The modular design of PeekingDuck allows users to pick and choose the nodes they want to use. Users
are also able to use PeekingDuck nodes with external libraries when designing their pipeline.

In this tutorial, we demonstrate how to users can construct a custom PeekingDuck pipeline using:

    * Data loaders such as `tf.keras.utils.image_dataset_from_directory
      <https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory>`_
      (available in ``tensorflow>=2.3.0``),
    * External models (not implemented as PeekingDuck nodes) such `easyocr
      <https://pypi.org/project/easyocr/>`_, and
    * Visualization packages such as `matplotlib <https://pypi.org/project/matplotlib/>`_.

The notebook corresponding in this tutorial can be found in the `notebooks <link>`_ folder of the
PeekingDuck repository and is also available at a `Colab notebook <link>`_.

.. raw:: html

    <h2>Running locally</h2>

.. raw:: html

    <h3>Prerequisites</h3>

.. code-block:: text

    > pip install easyocr
    > pip uninstall -y opencv-python-headless opencv-contrib-python
    > pip install "tensorflow<2.7.0,>=2.3.0" opencv-contrib-python==4.5.4.60 matplotlib oidv6 lap==0.4.0

.. note::
    
    The uninstallation step is necessary to ensure that the proper version of OpenCV is installed.

.. raw:: html

    <h3>Download Demo Data</h3>

We are using `Open Images Dataset V6 <https://storage.googleapis.com/openimages/web/index.html>`_
as the dataset for this demo. We recommend using the third party
`oidv6 PyPI package <https://pypi.org/project/oidv6/>`_ to download the images necessary for this
demo.

Run the following command after installing:

.. admonition:: Terminal Session

    | \ :blue:`[~user]` \ > \ :green:`mkdir pkd_project` \
    | \ :blue:`[~user]` \ > \ :green:`cd pkd_project` \
    | \ :blue:`[~user]/pkd_project` \ > \ :green:`oidv6 downloader en -\-dataset data/oidv6 -\-type_data train -\-classes car -\-limit 10 -\-yes` \

You should have the following directory structure at this point:

.. parsed-literal::

   \ :blue:`pkd_project/` \ |Blank|
   ├── demo_import_peekingduck.ipynb
   └── \ :blue:`data/` \ |Blank|
       └── \ :blue:`oidv6/` \ |Blank|
           ├── \ :blue:`boxes/` \ |Blank|
           ├── \ :blue:`metadata/` \ |Blank|
           └── \ :blue:`train/` \ |Blank|
               └── \ :blue:`car/` \ |Blank|

Importing the Modules
=====================

.. code-block:: python
    :linenos:

    import os
    from pathlib import Path

    import cv2
    import easyocr
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from peekingduck.pipeline.nodes import draw, model

    %matplotlib inline

Line 9: We recommend importing PeekingDuck modules using::

    from peekingduck.pipeline.nodes import model

    yolo_node = model.yolo.Node()

as it isolates the namespace to avoid potential conflicts.

Initialize PeekingDuck nodes
============================

.. code-block:: python
    :linenos:

    yolo_lp_node = model.yolo_license_plate.Node()

    bbox_config = {"show_labels": True}
    bbox_node = draw.bbox.Node(**bbox_config)

Line 3 - 4: To change the node configuration, you can pass the new values to the `Node()`
constructor as keyword arguments.

Refer to the :ref:`API Documentation <api_doc>` for the configurable settings for each node.

Create a Dataset Loader
=======================

.. code-block:: python
    :linenos:

    data_dir = Path.cwd().resolve() / "data" / "oidv6" / "train"
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir, batch_size=1, shuffle=False
    )

Line 2: We create the data loader using ``tf.keras.utils.image_dataset_from_directory()``, you can
also create your own data loader class.

Create a License Plate Parser Class
===================================

.. code-block:: python
    :linenos:

    class LPReader:
        def __init__(self, use_gpu):
            self.reader = easyocr.Reader(["en"], gpu=use_gpu)

        def read(self, image):
            """Reads text from the image and joins multiple multiple strings to a
            single string.
            """
            return " ".join(self.reader.readtext(image, detail=0))
    
    reader = LPReader(False)

We chose to create the license plate parser class in a Python class using ``easyocr`` to
demonstrate how users can integrate the PeekingDuck pipeline with external processes. It is also
possible to create a custom node for parsing license plates and run the pipeline through the
command-line interface (CLI) instead. Refer to the `custom nodes <link>`_ tutorial for more
information.

**TODO: update link**

The Inference Loop
==================

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
        # TODO: Ensure model takes in BGR image after it's fixed
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



Line 1 - 11: We define a utility function for retrieving the image region of the license plate with a
highest confidence score to improve code clarity. For more information on how to convert between
bounding box and image coordinates, please refer to the `Bounding Box vs Image Coordinates <link>`_
section in our tutorials.

Line 26 - 34: By carefully constructing the input for each of the nodes, we can perform the
inference loop without having to use PeekingDuck's `Runner <link>`_.

Line 36 - 37: We plot the data for debugging and visualization purposes.

Line 41 - 47: We integrate the inference loop external processes such as the license plate parser
we have created earlier.

**TODO: update link**
