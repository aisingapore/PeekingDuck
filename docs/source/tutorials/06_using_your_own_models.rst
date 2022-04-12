*********************
Using Your Own Models
*********************

.. include:: /include/substitution.rst

PeekingDuck offers pre-trained :mod:`model` nodes that can be used to tackle a wide variety of
problems, but you may need to train your own model on a custom dataset sometimes. This tutorial
will show you how to package your model into a custom :mod:`model` node, and use it with
PeekingDuck. We will be tackling a manufacturing use case here - classifying images of steel
castings into "defective" or "normal" classes.

Casting is a manufacturing process in which a material such as metal in liquid form is poured into
a mold and allowed to solidify. The solidified result is also called a casting. Sometimes,
defective castings are produced, and quality assurance departments are responsible for preventing
defective pieces from being used downstream. As inspections are usually done manually, this adds
a significant amount of time and cost, and thus there is an incentive to automate this process.

The images of castings used in this tutorial are the front faces of steel `pump impellers 
<https://en.wikipedia.org/wiki/Impeller>`_. From the comparison below, it can be seen that the
"defective" casting has a rough, uneven edges while the "normal" casting has smooth edges.

   .. figure:: /assets/tutorials/normal_vs_defective.png
      :width: 416
      :alt: Normal casting compared to defective casting

      Normal Casting Compared to Defective Casting


.. _tutorial_model_training:

Model Training
==============

PeekingDuck is designed for model *inference* rather than model *training*. This optional section
shows how a simple Convolutional Neural Network (CNN) model can be trained separately from the
PeekingDuck framework. If you have already trained your own model, the
:ref:`following section <tutorial_converting_to_custom_model_node>` describes how you can convert
it to a custom model node, and use PeekingDuck for inference.



Setting Up
----------

Create the following project folder:

.. admonition:: Terminal Session

   | \ :blue:`[~user]` \ > \ :green:`mkdir castings_project` \
   | \ :blue:`[~user]` \ > \ :green:`cd castings_project` \

Within the ``castings_project`` folder:

* Download the `castings dataset <https://storage.googleapis.com/peekingduck/data/castings_data.zip>`_
  and unzip the file.
* Create an empty Python script named ``train_classifier.py``.

.. note::

   The castings dataset used in this example is modified from the original dataset from
   `Kaggle <https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product?resource=download&select=casting_data>`_.


You should have the following directory structure at this point:

.. parsed-literal::

   \ :blue:`castings_project/` \ |Blank|
   ├── train_classifier.py
   └── \ :blue:`castings_data/` \ |Blank|
         ├── \ :blue:`train/` \ |Blank|
         ├── \ :blue:`validation/` \ |Blank|
         └── \ :blue:`test/` \ |Blank|

Install the following prerequisite for visualization. ::

> pip install matplotlib

Update Training Script
----------------------

#. **train_classifier.py**:

   Update the empty ``train_classifier.py`` file you have created with the following code:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for train_classifier.py**

      .. code-block:: python
         :linenos:

         """
         Script to train a classification model on images, save the model, and plot the training results

         Adapted from: https://www.tensorflow.org/tutorials/images/classification
         Dataset from: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product?resource=download
         """

         import pathlib
         from typing import List, Tuple

         import matplotlib.pyplot as plt
         import tensorflow as tf
         from tensorflow.keras import layers
         from tensorflow.keras.models import Sequential
         from tensorflow.keras.layers.experimental.preprocessing import Rescaling

         # setup global constants
         DATA_DIR = "./castings_data"
         WEIGHTS_DIR = "./weights"
         RESULTS = "training_results.png"
         EPOCHS = 10
         BATCH_SIZE = 32
         IMG_HEIGHT = 180
         IMG_WIDTH = 180


         def prepare_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
            """
            Generate training and validation datasets from a folder of images.

            Returns:
               train_ds (tf.data.Dataset): A pre-fetched training dataset.
               val_ds (tf.data.Dataset): A pre-fetched validation dataset.
               class_names (List[str]): Names of all classes to be classified.
            """

            data_dir = pathlib.Path(DATA_DIR)

            train_ds = tf.keras.preprocessing.image_dataset_from_directory(
               data_dir,
               validation_split=0.2,
               subset="training",
               seed=123,
               image_size=(IMG_HEIGHT, IMG_WIDTH),
               batch_size=BATCH_SIZE,
            )

            val_ds = tf.keras.preprocessing.image_dataset_from_directory(
               data_dir,
               validation_split=0.2,
               subset="validation",
               seed=123,
               image_size=(IMG_HEIGHT, IMG_WIDTH),
               batch_size=BATCH_SIZE,
            )

            class_names = train_ds.class_names

            return train_ds, val_ds, class_names


         def train_and_save_model(
            train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, class_names: List[str]
         ) -> tf.keras.callbacks.History:
            """
            Train and save a classification model on the provided data.

            Args:
               train_ds (tf.data.Dataset): A pre-fetched training dataset.
               val_ds (tf.data.Dataset): A pre-fetched validation dataset.
               class_names (List[str]): Names of all classes to be classified.

            Returns:
               history (tf.keras.callbacks.History): A History object containing recorded events from
                     model training.
            """

            num_classes = len(class_names)

            model = Sequential(
               [
                     Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                     layers.Conv2D(16, 3, padding="same", activation="relu"),
                     layers.MaxPooling2D(),
                     layers.Conv2D(32, 3, padding="same", activation="relu"),
                     layers.MaxPooling2D(),
                     layers.Conv2D(64, 3, padding="same", activation="relu"),
                     layers.MaxPooling2D(),
                     layers.Dropout(0.2),
                     layers.Flatten(),
                     layers.Dense(128, activation="relu"),
                     layers.Dense(num_classes),
               ]
            )

            model.compile(
               optimizer="adam",
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"],
            )

            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
            model.save(WEIGHTS_DIR)

            return history


         def plot_training_results(history: tf.keras.callbacks.History) -> None:
            """
            Plot training and validation accuracy and loss curves, and save the plot.

            Args:
               history (tf.keras.callbacks.History): A History object containing recorded events from
                     model training.
            """
            acc = history.history["accuracy"]
            val_acc = history.history["val_accuracy"]
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs_range = range(EPOCHS)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label="Training Accuracy")
            plt.plot(epochs_range, val_acc, label="Validation Accuracy")
            plt.legend(loc="lower right")
            plt.title("Training and Validation Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label="Training Loss")
            plt.plot(epochs_range, val_loss, label="Validation Loss")
            plt.legend(loc="upper right")
            plt.title("Training and Validation Loss")
            plt.savefig(RESULTS)


         if __name__ == "__main__":
            train_ds, val_ds, class_names = prepare_data()
            history = train_and_save_model(train_ds, val_ds, class_names)
            plot_training_results(history)

Training the Model
------------------

Train the model by running the following command. 

.. admonition:: Terminal Session

   | \ :blue:`[~user/castings_project]` \ > \ :green:`python train_classifier.py` \

The model will be trained for 10 epochs, and when training is completed, a new ``weights``
folder and ``training_results.png`` will be created:

.. parsed-literal::

   \ :blue:`castings_project/` \ |Blank|
   ├── train_classifier.py
   ├── training_results.png
   ├── \ :blue:`castings_data/` \ |Blank|
   │     ├── \ :blue:`train/` \ |Blank|
   │     ├── \ :blue:`validation/` \ |Blank|
   │     └── \ :blue:`test/` \ |Blank|
   └── \ :blue:`weights/` \ |Blank|
         ├── saved_model.pb
   │     ├── \ :blue:`assets/` \ |Blank|
   │     └── \ :blue:`variables/` \ |Blank|

The plots below indicate that the model has performed well on the validation dataset, and we are
ready to create a custom :mod:`model` node from it.

   .. figure:: /assets/tutorials/training_results.png
      :width: 416
      :alt: Model training results

      Model Training Results

.. _tutorial_converting_to_custom_model_node:

Converting to Custom Model Node
===============================

This section will show you how to convert the trained model into a custom PeekingDuck node. It
assumes that you are already familiar with the process of creating custom nodes, covered in the
earlier :ref:`custom node <tutorial_custom_nodes>` tutorial. 

First, let's create a new PeekingDuck project within the existing ``castings_project`` folder.

.. admonition:: Terminal Session

   | \ :blue:`[~user/castings_project]` \ > \ :green:`peekingduck init` \

Next, we'll use the :greenbox:`peekingduck create-node` command to create a custom node:

.. admonition:: Terminal Session

   | \ :blue:`[~user/castings_project]` \ > \ :green:`peekingduck create-node` \ 
   | Creating new custom node...
   | Enter node directory relative to ~user/castings_project [src/custom_nodes]: \ :green:`⏎` \
   | Select node type (input, augment, model, draw, dabble, output): \ :green:`model` \
   | Enter node name [my_custom_node]: \ :green:`casting_classifier` \
   | 
   | Node directory:	~user/castings_project/src/custom_nodes
   | Node type:	model
   | Node name:	casting_classifier
   | 
   | Creating the following files:
   |    Config file: ~user/castings_project/src/custom_nodes/configs/model/casting_classifier.yml
   |    Script file: ~user/castings_project/src/custom_nodes/model/casting_classifier.py
   | Proceed? [Y/n]: \ :green:`⏎` \
   | Created node!

The ``castings_project`` folder structure should now look like this:

.. parsed-literal::

   \ :blue:`castings_project/` \ |Blank|
   ├── pipeline_config.yml
   ├── train_classifier.py
   ├── training_results.png
   ├── \ :blue:`castings_data/` \ |Blank|
   │     ├── \ :blue:`train/` \ |Blank|
   │     ├── \ :blue:`validation/` \ |Blank|
   │     └── \ :blue:`test/` \ |Blank|
   ├── \ :blue:`src/` \ |Blank|
   │     └── \ :blue:`custom_nodes/` \ |Blank|
   │           ├── \ :blue:`configs/` \ |Blank|
   │           │     └── \ :blue:`model/` \ |Blank|
   │           │           └── casting_classifier.yml
   │           └── \ :blue:`model/` \ |Blank|
   │                 └── casting_classifier.py
   └── \ :blue:`weights/` \ |Blank|
         ├── saved_model.pb
         ├── \ :blue:`assets/` \ |Blank|
         └── \ :blue:`variables/` \ |Blank|

``castings_project`` now contains **two files** that we need to modify to implement our custom
node.

#. **src/custom_nodes/configs/model/casting_classifier.yml**:

   ``casting_classifier.yml`` updated content:

   .. code-block:: yaml
      :linenos:

      input: ["img"]
      output: ["pred_label", "pred_score"]

      weights_parent_dir: weights
      class_label_map: {0: "defective", 1: "normal"}

#. **src/custom_nodes/model/casting_classifier.py**:

   ``casting_classifier.py`` updated content:

   .. container:: toggle

      .. container:: header

         **Show/Hide Code for casting_classifier.py**

      .. code-block:: python
         :linenos:

         """
         Casting classification model.
         """

         from typing import Any, Dict

         import cv2
         import numpy as np
         import tensorflow as tf

         from peekingduck.pipeline.nodes.node import AbstractNode

         IMG_HEIGHT = 180
         IMG_WIDTH = 180


         class Node(AbstractNode):
            """Initializes and uses a CNN to predict if an image frame shows a normal
            or defective casting.
            """

            def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
               super().__init__(config, node_path=__name__, **kwargs)
               self.model = tf.keras.models.load_model(self.weights_parent_dir)

            def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
               """Reads the image input and returns the predicted class label and
               confidence score.

               Args:
                     inputs (dict): Dictionary with key "img".

               Returns:
                     outputs (dict): Dictionary with keys "pred_label" and "pred_score".
               """
               img = cv2.cvtColor(inputs["img"], cv2.COLOR_BGR2RGB)
               img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
               img = np.expand_dims(img, axis=0)
               predictions = self.model.predict(img)
               score = tf.nn.softmax(predictions[0])

               return {
                     "pred_label": self.class_label_map[np.argmax(score)],
                     "pred_score": 100.0 * np.max(score),
               }


.. _tutorial_building_the_complete_solution:

Building The Complete Solution
==============================

Object Detection, etc