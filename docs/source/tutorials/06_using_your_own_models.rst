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

.. raw:: html

    <h3>Install Prerequisites</h3>



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

         # setup global constants
         DATA_DIR = "./data"
         WEIGHTS_DIR = "./weights"
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

            train_ds = tf.keras.utils.image_dataset_from_directory(
               data_dir,
               validation_split=0.2,
               subset="training",
               seed=123,
               image_size=(IMG_HEIGHT, IMG_WIDTH),
               batch_size=BATCH_SIZE,
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
               data_dir,
               validation_split=0.2,
               subset="validation",
               seed=123,
               image_size=(IMG_HEIGHT, IMG_WIDTH),
               batch_size=BATCH_SIZE,
            )

            class_names = train_ds.class_names
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
                     layers.Rescaling(1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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

            print(model.summary())
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
            plt.savefig("training_results.png")


         if __name__ == "__main__":
            train_ds, val_ds, class_names = prepare_data()
            history = train_and_save_model(train_ds, val_ds, class_names)
            plot_training_results(history)


.. _tutorial_converting_to_custom_model_node:

Converting to Custom Model Node
===============================



.. _tutorial_building_the_complete_solution:

Building The Complete Solution
==============================

Object Detection, etc