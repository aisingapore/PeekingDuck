import os

import logging

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from omegaconf import DictConfig
import hydra

# from hydra.core.hydra_config import HydraConfig

from tensorflow_model import ClassificationModelBuilder

logger = logging.getLogger("TF_test")


class Dataset:
    def __init__(self):
        # self.train_dataset = None
        # self.validation_dataset = None
        self.init_data()

    def init_data(self) -> None:
        # download dataset
        _URL = (
            "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
        )
        path_to_zip = tf.keras.utils.get_file(
            "cats_and_dogs.zip", origin=_URL, extract=True
        )
        PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

        train_dir = os.path.join(PATH, "train")
        print(train_dir)
        validation_dir = os.path.join(PATH, "validation")

        BATCH_SIZE = 32
        IMG_SIZE = (160, 160)  # need to point to data config

        # assign train and validation datasets to the instance
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            shuffle=True,
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            label_mode="categorical",
        )
        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
            validation_dir,
            shuffle=True,
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            label_mode="categorical",
        )

        # data_augmentation = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.RandomFlip("horizontal"),
        #         tf.keras.layers.RandomRotation(0.2),
        #     ]
        # )

        # for image, _ in self.train_dataset.take(1):
        #     plt.figure(figsize=(10, 10))
        #     first_image = image[0]
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        #         plt.imshow(augmented_image[0] / 255)
        #         plt.axis("off")
        # plt.show()


testing_dataset = Dataset()

image_batch, label_batch = next(iter(testing_dataset.train_dataset))
logger.info(image_batch.shape)
logger.info(label_batch.shape)


class TestTrainer:
    def __init__(self, model_cfg: DictConfig, dataset):
        self.config = model_cfg
        self.dataset = dataset
        self.model_builder = ClassificationModelBuilder(self.config)
        self.model = self.model_builder.model
        self.train_dataset = self.dataset.train_dataset
        self.validation_dataset = self.dataset.validation_dataset
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.metrics = ["accuracy"]
        self.compile_model()

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
        logger.info("model compiled!")
        self.model.summary()

    def train(self) -> None:
        loss0, accuracy0 = self.model.evaluate(self.validation_dataset)
        logger.info("initial loss: {:.2f}".format(loss0))
        logger.info("initial accuracy: {:.2f}".format(accuracy0))

        # training
        initial_epochs = 5
        history = self.model.fit(
            self.train_dataset,
            epochs=initial_epochs,
            validation_data=self.validation_dataset,
        )

        # show leaning curve
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Training and Validation Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        plt.ylim([0, 1.0])
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def run(cfg: DictConfig):
    test_trainer = TestTrainer(cfg.model[cfg.framework], testing_dataset)
    test_trainer.train()


if __name__ == "__main__":
    run()


"""
def run(cfg: DictConfig) -> None:
    # print(cfg.keys())
    # logger.info(OmegaConf.to_yaml(cfg))
    # logger.info(f"runtime.output_dir{HydraConfig.get().runtime.output_dir}")
    print(cfg.model.tensorflow)
    classification_model = ImageClassificationModel(cfg.model.tensorflow)
    initial_epochs = 5
    loss0, accuracy0 = classification_model.model.evaluate(
        classification_model.validation_dataset
    )
    classification_model.logger.info("initial loss: {:.2f}".format(loss0))
    classification_model.logger.info("initial accuracy: {:.2f}".format(accuracy0))

    # training
    history = classification_model.model.fit(
        classification_model.train_dataset,
        epochs=initial_epochs,
        validation_data=classification_model.validation_dataset,
    )

    # show leaning curve
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")
    plt.show()


run()
"""
