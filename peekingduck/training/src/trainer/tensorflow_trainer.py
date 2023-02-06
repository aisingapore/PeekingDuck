# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.metrics import log_loss
from src.trainer.base import Trainer
from hydra.utils import instantiate
from omegaconf import DictConfig

import tensorflow as tf
import logging
from configs import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name

class tensorflowTrainer(Trainer):

    def __init__(self, framework: str = "tensorflow") -> None:
        self.framework = framework
        self.model = None
        self.scheduler = None
        self.opt = None
        self.loss = None
        self.metrics = None
        self.callbacks = None

    def setup(
        self,
        trainer_config: DictConfig,
        model_config: DictConfig,
        callbacks_config: DictConfig,
        metrics_config: DictConfig,
        data_config: DictConfig,
        device: str = "cpu",
    ) -> None:
        """Called when the trainer begins."""
        self.trainer_config = trainer_config[self.framework]


        tf_model = instantiate(model_config[self.framework].model_type, model_config[self.framework])
        self.model = tf_model.create_model(model_config[self.framework])
       
        # scheduler
        decay_steps = 1000
        initial_learning_rate=0.001
        self.scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, decay_steps)

        # init_optimizer
        adam = tf.keras.optimizers.Adam(learning_rate=self.scheduler, beta_1=0.9, beta_2=0.999, amsgrad=False, weight_decay=None, name='Adam')
        self.opt = adam

        # loss    
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.loss = [scce]

        # metric
        acc = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        precision = tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
        recall = tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
        auroc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name=None, dtype=None, thresholds=None, multi_label=False, num_labels=None, label_weights=None, from_logits=False)
        self.metrics = [acc, precision, recall, auroc]

        # callback
        es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_acc")
        self.callbacks = [es]

        self.model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics, callback=self.callbacks) 


    def train(self, train_dl, val_dl):
        # self.model.fit(train_dl, val_dl)
        BATCH_SIZE = 32
        EPOCHS = 10

        history = self.model.fit(train_dl, batch_size = BATCH_SIZE, epochs= EPOCHS, validation_data=val_dl)
        
        return history