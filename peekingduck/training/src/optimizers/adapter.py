

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class OptimizersAdapter:

    def __init__(self) -> None:
        pass

    def Adam(self, learning_rate: LearningRateSchedule, parameters):
        print("HELLOOOOO:", parameters)
        return tf.keras.optimizers.Adam(learning_rate=learning_rate, **parameters)


    def SGD(self, learning_rate: LearningRateSchedule, parameters):
        # Parameters
            # learning_rate=0.01,
            # momentum=0.0,
            # nesterov=False,
            # amsgrad=False,
            # weight_decay=None,
            # clipnorm=None,
            # clipvalue=None,
            # global_clipnorm=None,
            # use_ema=False,
            # ema_momentum=0.99,
            # ema_overwrite_frequency=None,
            # jit_compile=True,
            # name='SGD',
            # **kwargs
        return tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate, **parameters)