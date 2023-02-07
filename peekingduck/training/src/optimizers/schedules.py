"""
Scheduler Class to interact with Tensorflow

        # values = [1.0, 0.5, 0.1]
        # boundaries = [100000, 110000]
        # self.scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #     boundaries, values, name=None
        # )


        # decay_steps = 1.0
        # decay_rate = 0.5
        # initial_learning_rate = 0.001
        # first_decay_steps = 1000
        # initial_learning_rate = 0.1

        # self.scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None
        # )

        # self.scheduler = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate, decay_steps
        # )


        # 
        # self.scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(
        #     initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None
        # )
        
        # self.scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        #     initial_learning_rate,
        #     decay_steps,
        #     end_learning_rate=0.0001,
        #     power=1.0,
        #     cycle=False,
        #     name=None
        # )

        # self.scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        #     initial_learning_rate,
        #     first_decay_steps,
        #     t_mul=2.0,
        #     m_mul=1.0,
        #     alpha=0.0,
        #     name=None
        # )

"""

import tensorflow as tf

class OptimizerSchedules:

    @staticmethod
    def PiecewiseConstantDecay(parameters):
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(**parameters)

    @staticmethod
    def ExponentialDecay(parameters):
        return tf.keras.optimizers.schedules.ExponentialDecay(**parameters)

    @staticmethod
    def CosineDecay(parameters):
        return tf.keras.optimizers.schedules.CosineDecay(**parameters)

    @staticmethod
    def InverseTimeDecay(parameters):
        return tf.keras.optimizers.schedules.InverseTimeDecay(**parameters)

    @staticmethod
    def PolynomialDecay(parameters):
        return tf.keras.optimizers.schedules.PolynomialDecay(**parameters)

    @staticmethod
    def CosineDecayRestarts(parameters):
        return tf.keras.optimizers.schedules.CosineDecayRestarts(**parameters)

