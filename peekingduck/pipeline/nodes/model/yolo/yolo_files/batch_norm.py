import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):  # pylint: disable=too-many-ancestors
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, inputs, training=None):
        """Make trainable=False freeze BN for real (the og version is sad)

        Input:
            - inputs: input image matrix
            - training: the training data

        Output:
            - inputs_image: transformed image matrix
        """
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training)
