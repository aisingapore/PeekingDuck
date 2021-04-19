import tensorflow as tf

def transform_images(x_image, size):
    """transform image to size x size

    Input:
        - x_image: input image matrix
        - size: integer size of the image

    Output:
        - x_image: transformed image matrix
    """
    x_image = tf.image.resize(x_image, (size, size))
    x_image = x_image / 255
    return x_image
