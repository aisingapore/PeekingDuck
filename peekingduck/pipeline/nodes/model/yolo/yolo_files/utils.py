import numpy as np
import cv2


def draw_outputs(img, outputs, class_names):
    """Draw object bounding box, confident score, and class name on
    the object in the image.

    Input:
        - img:      the image
        - outputs:  the outputs of prediction, which contain the
                    object' bounding box, confident score, and class
                    index
        - class_names: dictionary of classes string names

    Output:
        - img:      the image with the draw of outputs
    """
    boxes, objectness, classes = outputs
    width_height = np.flip(img.shape[0:2])
    for i, oneclass in enumerate(classes):
        x1y1 = tuple((np.array(boxes[i][0:2]) * width_height).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * width_height).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[oneclass], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img
