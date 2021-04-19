import os
import logging
import numpy as np

from peekingduck.weights_utils import checker, downloader
from .yolo_files.detector import Detector


class YoloModel:
    """Yolo model with model types: v3 and v3tiny"""

    def __init__(self, config):
        super().__init__()

        # check for yolo weights, if none then download into weights folder
        if not checker.has_weights(config['root'],
                                   config['weights_dir']):
            print('---no yolo weights detected. proceeding to download...---')
            downloader.download_weights(config['root'],
                                        config['weights_id'])
            print('---yolo weights download complete.---')

        #get classnames path to read all the classes
        classes_path = os.path.join(config['root'], config['classes'])
        self.class_names = [c.strip() for c in open(classes_path).readlines()]
        self.detect_ids = config['detect_ids']
        print('yolo model detecting ids: {}'.format(self.detect_ids))

        self.detector = Detector(config)

    def predict(self, frame):
        """predict the bbox from frame

        returns:
        object_bboxes(List[Numpy Array]): list of bboxes detected
        object_labels(List[str]): list of string labels of the
            object detected for the corresponding bbox
        object_scores(List(float)): list of confidence scores of the
            object detected for the corresponding bbox
        """
        assert isinstance(frame, np.ndarray)

        # return bboxes, object_bboxes, object_labels, object_scores
        return self.detector.predict_object_bbox_from_image(
            self.class_names, frame, self.detect_ids
        )

