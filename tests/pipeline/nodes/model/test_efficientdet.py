import pytest
import numpy as np
from peekingduck.pipeline.nodes.model.efficientdet import Node


@pytest.fixture
def efficientdet(root_dir):
    node = Node({'input': ['img'],
                 'output': ['bboxes', 'bbox_labels', 'bbox_scores'],
                 'root': root_dir,
                 'model_type': 0,
                 'classes': '../weights/efficientdet/coco_90.json',
                 'weights_dir': ['../weights/efficientdet'],
                 'blob_file': 'efficientdet.zip',
                 'graph_files':
                 {
        0: '../weights/efficientdet/efficientdet-d0.pb',
        1: '../weights/efficientdet/efficientdet-d1.pb',
        2: '../weights/efficientdet/efficientdet-d2.pb',
        3: '../weights/efficientdet/efficientdet-d3.pb',
        4: '../weights/efficientdet/efficientdet-d4.pb',
    },
        'size': [512, 640, 768, 896, 1024, 1280, 1408],
        'num_classes': 90,
        'score_threshold': 0.3,
        'detect_ids': [0],
        'MODEL_NODES':
        {'inputs': ['x:0'],
         'outputs': ['Identity:0', 'Identity_1:0', 'Identity_2:0']}
    })

    return node


@pytest.fixture
def create_blank_image():

    def _create_image():
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        return img

    return _create_image


class TestEfficientDet:
    def test_black_image(self, create_blank_image, efficientdet, root_dir):
        blank_image = create_blank_image()
        output = efficientdet.run({'img': blank_image})
        expected_output = {'bboxes': np.empty((0, 4), dtype=np.float32)}
        assert np.array_equal(output['bboxes'], expected_output['bboxes'])
