import pytest
import numpy as np
import numpy.testing as npt
import cv2
from peekingduck.pipeline.nodes.model.efficientdet import Node
from peekingduck.pipeline.nodes.model.efficientdet_d04.efficientdet_files.detector import Detector


@pytest.fixture
def efficientdet_config(root_dir):
    config = {'input': ['img'],
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
    }
    return config


@pytest.fixture
def efficientdet(efficientdet_config):
    node = Node(efficientdet_config)

    return node


@pytest.fixture
def efficientdet_detector(efficientdet_config):
    detector = Detector(efficientdet_config)

    return detector


class TestEfficientDet:
    def test_black_image(self, test_black_image, efficientdet):
        blank_image = cv2.imread(test_black_image)
        output = efficientdet.run({'img': blank_image})
        expected_output = {'bboxes': np.empty((0, 4), dtype=np.float32)}
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output['bboxes'], expected_output['bboxes'])

    def test_no_human_image(self, test_animal_image, efficientdet):
        blank_image = cv2.imread(test_animal_image)
        output = efficientdet.run({'img': blank_image})
        expected_output = {'bboxes': np.empty((0, 4), dtype=np.float32)}
        assert output.keys() == expected_output.keys()
        npt.assert_equal(output['bboxes'], expected_output['bboxes'])

    def test_return_at_least_one_person_and_one_bbox(self, test_human_images, efficientdet):
        for img in test_human_images:
            test_img = cv2.imread(img)
            output = efficientdet.run({'img': test_img})
            assert 'bboxes' in output
            assert output['bboxes'].size != 0

    def test_efficientdet_preprocess(self, create_image, efficientdet_detector):
        test_img1 = create_image((720, 1280, 3))
        test_img2 = create_image((1280, 720, 3))
        actual_img1, actual_scale1 = efficientdet_detector.preprocess(test_img1, 512)
        actual_img2, actual_scale2 = efficientdet_detector.preprocess(test_img2, 512)

        assert actual_img1.shape == (512, 512, 3)
        assert actual_img2.shape == (512, 512, 3)
        assert actual_img1.dtype == np.float32
        assert actual_img2.dtype == np.float32

    def test_efficientdet_postprocess(self, efficientdet_detector):
        output_bbox = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        output_label = np.array([0, 0])
        output_score = np.array([0.9, 0.2])
        network_output = (output_bbox, output_score, output_label)
        scale = 0.5
        img_shape = (720, 1280)
        detect_ids = [0]
        boxes, labels, scores = efficientdet_detector.postprocess(
            network_output, scale, img_shape, detect_ids)

        expected_bbox = np.array([[1, 2, 3, 4]])/scale
        expected_bbox[:, [0, 2]] /= img_shape[1]
        expected_bbox[:, [1, 3]] /= img_shape[0]

        expected_score = np.array([0.9])
        npt.assert_almost_equal(expected_bbox, boxes)
        npt.assert_almost_equal(expected_score, scores)
        npt.assert_equal(np.array(['person']), labels)
