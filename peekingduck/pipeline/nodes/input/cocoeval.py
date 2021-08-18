# Copyright 2021 AI Singapore
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

"""

"""

import os
import logging
from typing import Any, Dict, List

import cv2
from pycocotools.coco import COCO

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.preprocess import resize_image
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread

class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any]=None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.logger = logging.getLogger(__name__)

        self._allowed_extensions = ["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"]
        self.file_end = False
        self.frame_counter = -1
        self.tens_counter = 10

        self.coco_instance = COCO(config['instances_dir'])

        self.images_dir = config['images_dir']
        self._filepaths = []

        self.evaluation_classes = config['detect_class']

        self.evaluation_type = config['type']
        if self.evaluation_type == "instances":
            self.category = ['all']
        elif self.evaluation_type == "keypoints":
            self.category = ['person']

        self.images_info, self.filename_info = self._load_coco_images()

        self._get_next_input()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        outputs = self._run_single_file()

        approx_processed = round((self.frame_counter/self.videocap.frame_count)*100)
        self.frame_counter += 1

        if approx_processed > self.tens_counter:
            self.logger.info('Approximately Processed: %s%%...', self.tens_counter)
            self.tens_counter += 10

        if self.file_end:
            self.logger.info('Completed processing file: %s', self._file_name)
            self._get_next_input()
            outputs = self._run_single_file()
            self.frame_counter = 0
            self.tens_counter = 10

        return outputs

    def _load_coco_images(self):
        if self.evaluation_classes[0] == 'all':
            self.logger.info(" Using images from all the categories for evaluation.")
            img_ids = sorted(self.coco_instance.getImgIds())
        else:
            self.logger.info(" Using images from: %s", self.evaluation_classes)
            cat_name = self.evaluation_classes
            catIds = self.coco_instance.getCatIds(catNms=cat_name)
            img_ids = self.coco_instance.getImgIds(catIds=catIds)
            self.logger.info(" Using images from: %s", catIds)

        images_info = {}
        filename_info = {}
        prefix = os.path.join(os.getcwd(), self.images_dir)
        for img_id in img_ids:
            img = self.coco_instance.loadImgs(img_id)[0]
            image_dir = os.path.join(prefix, img['file_name'])
            profile = {'image_dir': image_dir,
                       'image_size': (img['width'], img['height'])}
            images_info[img_id] = profile

            filename_info[img['file_name']] = {'id': img_id,
                                                'image_size': (img['width'], img['height'])}

            self._filepaths.append(image_dir)

        return images_info, filename_info

    def _run_single_file(self) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()  # type: ignore

        self.file_end = True
        outputs = {"img": None,
                   "pipeline_end": True,
                   "filename": self._file_name,
                   "saved_video_fps": self._fps,
                   "image_id": self.filename_info[self._file_name]['id'],
                   "image_size": self.filename_info[self._file_name]['image_size'],
                   "coco_instance": self.coco_instance}
        if success:
            self.file_end = False
            if self.resize['do_resizing']:
                img = resize_image(img,
                                   self.resize['width'],
                                   self.resize['height'])
            outputs = {"img": img,
                       "pipeline_end": False,
                       "filename": self._file_name,
                       "saved_video_fps": self._fps,
                       "image_id": self.filename_info[self._file_name]['id'],
                       "image_size": self.filename_info[self._file_name]['image_size'],
                       "coco_instance": None}

        return outputs

    def _get_files(self, path: str) -> None:
        self._filepaths = [path]

        if os.path.isdir(path):
            self._filepaths = os.listdir(path)
            self._filepaths = [os.path.join(path, filepath)
                               for filepath in self._filepaths]
            self._filepaths.sort()

        if not os.path.exists(path):
            raise FileNotFoundError("Filepath does not exist")
        if not self._filepaths:
            raise FileNotFoundError("No Media files available")

    def _get_next_input(self) -> None:

        if self._filepaths:
            file_path = self._filepaths.pop(0)
            self._file_name = os.path.basename(file_path)

            if self._is_valid_file_type(file_path):
                self.videocap = VideoNoThread(
                    file_path,
                    self.mirror_image
                )
                self._fps = self.videocap.fps
            else:
                self.logger.warning("Skipping '%s' as it is not an accepted file format %s",
                                    file_path,
                                    str(self._allowed_extensions)
                                    )
                self._get_next_input()

    def _is_valid_file_type(self, filepath: str) -> bool:

        if filepath.split(".")[-1] in self._allowed_extensions:
            return True
        return False