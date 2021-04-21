import os
import logging
from typing import Any, Dict
from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread

logger = logging.getLogger(__name__)

class Node(AbstractNode):
    def __init__(self, config):
        super().__init__(config, name='input.recorded')
        self._allowed_extensions = ["jpg", "jpeg", "png", "mp4", "avi"]
        input_source = config['input_source']
        self._resolution = config['resolution']
        self._mirror_image = config['mirror_image']

        self._get_files(input_source)
        self._get_next_input()

    def run(self, inputs: dict):
        '''
        input: ["source"],
        output: ["img", "end"]
        '''
        outputs = self._run_single_file()

        if outputs[self.outputs[1]]:
            self._get_next_input()
            outputs = self._run_single_file()

        return outputs

    def _run_single_file(self) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()

        outputs = {self.outputs[0]: None, self.outputs[1]: True}
        if success:
            outputs = {self.outputs[0]: img, self.outputs[1]: False}

        return outputs

    def _get_files(self, path) -> None:
        self._filepaths = [path]

        if os.path.isdir(path):
            self._filepaths = os.listdir(path)
            self._filepaths = [os.path.join(path, filepath)
                               for filepath in self._filepaths]
            self._filepaths.sort()

    def _get_next_input(self):

        if self._filepaths:
            file_path = self._filepaths.pop(0)
            if self._is_valid_file_type(file_path):
                self.videocap = VideoNoThread(
                    self._resolution,
                    file_path,
                    self._mirror_image
                )
            else:
                logger.warning("Skipping '%s' as it is not an accepted file format %s",
                                file_path,
                                str(self._allowed_extensions)
                                )
                self._get_next_input()

    def _is_valid_file_type(self, filepath):

        if filepath.split(".")[-1] in self._allowed_extensions:
            return True
        return False
