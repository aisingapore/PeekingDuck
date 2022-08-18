"""
Node template for creating custom nodes.
"""
import base64
from typing import Any, Dict

import cv2
import numpy as np
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        nparr = np.fromstring(base64.b64decode(inputs["message"]["image"]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = inputs["message"]["name"] + inputs["message"]["timestamp"] + ".jpg"

        return {
            "img": img,
            "filename": filename,
            "pipeline_end": False,
            "saved_video_fps": 1.0,
        }
