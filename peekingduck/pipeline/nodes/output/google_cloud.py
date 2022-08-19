"""
Node template for creating custom nodes.
"""
from typing import Any, Dict

import cv2
from google.cloud import storage
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(self.bucket_name)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        filename = inputs["filename"]
        if self.folder_name:
            blob = self.bucket.blob(self.folder_name + "/" + filename)
        else:
            blob = self.bucket.blob(filename)
        img_str = cv2.imencode(".jpg", inputs["img"])[1].tostring()
        blob.upload_from_string(img_str)
        return {}
