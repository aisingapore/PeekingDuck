# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Object detection class using TensorRT yolo ,with custom yolo_layer
plugins, single label model to find license plate object bboxes
"""

import ctypes
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    """
    Simple helper data class that's a little nicer to use than a 2-tuple.
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Detector:
    def __init__(self, config: Dict[str, Any], cuda_ctx=None) -> None:
        """
        Object detection class using yolo model to find object bboxes
        """

        self.config = config
        self.root_dit = config["root"]
        self.model_type = config["model_type"]
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        try:
            ctypes.cdll.LoadLibrary(self.config["yolo_plugin_path"])
        except OSError as error:
            raise SystemExit(
                "ERROR: failed to load ./plugins/libyolo_layer.so.  "
                'Did you forget to do a "make" in the "./plugins/" '
                "subdirectory?"
            ) from error

        self.engine = self._load_engine()
        self.input_shape = self.get_input_shape(self.engine)

        try:
            self.context = self.engine.create_execution_context()
            # self.inputs, self.outputs, self.bindings, self.stream = \
            #     self.allocate_buffers()
        except Exception as error:
            raise RuntimeError("fail to allocate CUDA resources") from error
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def _load_engine(self) -> Any:
        trtbin = self.config["TensorRT_path"][self.config["model_type"]]
        with open(trtbin, "rb") as file, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(file.read())

    def allocate_buffers(self) -> Tuple[List, List, List, Any]:
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        output_idx = 0
        stream = cuda.Stream()
        for binding in self.engine:
            binding_dims = self.engine.get_binding_shape(binding)
            if len(binding_dims) == 4:
                # explicit batch case (TensorRT 7+)
                size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                # implicit batch case (TensorRT 6 or older)
                size = trt.volume(binding_dims) * self.engine.max_batch_size
            else:
                raise ValueError(
                    "bad dims of binding %s: %s" % (binding, str(binding_dims))
                )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                # each grid has 3 anchors, each anchor generates a detection
                # output of 7 float32 values
                assert size % 7 == 0
                outputs.append(HostDeviceMem(host_mem, device_mem))
                output_idx += 1
        assert len(inputs) == 1
        assert len(outputs) == 1
        return inputs, outputs, bindings, stream

    @staticmethod
    def get_input_shape(engine):
        """Get input shape of the TensorRT YOLO engine."""
        binding = engine[0]
        assert engine.binding_is_input(binding)
        binding_dims = engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            return tuple(binding_dims[2:])
        if len(binding_dims) == 3:
            return tuple(binding_dims[1:])
        raise ValueError("bad dims of binding %s: %s" % (binding, str(binding_dims)))

    def do_inference_v2(self):
        """do_inference_v2 (for TensorRT 7.0+)

        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be 2 tuple of (host_mem, device_mem)
        from allocate_buffers function.
        """
        # Transfer input data to the GPU.
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
            for inp in self.inputs
        ]
        # Run inference.
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            for out in self.outputs
        ]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    @staticmethod
    def _nms_boxes(detections, nms_threshold):
        """
        Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
        boxes with their confidence scores and return an array with the
        indexes of the bounding boxes we want to keep.

        # Args
            detections: Nx7 numpy arrays of
                        [[x, y, w, h, box_confidence, class_id, class_prob],
                        ......]
        """
        x_coord = detections[:, 0]
        y_coord = detections[:, 1]
        width = detections[:, 2]
        height = detections[:, 3]
        box_confidences = detections[:, 4]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(
                x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]]
            )
            yy2 = np.minimum(
                y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]]
            )

            width1 = np.maximum(0.0, xx2 - xx1)
            height1 = np.maximum(0.0, yy2 - yy1)
            intersection = width1 * height1
            union = areas[i] + areas[ordered[1:]] - intersection
            iou = intersection / union
            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

    def _post_process(self, trt_outputs: List[np.ndarray]):
        """
        Post process TRT output

        args:
                trt_ouputs: a list of tensors where each tensor contans a
                    multiple of 7 float32 numbers in the order of
                    [x, y, w, h, box_conf_score, class_id, class_prob]
        return:
                bbox: list of bbox coordinate in [x1,y1,x2,y2]
                scores: list of bbox conf score
                classes: list of class id
        """
        detections = []
        for output in trt_outputs:
            detection = output.reshape((-1, 7))
            detection = detection[detection[:, 4] >= 0.1]
            detections.append(detection)
        detections = np.concatenate(detections, axis=0)

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = self._nms_boxes(cls_detections, 0.3)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0
            )

        x = nms_detections[:, 0].reshape(-1, 1)
        y = nms_detections[:, 1].reshape(-1, 1)
        w = nms_detections[:, 2].reshape(-1, 1)
        h = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([x, y, x + w, y + h], axis=1)
        scores = nms_detections[:, 4]
        classes = nms_detections[:, 5]

        # update the labels names of the object detected
        # labels = np.asarray([self.class_labels[int(i)] for i in classes])

        return boxes, classes, scores

    @staticmethod
    def bbox_scaling(bboxes: List[list], scale_factor: float) -> List[list]:
        """
        To scale the width and height of bboxes from v4tiny
        After the conversion of the model in .cfg and .weight file format, from
        Alexey's Darknet repo, to tf model, bboxes are bigger.
        So downscaling is required for a better fit
        """
        for idx, box in enumerate(bboxes):
            x1, y1, x2, y2 = tuple(box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            scaled_x1 = center_x - ((x2 - x1) / 2 * scale_factor)
            scaled_x2 = center_x + ((x2 - x1) / 2 * scale_factor)
            scaled_y1 = center_y - ((y2 - y1) / 2 * scale_factor)
            scaled_y2 = center_y + ((y2 - y1) / 2 * scale_factor)
            bboxes[idx] = [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

        return bboxes

    def predict(self, frame: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect all license plate objects' bounding box from one image

        args:
                image: (Numpy Array) input image

        return:
                boxes: (Numpy Array) an array of bounding box with
                    definition like (x1, y1, x2, y2), in a
                    coordinate system with origin point in
                    the left top corner
                labels: (Numpy Array) an array of class labels
                scores: (Numpy Array) an array of confidence scores
        """
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

        image_data = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))

        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = image_data.transpose((2, 0, 1)).astype(np.float32)
        image_data = image_data / 255.0

        self.inputs[0].host = np.ascontiguousarray(image_data)
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.do_inference_v2()
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        bboxes, labels, scores = self._post_process(trt_outputs)

        return bboxes, labels, scores
