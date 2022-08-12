# Copyright 2022 AI Singapore
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

"""TensorRT model for PeekingDuck"""

from typing import Any, List, Tuple
import numpy as np
import tensorrt as trt  # pylint: disable=import-error
import pycuda.driver as cuda  # pylint: disable=import-error

# NB: need below autoinit import to create CUDA context!
import pycuda.autoinit  # pylint: disable=import-error, unused-import


class HostDeviceMem:
    """Encapsulation for host CUDA device"""

    def __init__(self, host_mem: Any, device_mem: Any):
        self.host = host_mem
        self.device = device_mem

    def __str__(self) -> str:
        return f"Host:\n{self.host}\nDevice:\n{self.device}"

    def __repr__(self) -> str:
        return self.__str__()


class TrtModel:  # pylint: disable=too-many-instance-attributes
    """YoloX TensorRT model class to load model engine and perform inference"""

    def __init__(self, engine_path: str, max_batch_size: int = 1):
        self.dtype = np.float32  # TensorRT support float32, not 64
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.engine = self.load_engine(self.engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(engine_path: str) -> Any:
        """Load TensorRT model engine file

        Args:
            engine_path (str): engine file full path

        Returns:
            TensorRT engine
        """
        trt.init_libnvinfer_plugins(None, "")
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as trt_engine_file:
            engine_data = trt_engine_file.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self) -> Tuple[List[Any], List[Any], List[Any], Any]:
        """Allocate CUDA working memory buffers

        Returns:
            (Tuple[List[Any], List[Any], List[Any], Any]): List of input, output,
                                                           bindings, stream buffers
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            )
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, data: np.ndarray, batch_size: int = 1) -> np.ndarray:
        """To allow making inference calls via `model(img)`

        Args:
            data (np.ndarray): input image data
            batch_size (int): inference batch size. Default = 1.

        Returns:
            (np.ndarray): inference result
        """
        data = data.astype(self.dtype)
        np.copyto(self.inputs[0].host, data.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        res = [out.host.reshape(batch_size, -1) for out in self.outputs][0]
        # reshape the linear (1, N) res into YoloX-friendly shape (1, M, 85)
        result = res.reshape(1, -1, 85)
        return result
