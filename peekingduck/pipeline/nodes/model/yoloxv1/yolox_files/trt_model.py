import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
    """Encapsulation for host CUDA device"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:
    """YoloX TensorRT model class to load model engine and perform inference"""
    def __init__(self, engine_path: str, max_batch_size: int = 1):
        self.dtype = np.float32 # TensorRT support float32, not 64
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        # self.runtime = trt.Runtime(self.logger)
        # self.engine = self.load_engine(self.engine_path, self.runtime)
        self.engine = self.load_engine(self.engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    # def load_engine(engine_path: str, trt_runtime):
    def load_engine(engine_path: str):
        """Load TensorRT model engine file

        Args:
            engine_path (str): engine file full path

        Returns:
            TensorRT engine
        """
        trt.init_libnvinfer_plugins(None, "")
        trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        """Allocate CUDA working memory buffers

        Returns:
            (List[List[bytes], List[bytes], List[bytes], ?]): List of input, output, bindings, stream buffers
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


