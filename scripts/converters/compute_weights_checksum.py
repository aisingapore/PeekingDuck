import hashlib
from pathlib import Path


# path names to MoveNet and YOLOX TensorRT files
MOVENET_MPL = "movenet_tensorrt/multipose_lightning_fp16"
MOVENET_SPL = "movenet_tensorrt/singlepose_lightning_fp16"
MOVENET_SPT = "movenet_tensorrt/singlepose_thunder_fp16"
YOLOX_TINY = "yolox_tensorrt/yolox-tiny.trt"
YOLOX_S = "yolox_tensorrt/yolox-s.trt"
YOLOX_M = "yolox_tensorrt/yolox-m.trt"
YOLOX_L = "yolox_tensorrt/yolox-l.trt"

MODELS = [MOVENET_MPL, MOVENET_SPL, MOVENET_SPT, YOLOX_TINY, YOLOX_S, YOLOX_M, YOLOX_L]


def sha256sum(path: Path, hash_func: "hashlib._Hash" = None) -> "hashlib._Hash":
    """Hashes the specified file/directory using SHA256. Reads the file in
    chunks to be more memory efficient.

    When a directory path is passed as the argument, sort the folder
    content and hash the content recursively.

    Args:
        path (Path): Path to the file to be hashed.
        hash_func (Optional[hashlib._Hash]): A hash function which uses the
            SHA-256 algorithm.

    Returns:
        (hashlib._Hash): The updated hash function.
    """
    if hash_func is None:
        hash_func = hashlib.sha256()

    if path.is_dir():
        for subpath in sorted(path.iterdir()):
            if subpath.name not in {".DS_Store", "__MACOSX"}:
                hash_func = sha256sum(subpath, hash_func)
    else:
        buffer_size = hash_func.block_size * 1024
        with open(path, "rb") as infile:
            for chunk in iter(lambda: infile.read(buffer_size), b""):
                hash_func.update(chunk)
    return hash_func


def main():
    for model in MODELS:
        model_path = Path(model)
        hash = sha256sum(model_path)
        print(f"{model} -> {hash.hexdigest()}")


if __name__ == "__main__":
    main()
