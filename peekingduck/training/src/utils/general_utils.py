# Copyright 2023 AI Singapore
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

# mypy: ignore-errors
import functools
import gc
import inspect
import logging
import os
import random
import sys
import uuid
import zipfile
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple, Union

import imagesize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from contextlib import nullcontext
from tqdm import tqdm
from src.config import TORCH_AVAILABLE, TF_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch import autocast
    import torchvision.transforms.functional as F
else:
    raise ImportError("Called a torch-specific function but torch is not installed.")


## set and get attribute dynamically
# TODO: recursive_setattr and recursive_getattr
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class State(dict):
    """Container object exposing keys as attributes.

    State objects extend dictionaries by enabling values to be accessed by key,
    `state["value_key"]`, or by an attribute, `state.value_key`.

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs)

    def __setattr__(self, key, value) -> None:
        """Setattr method.

        Args:
            key: The input key.
            value: The corresponding value to the key.
        """
        self[key] = value

    def __dir__(self):
        """Method to return all the keys."""
        return self.keys()

    def __getattr__(self, key):
        """Method to access value associated with the key.

        Args:
            key: The input key.
        """
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError from exc


def generate_uuid4() -> str:
    """Generate a random UUID4.

    Returns:
        (str): Random UUID4
    """
    return str(uuid.uuid4())


def download_to(url: str, filename: str, destination_dir: Path) -> None:
    """Downloads publicly shared files from Google Cloud Platform.

    Saves download content in chunks. Chunk size set to large integer as
    weights are usually pretty large.

    Args:
        destination_dir (Path): Destination directory of downloaded file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    with open(destination_dir / filename, "wb") as outfile, requests.get(
        url, stream=True
    ) as response:
        for chunk in tqdm(response.iter_content(chunk_size=32768)):
            if chunk:  # filter out keep-alive new chunks
                outfile.write(chunk)


def extract_file(destination_dir: Path, blob_file: str) -> None:
    """Extracts the zip file to ``destination_dir``.

    Args:
        destination_dir (Path): Destination directory for extraction.
    """
    zip_path = destination_dir / blob_file
    with zipfile.ZipFile(zip_path, "r") as infile:
        file_list = infile.namelist()
        for file in tqdm(file=sys.stdout, iterable=file_list, total=len(file_list)):
            infile.extract(member=file, path=destination_dir)

    os.remove(zip_path)


class HyperParameters:
    """PyTorch Lightning/D2L style to save attributes.

    See https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/core/mixins/hparams_mixin.py
    """

    def save_hyperparameters(self, ignore: Optional[List[Any]] = None):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""

        if ignore is None:
            ignore = []

        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)


def seed_all(seed: int = 1992) -> None:
    """Seed all random number generators."""
    print(f"Using Seed Number {seed}")

    # set PYTHONHASHSEED env var at fixed value
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def seed_worker(_worker_id) -> None:
    """Seed a worker with the given ID."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def convert_path_to_str(path: Union[Path, str]) -> str:
    """Convert Path to str.
    Args:
        path (Union[Path, str]): Path to convert.
    Returns:
        str: Converted path.
    """
    if isinstance(path, (Path, PurePath)):
        return Path(path).as_posix()
    return path


def return_list_of_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    return_string: bool = True,
) -> Union[List[str], List[Path]]:
    """Returns a list of files in a directory based on extensions.
    If extensions is None, all files are returned.

    Note:
        all_image_extensions = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]

    Args:
        directory (Union[str, Path]): The directory to search.
        extensions (Optional[List[str]]): The extension of the files to search for.
            Defaults to None.
        return_string (bool): Whether to return a list of strings or Paths.
            Defaults to True.

    Returns:
        List[str, Path]: List of files in the directory.
    """
    if isinstance(directory, str):
        directory = Path(directory)

    if extensions is None and return_string:
        return [
            f.as_posix()
            for f in directory.resolve().glob("[!__]*/**/[!__]*")
            if f.is_file()
        ]

    if extensions is None and not return_string:
        return [f for f in directory.resolve().glob("[!__]*/**/[!__]*") if f.is_file()]

    if return_string:
        list_of_files = sorted(
            [
                path.as_posix()
                for path in filter(
                    lambda path: path.suffix in extensions,
                    directory.glob("[!__]*/**/[!__]*"),
                )
            ]
        )
    else:
        list_of_files = sorted(
            filter(
                lambda path: path.suffix in extensions,
                directory.glob("[!__]*/**/[!__]*"),
            )
        )
    return list_of_files


def create_dataframe_with_image_info(
    image_dir: List[Path],
    class_name_to_id: Dict[str, int],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Creates a dataframe with image information."""
    data_list = []

    for image_path in tqdm(image_dir):  # image_path is the full abs path
        # get the label
        # assumes that the image_dir is structured as follows:
        # train_dir
        #   - class1
        #       - image1.jpg
        #       - image2.jpg
        #   - class2
        #       - image3.jpg
        #       - image4.jpg
        #   - class3
        #       - ...

        class_name = image_path.parents[0].name
        class_id = int(class_name_to_id[class_name])

        image_path = image_path.as_posix()
        width, height = imagesize.get(image_path)

        data_list.append([image_path, class_id, class_name, width, height])

    df = pd.DataFrame(  # image_path_col_name
        data_list,
        columns=["image_path", "class_id", "class_name", "width", "height"],
    )
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


def free_gpu_memory(
    *args,
) -> None:
    """Delete all variables from the GPU. Clear cache.
    Args:
        model ([type], optional): [description]. Defaults to None.
        optimizer (torch.optim, optional): [description]. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): [description]. Defaults to None.
    """

    if args is not None:
        # Delete all other variables
        # FIXME:TODO: Check my notebook on deleting global vars.
        for arg in args:
            del arg

    gc.collect()
    torch.cuda.empty_cache()


def choose_torch_device() -> str:
    """Convenience routine for guessing which GPU device to run model on"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_precision(device) -> str:
    """Returns an appropriate precision for the given torch device"""
    if device.type == "cuda":
        device_name: str = torch.cuda.get_device_name(device)
        if not ("GeForce GTX 1660" in device_name or "GeForce GTX 1650" in device_name):
            return "float16"
    return "float32"


def choose_autocast(precision: str):
    """Returns an autocast context or nullcontext for the given precision string"""
    # float16 currently requires autocast to avoid errors like:
    # 'expected scalar type Half but found Float'
    if precision == "autocast" or precision == "float16":
        return autocast
    return nullcontext


def set_tensorflow_device() -> Optional[str]:
    if TF_AVAILABLE:
        import tensorflow as tf
    else:
        raise ImportError(
            "Called a tensorflow-specific function but tensorflow is not installed."
        )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            return f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU"
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def get_mean_rgb_values(image: np.ndarray) -> Tuple[float, float, float]:
    """Get the mean RGB values of a single image of (C, H, W)."""
    if image.shape[0] != 3:  # if channel is not first, make it so, assume channels last
        image = image.transpose(0, 3, 1, 2)  # if tensor use permute instead
        # permutation applies the following mapping
        # axis0 -> axis0
        # axis1 -> axis3
        # axis2 -> axis1
        # axis3 -> axis2

    r_channel, g_channel, b_channel = (
        image[0, ...],
        image[1, ...],
        image[2, ...],
    )  # get rgb channels individually

    r_channel, g_channel, b_channel = (
        r_channel.flatten(),
        g_channel.flatten(),
        b_channel.flatten(),
    )  # flatten each channel into one array

    r_mean, g_mean, b_mean = (
        r_channel.mean(axis=None),
        g_channel.mean(axis=None),
        b_channel.mean(axis=None),
    )

    return r_mean, g_mean, b_mean


plt.rcParams["savefig.bbox"] = "tight"


def show(imgs) -> None:
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()


# Logger
def init_logger(
    log_file: str,
    module_name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Initialize logger and save to file.
    Consider having more log_file paths to save, eg: debug.log, error.log, etc.
    Args:
        log_file (str): Where to save the log file.
        module_name (Optional[str]): Module name to be used in logger. Defaults to None.
        level (int): Logging level. Defaults to logging.INFO.
    Returns:
        logging.Logger: The logger object.
    """
    if module_name is None:
        logger: logging.Logger = logging.getLogger(__name__)
    else:
        # get module name, useful for multi-module logging
        logger = logging.getLogger(module_name)

    logger.setLevel(level)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def stratified_sample_df(df, col, n_samples):
    """Stratified by `col` for `n_samples` samples minimun.
    Returns:
         df_: stratified sampled dataframe
    """
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def merge_dict_of_list(dol1, dol2):
    result = dict(dol1, **dol2)
    result.update((k, dol1[k] + dol2[k]) for k in set(dol1).intersection(dol2))
    return result
