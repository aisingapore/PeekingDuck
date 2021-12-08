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
Functions to download model weights.
"""

import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/peekingduck/models"


def download_weights(weights_dir: Path, blob_file: str) -> None:
    """Downloads weights for specified ``blob_file``.

    Args:
        weights_dir (:obj:`Path`): Path to where all weights are stored.
        blob_file (:obj:`str`): Name of file to be downloaded.
    """
    zip_path = weights_dir / "temp.zip"

    download_file_from_blob(blob_file, zip_path)

    # search for downloaded .zip file and extract, then delete
    with zipfile.ZipFile(zip_path, "r") as temp:
        for file in tqdm(iterable=temp.namelist(), total=len(temp.namelist())):
            temp.extract(member=file, path=weights_dir)

    os.remove(zip_path)


def download_file_from_blob(file_name: str, destination: Path) -> None:
    """Downloads publicly shared files from Google Cloud Platform.

    Args:
        file_name (:obj:`str`): Name of file to be downloaded.
        destination (:obj:`Path`): Destination directory of download.
    """
    url = f"{BASE_URL}/{file_name}"
    session = requests.Session()

    response = session.get(url, stream=True)
    save_response_content(response, destination)


def save_response_content(response: requests.Response, destination: Path) -> None:
    """Saves download content in chunks.

    Chunk size set to large integer as weights are usually pretty large.

    Args:
        response (:obj:`requests.Response`): HTML response.
        destination (:obj:`Path`): Destination directory of download.
    """
    chunk_size = 32768

    with open(destination, "wb") as temp:
        for chunk in tqdm(response.iter_content(chunk_size)):
            if chunk:  # filter out keep-alive new chunks
                temp.write(chunk)
