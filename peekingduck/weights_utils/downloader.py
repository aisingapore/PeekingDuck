"""
Copyright 2021 AI Singapore

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import zipfile
import requests
from tqdm import tqdm

def download_weights(root, url):
    """Download weights for specified url

    Args:
        root (str): root directory of peekingduck
        url (str): url to download weights from
    """

    extract_dir = os.path.join(root, "..", "weights")
    zip_path = os.path.join(root, "..", "weights", "temp.zip")

    download_file_from_google_drive(url, zip_path)

    # search for downloaded .zip file and extract, then delete
    with zipfile.ZipFile(zip_path, "r") as temp:
        for file in tqdm(iterable=temp.namelist(), total=len(temp.namelist())):
            temp.extract(member=file, path=extract_dir)

    os.remove(zip_path)

def download_file_from_google_drive(file_id, destination):
    """Method to download publicly shared files from google drive

    Args:
        file_id (str): file id of google drive file
        destination (str): destination directory of download
    """
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """Method to get confirmation token

    Args:
        response (Response): html response from call

    Returns:
        value (str): confirmation token
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    """Chunk saving of download content. Chunk size set to large
    integer as weights are usually pretty large

    Args:
        response (Reponse): html response
        destination (str): destintation directory of download
    """
    chunk_size = 32768

    with open(destination, "wb") as temp:
        for chunk in tqdm(response.iter_content(chunk_size)):
            if chunk: # filter out keep-alive new chunks
                temp.write(chunk)
