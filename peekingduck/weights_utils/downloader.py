import os
import urllib
import zipfile
import requests
import re

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
    with zipfile.ZipFile(zip_path, "r") as f:
        for file in tqdm(iterable=f.namelist(), total=len(f.namelist())):
            f.extract(member=file, path=extract_dir)

    os.remove(zip_path)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)