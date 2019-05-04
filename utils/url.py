import os
import urllib.request
import shutil
from urllib.parse import urlparse

from io import BytesIO

import uuid


def is_absolute_path(url):
    return bool(urlparse(url).netloc)


def download_file(url, temp_dir='./temp'):
    file_name = temp_dir + '/' + str(uuid.uuid4().hex) + '.jpg'
    os.makedirs(temp_dir, exist_ok=True)

    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    return file_name


def read_image_from_url(url, temp_dir='./temp'):
    # file_name = temp_dir + '/' + str(uuid.uuid4().hex) + '.jpg'
    # os.makedirs(temp_dir, exist_ok=True)
    # file_name = uid

    # Download the file from `url` and save it locally under `uid + file_name`:
    # with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        # data = response.read() # a `bytes` object
        # out_file.write(data)

    # return file_name

    with urllib.request.urlopen(url) as response:
        data = response.read()  # a `bytes` object
        data = BytesIO(data)

    return data
