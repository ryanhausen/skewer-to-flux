# MIT License

# Copyright (c) 2022 Computational Astrophysics Research Group

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from functools import reduce
from typing import Tuple

import requests
from tqdm import tqdm

DATA_PATH_RAW = "."


def save_response_content(response: requests.Response, destination: str):
    CHUNK_SIZE = 32768

    pbar_iter = tqdm(
        response.iter_content(CHUNK_SIZE),
        desc=f" Downloading {os.path.split(destination)[1]}",
    )

    def write_chunk(f_obj, chunk):
        f_obj.write(chunk)
        return f_obj

    f = open(destination, "wb")

    reduce(write_chunk, filter(None, pbar_iter), f)


# based on https://stackoverflow.com/a/39225272
def download_item(manifest_item_job: Tuple[str, str]) -> None:
    item, url = manifest_item_job

    if os.path.exists(os.path.join(DATA_PATH_RAW, item)):
        print(os.path.join(DATA_PATH_RAW, item), " exists skipping.")
        return
    else:
        print("Downloading ", os.path.join(DATA_PATH_RAW, item))
        session = requests.Session()

        response = session.get(url, stream=True)

        is_token_cookie = lambda kv: kv[0].startswith("download_warning")
        get_cookie_value = lambda kv: kv[1]

        download_token = next(
            map(get_cookie_value, filter(is_token_cookie, response.cookies.items())),
            None,
        )

        if download_token:
            save_response_content(
                session.get(url, params={"confirm": download_token}),
                os.path.join(DATA_PATH_RAW, item),
            )
        else:
            save_response_content(response, os.path.join(DATA_PATH_RAW, item))


def main():
    import numpy as np

    item = (
        "data_ys.npy",
        "https://drive.google.com/a/ucsc.edu/uc?export=download&id=1yLqbpOQGUg35OtHOLiSL9xFd3EZ_5EjJ&confirm=t",
    )

    if os.path.exists("data_ys.npy"):
        print("data_ys.npy already exists, not downloading.")
    else:
        download_item(item)
        np.save(
            "data_ys.npy",
            np.transpose(np.load("data_ys.npy").astype(np.float32), axes=(0,2,1)),
        )


if __name__ == "__main__":
    main()
