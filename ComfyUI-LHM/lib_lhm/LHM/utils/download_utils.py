# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-20 14:38:28
# @Function      : auto download 


import os
import tarfile

import requests
from tqdm import tqdm


def extract_tar_file(tar_path, extract_path):
    os.makedirs(extract_path, exist_ok=True)

    print(f"tar... {tar_path}")

    with tarfile.open(tar_path, 'r:tar') as tar:  
        total_files = len(tar.getnames())  
        with tqdm(total=total_files, desc="extracting", unit="file") as bar:
            for member in tar.getmembers():
                tar.extract(member, path=extract_path)
                bar.update(1)  

    print(f"tar {tar_path} done!")

def download_file(url, save_path):


    file_name = os.path.basename(url)
    save_file = os.path.join(save_path, file_name)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  

        total_size = int(response.headers.get('content-length', 0))
        print("download file: ", file_name)
        
        with open(save_file, 'wb') as file, tqdm(
            desc=save_file,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))  

        print(f"download: {save_file}")
    except requests.exceptions.RequestException as e:

        print(f"error: {e}")
        raise FileExistsError(f"not find url: {url}")

    return save_file

def download_extract_tar_from_url(url, save_path='./'):

    save_file = download_file(url, save_path)
    extract_tar_file(save_file, save_path)
    
    if os.path.exists(save_file):
        os.remove(save_file)