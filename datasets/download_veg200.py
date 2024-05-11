#!/usr/bin/env python
# Downloads the Dogs
import os
import numpy as np
import json
from subprocess import check_output
import random
import imutil
import scipy.io
from tqdm import tqdm
from PIL import Image

DATA_DIR = '/home/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'Veg')
DATASET_NAME = 'Veg'


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        # print('Creating directory {}'.format(path))
        os.mkdir(path)


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        os.system('wget -nc {}'.format(url))
        if url.endswith('.tgz') or url.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')
        elif url.endswith('.zip'):
            os.system('ls *zip | xargs -n 1 unzip')


if __name__ == '__main__':
    print("Cars dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)

    # Download and extract dataset
    print("Downloading Veg dataset files...")
    download('veg200_images', 'veg200.zip')
    download('veg200_lists', 'veg200_lists.zip')

    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    print("Parsing train split...")
    with open(os.path.join(DATASET_DIR, 'veg200_lists/veg_train.txt'), 'r') as f:
        files = [l.strip() for l in f.readlines()]
        for i in tqdm(range(len(files))):
            line = files[i]
            dir_name, _ = line.split(' ')
            category, file_name = dir_name.split('/')
            mkdir(os.path.join(DATASET_DIR, 'train', category))
            os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'veg200_images', repr(dir_name)),
                                        os.path.join(DATASET_DIR, 'train', repr(category), file_name)))

    print("Parsing test split...")
    with open(os.path.join(DATASET_DIR, 'veg200_lists/veg_val.txt'), 'r') as f:
        files = [l.strip() for l in f.readlines()]
        for i in tqdm(range(len(files))):
            line = files[i]
            dir_name, _ = line.split(' ')
            category, file_name = dir_name.split('/')
            mkdir(os.path.join(DATASET_DIR, 'val', category))
            os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'veg200_images', repr(dir_name)),
                                        os.path.join(DATASET_DIR, 'val', repr(category), file_name)))

    print("Finished building Veg200 dataset")
