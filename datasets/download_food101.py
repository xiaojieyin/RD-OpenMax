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
DATASET_DIR = os.path.join(DATA_DIR, 'Food')
DATASET_NAME = 'Food'


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
    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    # Download and extract dataset
    print("Downloading DeepFashion dataset files...")
    # download('images', 'https://storage.googleapis.com/kaggle-data-sets/1864/33884/upload/images.zip')
    # download('meta', 'https://storage.googleapis.com/kaggle-data-sets/1864/33884/upload/meta.zip')

    print("Parsing train split...")
    with open(os.path.join(DATASET_DIR, 'meta/train.txt'), 'r') as f:
        files = [l.strip() for l in f.readlines()]
        for i in tqdm(range(len(files))):
            line = files[i]
            dir_name, id = line.split('/')
            mkdir(os.path.join(DATASET_DIR, 'train', dir_name))
            os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'images', dir_name, id + '.jpg'),
                                        os.path.join(DATASET_DIR, 'train', dir_name, id + '.jpg')))

    print("Parsing test split...")
    with open(os.path.join(DATASET_DIR, 'meta/test.txt'), 'r') as f:
        files = [l.strip() for l in f.readlines()]
        for i in tqdm(range(len(files))):
            line = files[i]
            dir_name, id = line.split('/')
            mkdir(os.path.join(DATASET_DIR, 'val', dir_name))
            os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'images', dir_name, id + '.jpg'),
                                        os.path.join(DATASET_DIR, 'val', dir_name, id + '.jpg')))

    print("Finished building Food-101 dataset")
