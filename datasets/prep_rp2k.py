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
import cv2

DATA_DIR = '/home/sdb2/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'RP2K')
DATASET_NAME = 'RP2K'


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
    files = os.listdir(DATASET_DIR)
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            dir = os.path.join(root, file)
            print(dir)
            img = cv2.imdecode(np.fromfile(dir, dtype=np.uint8), -1)
            img = img[:, :, 0:3]
            cv2.imwrite(dir, img)

    raise BaseException

    mkdir(os.path.join(DATASET_DIR, 'data'))
    os.system('mv -rf train {}'.format(os.path.join(DATASET_DIR, 'data')))
    os.system('mv -rf val {}'.format(os.path.join(DATASET_DIR, 'data')))

    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    print("Finished")
