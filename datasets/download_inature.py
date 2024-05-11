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
import shutil

# DATA_DIR = '/home/yinxiaojie/datasets'
DATA_DIR = '/media/sdb/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'iNat')
DATASET_NAME = 'iNat'


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
    print("iNat dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)

    # Download and extract dataset
    print("Downloading iNat dataset files...")

    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    num_min, num_max = 1000, 0
    train = json.loads(open('train2017.json').read())['images']
    img_list = [img['file_name'].split('/')[2].replace(' ', '_') for img in train]
    category_list = []
    for img in list(set(img_list)):
        # if img_list.count(img) < 50:
        #     print(img_list.count(img))
        if img_list.count(img) < num_min:
            num_min = img_list.count(img)
        if img_list.count(img) > num_max:
            num_max = img_list.count(img)
        category_list.append(img)
    print("#category: {}".format(len(category_list)))
    print('min, max: {} {}'.format(num_min, num_max))
    # raise BaseException

    print("Parsing train split...")
    for i in tqdm(range(len(train))):
        img = train[i]['file_name']
        category, file_name = img.split('/')[2].replace(' ', '_'), img.split('/')[-1]
        if category not in category_list:
            continue
        mkdir(os.path.join(DATASET_DIR, 'train', category))
        # bbx
        shutil.copy(os.path.join(DATASET_DIR, img), os.path.join(DATASET_DIR, 'train', category, file_name))

    print("Parsing test split...")
    test = json.loads(open('val2017.json').read())['images']
    for i in tqdm(range(len(test))):
        img = test[i]['file_name']
        category, file_name = img.split('/')[2].replace(' ', '_'), img.split('/')[-1]
        if category not in category_list:
            continue
        mkdir(os.path.join(DATASET_DIR, 'val', category))
        # bbx
        shutil.copy(os.path.join(DATASET_DIR, img), os.path.join(DATASET_DIR, 'val', category, file_name))

    print("Finished building iNat dataset")
