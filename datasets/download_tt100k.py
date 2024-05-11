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
import json

DATA_DIR = '/home/sdb2/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'TT100K')
DATASET_NAME = 'TT100K'

CATEGORY = ['pl100', 'i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'il90', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'pl120',
            'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po',
            'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pr40', 'w13', 'w32', 'w55', 'w57',
            'w59', 'wo']


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        # print('Creating directory {}'.format(path))
        os.mkdir(path)


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        # os.system('wget -nc {}'.format(url))
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
    print("Downloading TT100K dataset files...")
    download('data', 'data.zip')

    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    print("Parsing train & test split...")
    annotations = json.loads(open('data/annotations.json').read())
    count = 1
    ids = list(annotations['imgs'].keys())
    for i in tqdm(range(len(ids))):
        img = annotations['imgs'][ids[i]]
        if img['path'].split('/')[0] == 'train':
            label = 'train'
        elif img['path'].split('/')[0] == 'test':
            label = 'val'
        else:
            continue

        img_data = Image.open(os.path.join('data', img['path']))
        if len(img_data.split()) != 3:
            img_data = img_data.convert("RGB")
        for obj in img['objects']:
            category = obj['category']
            if category not in CATEGORY:
                continue
            mkdir(os.path.join(DATASET_DIR, label, category))

            bbox = (obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax'])
            img_crop = img_data.crop(bbox)
            img_crop.save(os.path.join(DATASET_DIR, label, category, '{:0>6d}.jpg'.format(count)))

            count += 1

    print("Finished building TT100K dataset")
