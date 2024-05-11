#!/usr/bin/env python
# Downloads the FGVC Aircraft
import os
import numpy as np
import json
from subprocess import check_output
import random
import imutil
import scipy.io
from tqdm import tqdm

DATA_DIR = '/home/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'Cars')
DATASET_NAME = 'Cars'


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


if __name__ == '__main__':
    print("Cars dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)
    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    # Download and extract dataset
    print("Downloading Cars dataset files...")
    download('cars_train', 'http://imagenet.stanford.edu/internal/car196/cars_train.tgz')
    download('cars_test', 'http://imagenet.stanford.edu/internal/car196/cars_test.tgz')
    download('devkit', 'http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz')
    download('cars_test_annos_withlabels.mat', 'http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat')

    os.system('cp cars_test_annos_withlabels.mat devkit/cars_test_annos_withlabels.mat')

    data1 = scipy.io.loadmat(os.path.join('devkit', 'cars_train_annos.mat'))
    data2 = scipy.io.loadmat(os.path.join('devkit', 'cars_test_annos_withlabels.mat'))
    data3 = scipy.io.loadmat(os.path.join('devkit', 'cars_meta.mat'))

    class_names = [str(_)[2:-2].replace(' ', '_').replace('/', '') for _ in list(data3['class_names'])[0]]

    print("Parsing train split...")
    for i in tqdm(range(len(data1['annotations'][0]))):
        filename = data1['annotations'][0][i][5][0]

        label = int(data1['annotations'][0][i][4][0]) - 1
        mkdir(os.path.join(DATASET_DIR, 'train', class_names[label]))
        os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'cars_train', filename),
                                    os.path.join(DATASET_DIR, 'train', class_names[label], filename)))

    print("Parsing test split...")
    for i in tqdm(range(len(data2['annotations'][0]))):
        filename = data2['annotations'][0][i][5][0]

        label = int(data2['annotations'][0][i][4][0]) - 1
        mkdir(os.path.join(DATASET_DIR, 'val', class_names[label]))
        os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'cars_test', filename),
                                    os.path.join(DATASET_DIR, 'val', class_names[label], filename)))

    print("Finished building Cars dataset")
