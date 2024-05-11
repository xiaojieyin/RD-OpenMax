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

DATA_DIR = '/home/linkdata/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'Dogs')
DATASET_NAME = 'Dogs'


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
        elif url.endswith('.tar'):
            os.system('ls *tar | xargs -n 1 tar xvf')


if __name__ == '__main__':
    print("Cars dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)
    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))

    # Download and extract dataset
    print("Downloading Dogs dataset files...")
    download('Images', 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')
    download('Annotation', 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar')
    download('file_list.mat', 'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar')

    data1 = scipy.io.loadmat('train_list.mat')
    data2 = scipy.io.loadmat('test_list.mat')

    for class_name in os.listdir('Annotation'):
        mkdir(os.path.join(DATASET_DIR, 'train', class_name))
        mkdir(os.path.join(DATASET_DIR, 'val', class_name))

    print("Parsing train split...")
    for i in tqdm(range(len(data1['file_list']))):
        filename = data1['file_list'][i][0][0]

        os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'Images', filename),
                                    os.path.join(DATASET_DIR, 'train', filename)))

    print("Parsing test split...")
    for i in tqdm(range(len(data2['file_list']))):
        filename = data2['file_list'][i][0][0]

        os.system('cp {} {}'.format(os.path.join(DATASET_DIR, 'Images', filename),
                                    os.path.join(DATASET_DIR, 'val', filename)))

    print("Finished building Dogs dataset")
