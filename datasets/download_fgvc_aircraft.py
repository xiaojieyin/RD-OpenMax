#!/usr/bin/env python
# Downloads the FGVC Aircraft
import os
import numpy as np
import json
from subprocess import check_output
import random
import imutil
import re

DATA_DIR = '/home/linkdata/yinxiaojie/Project/fast-MPN-COV-master/datasets'
DATASET_DIR = os.path.join(DATA_DIR, 'Aircraft')
DATASET_NAME = 'Aircraft'


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        os.system('wget -nc {}'.format(url))
        if url.endswith('.tgz') or url.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')


def get_width_height(filename):
    from PIL import Image
    img = Image.open(os.path.expanduser(filename))
    return (img.width, img.height)


# def save_dataset(examples):
#     output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
#     print("Writing {} items to {}".format(len(examples), output_filename))
#     fp = open(output_filename, 'w')
#     for example in examples:
#         fp.write(json.dumps(example) + '\n')
#     fp.close()

def save_dataset(examples, output_filename):
    print("Writing {} items to {}".format(len(examples), output_filename))
    fp = open(output_filename, 'w')
    for example in examples:
        fp.write(json.dumps(example) + '\n')
    fp.close()


def train_test_split(image_filenames):
    # Training examples end with 1, test with 0, val with 2
    classes = []
    is_training = [0] * len(image_filenames)
    labels = [0] * len(image_filenames)
    for split in ['train', 'test', 'val']:
        filenames = [line.split()[0] for line in open("images_{}.txt".format(split))]
        variant = [line.split()[1] if len(line.split()[1:]) == 1 else '_'.join(line.split()[1:])
                   for line in open("images_variant_{}.txt".format(split))]
        for i, filename in enumerate(filenames):
            idx = image_filenames.index(filename)
            is_training[idx] = split
            labels[idx] = variant[i]
            if labels[idx] not in classes:
                classes.append(labels[idx])
    return is_training, labels


def get_attribute_names(filename='attributes.txt'):
    lines = open(filename).readlines()
    idx_to_name = {}
    for line in lines:
        idx, name = line.split()
        idx_to_name[int(idx)] = name
    return idx_to_name


def parse_attributes(filename):
    names = get_attribute_names()
    lines = open(filename).readlines()
    examples = {}
    for line in lines:
        tokens = line.split()
        # Note that the array starts at 1
        example_idx = int(tokens[0]) - 1
        if example_idx not in examples:
            examples[example_idx] = {}
        # Index into attribute names table
        attr_idx = int(tokens[1])
        # Value: 0 or 1
        attr_value = int(tokens[2])
        # Certainty Values
        # 1 not visible
        # 2 guessing
        # 3 probably
        # 4 definitely
        attr_certainty = int(tokens[3])
        # How many seconds the turker took
        attr_time = float(tokens[4])
        attr_name = names[attr_idx]
        if attr_name in examples[example_idx]:
            print("Warning: Double-entry for example {} attribute {}".format(
                example_idx, attr_name))
        examples[example_idx][attr_name] = attr_value
    # Format into a list with one entry per example
    return [examples[i] for i in range(len(examples))]


def crop_and_resize(examples):
    # resize_name = 'images_x{}'.format(RESIZE)
    # mkdir(os.path.join(CUB_DIR, resize_name))
    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))
    for i, e in enumerate(examples):
        filename = e['filename']
        img = imutil.load(filename)
        # examples[i]['filename'] = filename.replace('images', resize_name)
        print(examples[i]['filename'])

        # pth, _ = os.path.split(examples[i]['filename'])
        # mkdir(pth)

        # left, top, box_width, box_height = e['box']
        # x0 = int(left)
        # x1 = int(left + box_width)
        # y0 = int(top)
        # y1 = int(top + box_height)
        # img = img[y0:y1, x0:x1, :]

        # H, W, C = img.shape
        # if H >= W:
        #     img = imutil.resize(img, resize_width=RESIZE, resize_height=round(H / W * RESIZE))
        #     H, W, C = img.shape
        #     h0 = random.randint(0, H - W)
        #     img = img[h0:h0 + RESIZE, :, :]
        # else:
        #     img = imutil.resize(img, resize_width=round(W / H * RESIZE), resize_height=RESIZE)
        #     H, W, C = img.shape
        #     w0 = random.randint(0, W - H)
        #     img = img[:, w0:w0 + RESIZE, :]

        # imutil.show(img, display=False, filename=examples[i]['filename'])
        e['label'] = re.sub(r'[\/:*?"<>|]', '_', e['label'])
        if e['fold'] in ['train', 'val']:
            mkdir(os.path.join(DATASET_DIR, 'train', e['label']))
            filename = filename.replace('images', 'train/{}'.format(e['label']))
        else:
            mkdir(os.path.join(DATASET_DIR, 'val', e['label']))
            filename = filename.replace('images', 'val/{}'.format(e['label']))
        imutil.show(img, display=False, filename=filename)
    return examples


if __name__ == '__main__':
    print("FGVC_Aircraft dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)

    # Download and extract dataset
    print("Downloading FGVC_Aircraft dataset files...")
    download('images', 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz')

    if os.path.exists('fgvc-aircraft-2013b'):
        os.system('mv fgvc-aircraft-2013b/data/* . && rm -rf fgvc-aircraft-2013b')

    # Generate CSV file for the full dataset
    lines = open('images_box.txt').readlines()
    image_filenames = [line.split()[0] for line in lines]

    boxes = [[float(w) for w in line.split()[1:]] for line in lines]

    print("Parsing train/test split...")
    is_training, labels = train_test_split(image_filenames)

    os.chdir('../../../../')

    examples = []
    for i in range(len(image_filenames)):
        example = dict()

        example['filename'] = os.path.join(DATASET_DIR, 'images/{}.jpg'.format(image_filenames[i]))

        # width, height = get_width_height(example['filename'])
        # left, top, box_width, box_height = boxes[i]
        # x0 = left / width
        # x1 = (left + box_width) / width
        # y0 = top / height
        # y1 = (top + box_height) / height
        # example['box'] = (x0, x1, y0, y1)
        example['box'] = boxes[i]

        example['label'] = labels[i]

        example['fold'] = is_training[i]

        examples.append(example)

    examples = crop_and_resize(examples)

    # save_dataset(examples, '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME))
    #
    # # Select a random 10, 50, 100 classes and partition them out
    # classes = list(set(e['label'] for e in examples))
    #
    # random.seed(42)
    # for known_classes in [80]:
    #     for i in range(5):
    #         random.shuffle(classes)
    #         known = [e for e in examples if e['label'] in classes[:known_classes]]
    #         unknown = [e for e in examples if e['label'] not in classes[:known_classes]]
    #         save_dataset(known, '{}/{}-known-{}-split{}a.dataset'.format(DATA_DIR, DATASET_NAME, known_classes, i))
    #         save_dataset(unknown,
    #                      '{}/{}-known-{}-split{}b.dataset'.format(DATA_DIR, DATASET_NAME, known_classes, i))

    print("Finished building Aircraft dataset")
