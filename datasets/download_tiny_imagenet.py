#!/usr/bin/env python
import os
import random
import numpy as np
import json
from subprocess import check_output
from tqdm import tqdm

DATA_ROOT_DIR = '/home/yinxiaojie/datasets'
DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'TinyImageNet')
DATASET_NAME = 'TinyImageNet'

ANIMAL_CLASSES = [
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    'African elephant, Loxodonta africana',
    'American alligator, Alligator mississipiensis',
    'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    'Arabian camel, dromedary, Camelus dromedarius',
    'Chihuahua',
    'Egyptian cat',
    'European fire salamander, Salamandra salamandra',
    'German shepherd, German shepherd dog, German police dog, alsatian',
    'Labrador retriever',
    'Persian cat',
    'Yorkshire terrier',
    'albatross, mollymawk',
    'baboon',
    'bee',
    'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
    'bison',
    'black stork, Ciconia nigra',
    'black widow, Latrodectus mactans',
    'boa constrictor, Constrictor constrictor',
    'brown bear, bruin, Ursus arctos',
    'bullfrog, Rana catesbeiana',
    'centipede',
    'chimpanzee, chimp, Pan troglodytes',
    'cockroach, roach',
    'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    'dugong, Dugong dugon',
    'feeder, snake doctor, mosquito hawk, skeeter hawk',
    'fly',
    'gazelle',
    'golden retriever',
    'goldfish, Carassius auratus',
    'goose',
    'grasshopper, hopper',
    'guinea pig, Cavia cobaya',
    'hog, pig, grunter, squealer, Sus scrofa',
    'jellyfish',
    'king penguin, Aptenodytes patagonica',
    'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    'lion, king of beasts, Panthera leo',
    'mantis, mantid',
    'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    'orangutan, orang, orangutang, Pongo pygmaeus',
    'ox',
    'scorpion',
    'sea cucumber, holothurian',
    'sea slug, nudibranch',
    'sheep, Ovis canadensis',
    'slug',
    'snail',
    'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    'standard poodle',
    'sulphur butterfly, sulfur butterfly',
    'tabby, tabby cat',
    'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    'tarantula',
    'trilobite',
    'walking stick, walkingstick, stick insect',
]


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
        elif url.endswith('.zip'):
            os.system('ls *zip | xargs -n 1 unzip')


def get_width_height(filename):
    from PIL import Image
    img = Image.open(os.path.expanduser(filename))
    return (img.width, img.height)


def save_dataset(examples, output_filename):
    print("Writing {} items to {}".format(len(examples), output_filename))
    fp = open(output_filename, 'w')
    for example in examples:
        fp.write(json.dumps(example) + '\n')
    fp.close()


if __name__ == '__main__':
    print("Downloading dataset {}...".format(DATASET_NAME))
    mkdir(DATA_ROOT_DIR)
    mkdir(DATASET_DIR)
    mkdir(os.path.join(DATASET_DIR, 'train'))
    mkdir(os.path.join(DATASET_DIR, 'val'))
    os.chdir(DATASET_DIR)

    # Download and extract dataset
    print("Downloading dataset files...")
    download('tiny-imagenet-200', 'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    # Remove extra directory
    # os.system('mv tiny-imagenet-200/* . && rmdir tiny-imagenet-200')

    wnids = open('tiny-imagenet-200/wnids.txt').read().splitlines()

    wnid_names = {}
    for line in open('tiny-imagenet-200/words.txt').readlines():
        wnid, name = line.strip().split('\t')
        wnid_names[wnid] = name

    test_filenames = os.listdir('tiny-imagenet-200/test/images')

    examples = []

    # Collect training examples
    for wnid in os.listdir('tiny-imagenet-200/train'):
        filenames = os.listdir(os.path.join('tiny-imagenet-200/train', wnid, 'images'))
        for filename in filenames:
            file_path = os.path.join(DATASET_DIR, 'tiny-imagenet-200/train', wnid, 'images', filename)
            examples.append({
                'filename': file_path,
                'label': wnid,
                'fold': 'train',
            })

    # Use validation set as a test set
    for line in open('tiny-imagenet-200/val/val_annotations.txt').readlines():
        jpg_name, wnid, x0, y0, x1, y1 = line.split()
        examples.append({
            'filename': os.path.join(DATASET_DIR, 'tiny-imagenet-200/val', 'images', jpg_name),
            'label': wnid,
            'fold': 'val',
        })

    for wnid in wnids:
        mkdir(os.path.join(DATASET_DIR, 'train', wnid))
        mkdir(os.path.join(DATASET_DIR, 'val', wnid))

    for i in tqdm(range(len(examples))):
        e = examples[i]
        os.system('cp {} {}'.format(e['filename'], os.path.join(DATASET_DIR, e['fold'], e['label'])))

    print("Finished building dataset {}".format(DATASET_NAME))
