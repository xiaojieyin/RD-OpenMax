import argparse
import os
import random
import shutil
import time
import warnings
import libmr
import math

import numpy as np
# from tensorboardX import SummaryWriter

from torchvision import datasets
from functions import *
from folder import OpenSetImageFolder
from imagepreprocess import *
from model_init import *
from src.representation import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.metrics import roc_curve, roc_auc_score

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ')
parser.add_argument('--input_size', default=224, type=int, metavar='N',
                    help='number of input size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--lr-method', default='step', type=str,
#                     help='method of learning rate')
# parser.add_argument('--lr-params', default=[], dest='lr_params', nargs='*', type=float,
#                     action='append', help='params of lr method')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument('--print-freq', '-p', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
#                     help='use pre-trained model')
# parser.add_argument('--world-size', default=1, type=int,
#                     help='number of distributed processes')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
# #                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='gloo', type=str,
#                     help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--modeldir', default=None, type=str,
                    help='director of checkpoint')
parser.add_argument('--representation', default=None, type=str,
                    help='define the representation method')
parser.add_argument('--num-classes', default=None, type=int,
                    help='define the number of classes')
parser.add_argument('--freezed-layer', default=None, type=int,
                    help='define the end of freezed layer')
# parser.add_argument('--store-model-everyepoch', dest='store_model_everyepoch', action='store_true',
#                     help='store checkpoint in every epoch')
# parser.add_argument('--classifier-factor', default=None, type=int,
#                     help='define the multiply factor of classifier')
parser.add_argument('--benchmark', default=None, type=str,
                    help='name of dataset')
parser.add_argument('--tail-size', default=5, type=int,
                    help='number of tail size')
parser.add_argument('--sqrt', default=None, type=float,
                    help='sqrt for evt')
parser.add_argument('--revt', default=1, type=int,
                    help='robust evt')
parser.add_argument('--attention', default=None, type=str,
                    help='type of attention')


def main():
    global args, openmax_dist
    args = parser.parse_args()
    print(args)
    # writer = SummaryWriter(
    #     log_dir=args.modeldir,
    #     comment=os.path.basename(args.modeldir))

    if args.seed is not None:
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')

    # create model
    if args.representation == 'GAvP':
        if args.input_size >= 128:
            input_dim = 2048 if args.arch.startswith('mpncovresnet50') else 512
        else:
            input_dim = 512 if args.arch.startswith('mpncovresnet50') else 128
        representation = {'function': GAvP,
                          'input_dim': input_dim}
    elif args.representation == 'MPNCOV':
        if args.input_size >= 128:
            input_dim = 2048 if args.arch.startswith('mpncovresnet50') else 512
            dimension_reduction = None
        else:
            input_dim = 512 if args.arch.startswith('mpncovresnet50') else 128
            dimension_reduction = None
        representation = {'function': MPNCOV,
                          'iterNum': 5,
                          'is_sqrt': True,
                          'is_vec': True,
                          'input_dim': input_dim,
                          'dimension_reduction': dimension_reduction}
    elif args.representation == 'BCNN':
        representation = {'function': BCNN,
                          'is_vec': True,
                          'input_dim': 2048}
    elif args.representation == 'CBP':
        representation = {'function': CBP,
                          'thresh': 1e-8,
                          'projDim': 8192,
                          'input_dim': 512}
    else:
        warnings.warn('=> You did not choose a global image representation method!')
        representation = None  # which for original vgg or alexnet

    model = get_model(args.arch,
                      representation,
                      args.num_classes,
                      args.freezed_layer,
                      input_size=args.input_size,
                      attention=args.attention,
                      pretrained=True)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()

    model.load_state_dict(load_checkpoint(os.path.join(args.modeldir, 'model_best.pth.tar')))

    # load data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train_transforms, val_transforms, evaluate_transforms = preprocess_strategy(args.benchmark, args.input_size)

    train_dataset = OpenSetImageFolder(traindir, train_transforms, seed=args.seed, num_classes=args.num_classes)
    test_on_dataset = OpenSetImageFolder(valdir, evaluate_transforms, seed=args.seed, num_classes=args.num_classes,
                                         fold='known')
    test_off_dataset = OpenSetImageFolder(valdir, evaluate_transforms, seed=args.seed, num_classes=args.num_classes,
                                          fold='unknown')

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(
        OpenSetImageFolder(valdir, val_transforms, seed=args.seed, num_classes=args.num_classes),
        batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    dataloader_on = torch.utils.data.DataLoader(
        test_on_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    dataloader_off = torch.utils.data.DataLoader(
        test_off_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    networks = {
        'softmax': model,
        'openmax tail_size=5': model,
        'openmax tail_size=10': model,
        'openmax tail_size=15': model,
        'openmax tail_size=20': model,
        # 'auc': model,
    }

    results = evaluate_openset(networks, dataloader_on, dataloader_off, dataloader_train)

    acc = evaluate_classifier(model, dataloader_val)
    print('acc: {:.4f}'.format(acc))

    text = 'AUC:\t'
    for name, auc in results.items():
        print("{}: {:.4f}".format(name, auc))
        text += "{}: {:.4f}\t".format(name, auc)
    # writer.add_text('evaluate_openset/auc', text)

    # for i, dist in enumerate(openmax_dist):
    #     writer.add_histogram('evaluate_openset/openmax_dist_{}'.format(i), dist)
    #
    # writer.close()


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint['state_dict']


# Open Set Classification
# Given two datasets, one on-manifold and another off-manifold, predict
# whether each item is on-manifold or off-manifold using the discriminator
# or the autoencoder loss.
# Plot an ROC curve for each and report AUC
# dataloader_on: Test set of the same items the network was trained on
# dataloader_off: Separate dataset from a different distribution
def evaluate_openset(networks, dataloader_on, dataloader_off, dataloader_train):
    auc = {}
    for name, net in networks.items():
        net.eval()
        # print("Evaluate open set for {}.".format(name))

        d_scores_on = get_openset_scores(dataloader_on, dataloader_train, name, net, fold='known')
        d_scores_off = get_openset_scores(dataloader_off, dataloader_train, name, net, fold='unknown')

        y_true = np.array([0] * len(d_scores_on) + [1] * len(d_scores_off))
        y_discriminator = np.concatenate([d_scores_on, d_scores_off])

        dic = {
            'predicts': y_discriminator.tolist(),
            'labels': y_true.tolist()
        }
        import json
        f = open('data.json', 'w')
        json.dump(dic, f)
        f.close()

        auc_d, plot_d = plot_roc(y_true, y_discriminator, 'Discriminator ROC vs *')
        # save_plot(plot_d, 'roc_discriminator', **options)

        auc[name] = auc_d

        # print('{:.4f}'.format(auc_d))

    return auc


# softmax: k-classifier
# openmax: k-classifier
def get_openset_scores(dataloader_test, dataloader_train, network_name, net, fold=None):
    if network_name == 'softmax':
        openset_scores = openset_softmax_confidence(dataloader_test, net)
    else:
        openset_scores = openset_weibull(dataloader_test, dataloader_train, net, fold, network_name)
    return openset_scores


def openset_softmax_confidence(dataloader, netC):
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.cuda()

            # compute output
            ## modified by jiangtao xie
            if len(images.size()) > 4:  # 5-D tensor
                bs, crops, ch, h, w = images.size()
                output = netC(images.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = netC(images)

            preds = F.softmax(output, dim=1)
            openset_scores.extend(preds.max(dim=1)[0].data.cpu().numpy())
    return -np.array(openset_scores)


def openset_weibull(dataloader_test, dataloader_train, netC, fold, network_name=None):
    # First generate pre-softmax 'activation vectors' for all training examples
    # print("Weibull: computing features for all correctly-classified training data")
    global args, openmax_dist

    with torch.no_grad():
        activation_vectors = {}
        activation_vectors_temp = {}
        labels_temp = []
        for images, labels in dataloader_train:
            images = images.cuda()
            labels = labels.cuda()
            # compute output
            ## modified by jiangtao xie
            if len(images.size()) > 4:  # 5-D tensor
                bs, crops, ch, h, w = images.size()
                output = netC(images.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = netC(images)
            logits = output
            # logits = netC(images)
            correctly_labeled = (logits.data.max(1)[1] == labels)
            labels_np = labels.cpu().numpy()
            logits_np = logits.data.cpu().numpy()
            for i, label in enumerate(labels_np):
                labels_temp.append(label)
                if label not in activation_vectors_temp:
                    activation_vectors_temp[label] = []
                activation_vectors_temp[label].append(logits_np[i])

                # if not correctly_labeled[i]:
                #     continue
                # If correctly labeled, add this to the list of activation_vectors for this class
                if label not in activation_vectors:
                    activation_vectors[label] = []
                activation_vectors[label].append(logits_np[i])

        if len(activation_vectors.keys()) != args.num_classes:
            print("{} != {}".format(len(activation_vectors.keys()), args.num_classes))
            for label in list(set(labels_temp)):
                if label not in activation_vectors.keys():
                    activation_vectors[label] = activation_vectors_temp[label]

        # print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))
        # for class_idx in activation_vectors:
        #     print("Class {}: {} images".format(class_idx, len(activation_vectors[class_idx])))

        # Compute a mean activation vector for each class
        # print("Weibull computing mean activation vectors...")
        mean_activation_vectors = {}
        for class_idx in activation_vectors:
            mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)

        # Initialize one libMR Wiebull object for each class
        # print("Fitting Weibull to distance distribution of each class")
        weibulls = {}
        weibull_distances = []
        mr_params = {}
        for class_idx in activation_vectors:
            distances = []
            mav = mean_activation_vectors[class_idx]
            for v in activation_vectors[class_idx]:
                distances.append(np.linalg.norm(v - mav))

            # distances.append(np.linalg.norm(v[0] - v[1]))
            weibull_distances.append(distances)

            mr = libmr.MR()
            tail_size = min(len(distances), args.tail_size)
            mr.fit_high(distances, tail_size)
            """
            param_names = ['parmhat[0] -> scale', 'parmhat[1] -> shape', 'parmci[0]', 'parmci[1]', 'parmci[2]', 'parmci[3]',
                           'sign', 'alpha',
                           'iftype', 'fitting_size', 'translate_amount', 'small_score', 'scores_to_drop']
            """
            p = mr.get_params()

            # mr.set_params(p[0], math.pow(p[1], args.sqrt), p[2], p[3], p[4])
            def func(x, a=9.68426338, b=1.35437019, c=2.19824087):
                # def func(x, a=4.131392694772351, b=1.2657457783303367, c=1.091656139010852):
                return a * np.power(x, -b) + c

            if args.revt == 1:
                c11 = func(tail_size)
            else:
                c11 = 1
            mr.set_params(p[0], p[1] / c11, p[2], p[3], p[4])
            """
            p: scale, shape, iftype, translate_amount, small_score
            """
            weibulls[class_idx] = mr
            mr_params[class_idx] = p
            # print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))
        np.save(os.path.join(args.modeldir, 'mr.npy'), np.array(mr_params))
        openmax_dist = weibull_distances
        # np.save(options['result_dir'] + '/log/weibull_distance', np.array(weibull_distances))

        # Apply Weibull score to every logit
        weibull_scores = []
        logits = []
        _labels = []
        classes = activation_vectors.keys()
        activation_vectors_test = {}
        for images, labels in dataloader_test:
            images = images.cuda()
            # compute output
            ## modified by jiangtao xie
            if len(images.size()) > 4:  # 5-D tensor
                bs, crops, ch, h, w = images.size()
                output = netC(images.view(-1, ch, h, w))
                # fuse scores among all crops
                output = output.view(bs, crops, -1).mean(dim=1)
            else:
                output = netC(images)

            batch_logits = output.data.cpu().numpy()
            labels_np = labels.cpu().numpy()
            # batch_logits = netC(images).data.cpu().numpy()
            batch_weibull = np.zeros(shape=batch_logits.shape)
            for activation_vector, label in zip(batch_logits, labels_np):
                weibull_row = np.ones(len(classes))
                for class_idx in classes:
                    mav = mean_activation_vectors[class_idx]
                    dist = np.linalg.norm(activation_vector - mav)
                    # print(dist)
                    weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
                weibull_scores.append(weibull_row)
                logits.append(activation_vector)
                _labels.append(label)
                if label not in activation_vectors_test:
                    activation_vectors_test[label] = []
                activation_vectors_test[label].append(activation_vector)
        weibull_scores = np.array(weibull_scores)
        logits = np.array(logits)
        _labels = np.array(_labels)
        # if os.path.exists(os.path.join(args.modeldir, 'activation_vectors.npy')):
        #     activation_vectors_np = np.load(os.path.join(args.modeldir, 'activation_vectors.npy'),
        #                                     allow_pickle=True).item()
        # else:
        #     activation_vectors_np = {}
        # if fold == 'known':
        #     activation_vectors_np = {}
        # activation_vectors_np.update({'train': activation_vectors})
        # activation_vectors_np.update({'test_{}'.format(fold): activation_vectors_test})
        # activation_vectors_np.update({'logits_{}'.format(fold): logits})
        # activation_vectors_np.update({'weibull_scores_{}'.format(fold): weibull_scores})
        # activation_vectors_np.update({'labels_{}'.format(fold): _labels})

    # np.save(os.path.join(args.modeldir, 'activation_vectors.npy'), activation_vectors_np)

    # The following is as close as possible to the precise formulation in
    #   https://arxiv.org/pdf/1511.06233.pdf
    # N, K = logits.shape
    # alpha = np.ones((N, K))
    # for i in range(N):
    #    alpha[i][logits[i].argsort()] = np.arange(K) / (K - 1)
    # adjusted_scores = alpha * weibull_scores + (1 - alpha)
    # prob_open_set = (logits * (1 - adjusted_scores)).sum(axis=1)
    # return prob_open_set

    # But this is better
    # Logits must be positive (lower w score should mean lower probability)
    # shifted_logits = (logits - np.expand_dims(logits.min(axis=1), -1))
    # adjusted_scores = alpha * weibull_scores + (1 - alpha)
    # openmax_scores = -np.log(np.sum(np.exp(shifted_logits * adjusted_scores), axis=1))
    # return np.array(openmax_scores)

    # Let's just ignore alpha and ignore shifting
    openmax_scores = -np.log(np.sum(np.exp(logits * weibull_scores), axis=1))
    return np.array(openmax_scores)


def plot_roc(y_true, y_score, title="Receiver Operating Characteristic"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    plot = None
    # if options.get('roc_output'):
    #     print("Saving ROC scores to file {}".format(options['roc_output']))
    #     np.save(options['roc_output'], (fpr, tpr))
    return auc_score, plot


def evaluate_classifier(network, dataloader):
    netC = network

    classification_closed_correct = 0
    classification_total = 0
    for images, labels in dataloader:
        with torch.no_grad():
            images = Variable(images.cuda())
            labels = labels.cuda()
            # Predict a classification among known classes
            net_y = netC(images)
            class_predictions = F.softmax(net_y, dim=1)

            _, predicted = class_predictions.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)
    return float(classification_closed_correct) / (classification_total)


if __name__ == '__main__':
    main()
