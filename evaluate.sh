#set -e
:<<!
*****************Instruction*****************
Here you can easily creat a model by selecting
an arbitray backbone model and global method.
You can fine-tune it on your own datasets by
using a pre-trained model.
Modify the following settings as you wish !
*********************************************
!

#***************Backbone model****************
#Our code provides some mainstream architectures:
#alexnet
#vgg family:vgg11, vgg11_bn, vgg13, vgg13_bn,
#           vgg16, vgg16_bn, vgg19_bn, vgg19
#resnet family: resnet18, resnet34, resnet50,
#               resnet101, resnet152
#mpncovresnet: mpncovresnet50, mpncovresnet101
#inceptionv3
#You can also add your own network in src/network

#arch=resnet18atten
#arch=mpncovresnet18

#arch=resnet18
arch=mpncovresnet18atten

input_size=128

# Type of loss
# ce: Cross Entropy loss
# p: Pairwise Confusion loss
# e: Entropic Confusion loss
loss=p
sqrt=1
#*********************************************

#***************global method****************
#Our code provides some global method at the end
#of network:
#GAvP (global average pooling),
#MPNCOV (matrix power normalized cov pooling),
#BCNN (bilinear pooling)
#CBP (compact bilinear pooling)
#...
#You can also add your own method in src/representation

if [ ${arch:0:3} = mpn ]; then
    image_representation=MPNCOV
else
    image_representation=GAvP
fi

# short description of method
description=reproduce
#*********************************************

#*******************Dataset*******************
#Choose the dataset folder
#benchmark=CUB
#num_classes=120

#benchmark=Aircraft
#num_classes=60

#benchmark=Dogs
#num_classes=70  # 70/120

#benchmark=Food
#num_classes=60

#benchmark=iNat
#num_classes=3000  # 3000/5089

#benchmark=CIFAR
#num_classes=6

#benchmark=TinyImageNet
#num_classes=120

#benchmark=ImageNet_LT
#num_classes=1000

#benchmark=ImageNet
#num_classes=100

#benchmark=30_Aircraft
#num_classes=60

if [ $1 = Aircraft ]; then
    benchmark=Aircraft
    num_classes=60
else
    benchmark=TT100K
    num_classes=30
fi

#datadir=/home/yinxiaojie/datasets
#datadir=/media/sdb/_yinxiaojie/datasets
#datadir=/zkti/dataset/ILSVRC2012/images
datadir=./datasets

dataset=$datadir/$benchmark
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=100
#batchsize=100
# The number of total epochs for training
epoch=100



# The inital learning rate
# decreased by step method
lr=1.2e-3
#lr=1.2e-2

lr_method=step
lr_params=100
# log method
# description: lr = logspace(params1, params2, #epoch)

seed=42

tail_size=20

#lr_method=log
#lr_params=-1.1\ -5.0
weight_decay=1e-3
classifier_factor=5
#*********************************************
#if [ "$loss" = "p" ]; then
#modeldir=/home/yinxiaojie/Project/fast-MPN-COV-master/checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-loss_$loss
#else
#modeldir=/home/yinxiaojie/Project/fast-MPN-COV-master/checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize
#fi
#if [ "$loss" = "p" ]; then
#modeldir=/home/linkdata/yinxiaojie/Project/fast-MPN-COV-master/checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-loss_$loss
#else
#modeldir=/home/linkdata/yinxiaojie/Project/fast-MPN-COV-master/checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize
#fi
#if [ "$loss" = "p" ]; then
#modeldir=/home/linkdata/yinxiaojie/Project/fast-MPN-COV-master/checkpoints/30_FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-loss_$loss
#else
#modeldir=/home/linkdata/yinxiaojie/Project/fast-MPN-COV-master/checkpoints/30_FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize
#fi
#if [ "$loss" = "p" ]; then
#modeldir=./checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-loss_$loss
#else
#modeldir=./checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize
#fi
if [ "$loss" = "p" ]; then
modeldir=./checkpoints/$benchmark
else
modeldir=./checkpoints/$benchmark
fi
echo "Start evaluating!"
#*********************************************
#python evaluate.py $dataset\
#                 --benchmark $benchmark\
#                 -a $arch\
#                 --input_size $input_size\
#                 --seed $seed\
#                 -j 8\
#                 -b $batchsize\
#                 --num-classes $num_classes\
#                 --representation $image_representation\
#                 --freezed-layer $freeze_layer\
#                 --modeldir $modeldir\
#                 --benchmark $benchmark\
#                 --tail-size $tail_size\
#                 --sqrt $sqrt\
#                 --revt 1\
#                 | tee $modeldir/log_eval.txt

# 41 44 48 49 51 53
for seed in 9
do
echo "#$seed"
python evaluate.py $dataset\
                 --benchmark $benchmark\
                 -a $arch\
                 --input_size $input_size\
                 --seed $seed\
                 -j 8\
                 -b $batchsize\
                 --num-classes $num_classes\
                 --representation $image_representation\
                 --freezed-layer $freeze_layer\
                 --modeldir $modeldir\
                 --benchmark $benchmark\
                 --tail-size $tail_size\
                 --sqrt $sqrt\
                 --revt 1\
                 | tee $modeldir/log_eval.txt
done

