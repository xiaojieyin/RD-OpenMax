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

arch=resnet18
#arch=mpncovresnet18atten

#input_size=128
input_size=224

# Type of loss
# ce: Cross Entropy loss
# p: Pairwise Confusion loss
# e: Entropic Confusion loss
loss=p
sqrt=1
#*********************************************
attention=Cov
#attention=SE
#attention=ECA
#attention=CBAM
#attention=CA
#attention=A2

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
    revt=1
else
    image_representation=GAvP
    revt=0
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

benchmark=2_Aircraft
num_classes=60

#datadir=/media/sdd/yinxiaojie/datasets
#datadir=./datasets
datadir=/home/linkdata/yinxiaojie/Project/fast-MPN-COV-master/datasets
#datadir=/zkti/dataset/ILSVRC2012/images
#datadir=./datasets

dataset=$datadir/$benchmark
#*********************************************

#****************Hyper-parameters*************

# Freeze the layers before a certain layer.
freeze_layer=0
# Batch size
batchsize=10
#batchsize=100
# The number of total epochs for training
epoch=100



# The inital learning rate
# decreased by step method
lr=1.2e-3
#lr=0.8e-3

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
echo "Start finetuning!"
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
#if [ "$loss" = "p" ]; then
#modeldir=./checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-loss_$loss-$attention
#else
#modeldir=./checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-$attention
#fi
if [ "$loss" = "p" ]; then
modeldir=/media/sdb/yinxiaojie/checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-loss_$loss-$attention
else
modeldir=/media/sdb/yinxiaojie/checkpoints/FromScratch-$benchmark$num_classes-$arch-$input_size-$image_representation-seed$seed-lr$lr-bs$batchsize-$attention
fi


:<<!
!
if [ ${sqrt} = 1 ]; then
#*********************************************
f=false
for file in $modeldir/*.pth.tar
do
  if [ -e "$file" ]
  then
    f=true
    break
  fi
done
if [ "$f" = "false" ]; then

if [ ! -d  "$modeldir" ]; then
mkdir $modeldir
fi
if [ ! -f $modeldir/log_train.txt ]; then
    touch $modeldir/log_train.txt
fi

cp finetune_.sh $modeldir

python main.py $dataset\
               --benchmark $benchmark\
               --pretrained\
               -a $arch\
               --input_size $input_size\
               -p 100\
               --seed $seed\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --modeldir $modeldir\
               --loss $loss\
               --attention $attention\
               | tee $modeldir/log_train.txt

else
checkpointfile=$(ls -rt $modeldir/*.pth.tar | tail -1)

python main.py $dataset\
               --benchmark $benchmark\
               --pretrained\
               -a $arch\
               --input_size $input_size\
               -p 100\
               --seed $seed\
               --epochs $epoch\
               --lr $lr\
               --lr-method $lr_method\
               --lr-params $lr_params\
               -j 8\
               -b $batchsize\
               --num-classes $num_classes\
               --representation $image_representation\
               --freezed-layer $freeze_layer\
               --modeldir $modeldir\
               --classifier-factor $classifier_factor\
               --benchmark $benchmark\
               --resume $checkpointfile\
               --loss $loss\
               --attention $attention\
               | tee $modeldir/log_train.txt

fi

fi



#*********************************************
if [ ! -f $modeldir/log_eval.txt ]; then
    touch $modeldir/log_eval.txt
fi
echo "Start evaluating!"
#
#echo "##########################################################################"
#for seed in 0 1 2 3 4
#do
#    python evaluate.py $dataset\
#               --benchmark $benchmark\
#               -a $arch\
#               --input_size $input_size\
#               --seed $seed\
#               -j 8\
#               -b $batchsize\
#               --num-classes $num_classes\
#               --representation $image_representation\
#               --freezed-layer $freeze_layer\
#               --modeldir $modeldir\
#               --benchmark $benchmark\
#               --tail-size $tail_size\
#               --sqrt $sqrt\
#               --revt 0\
#               | tee $modeldir/log_eval.txt
#done
#
#echo "##########################################################################"
#for seed in 0 1 2 3 4
#do
#  python evaluate.py $dataset\
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
#                  --attention $attention\
#                 | tee $modeldir/log_eval.txt
#done
#
#echo "Done!"

#python plot_dataset.py $dataset\
#               --benchmark $benchmark\
#               -a $arch\
#               --input_size $input_size\
#               --seed $seed\
#               -j 8\
#               -b $batchsize\
#               --num-classes $num_classes\
#               --representation $image_representation\
#               --freezed-layer $freeze_layer\
#               --modeldir $modeldir\
#               --benchmark $benchmark\
#               --tail-size $tail_size\
#               --sqrt $sqrt\
#
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
                 --revt $revt\
                 --attention $attention\
                 | tee $modeldir/log_eval.txt

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
                 --revt 0\
                 --attention $attention\
                 | tee $modeldir/log_eval.txt