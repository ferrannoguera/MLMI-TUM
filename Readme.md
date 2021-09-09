# Repository for oct models and benchmarks - train, predict scripts - datasets - models
This repository serves to compare oct models and benchmarks.
A train and a test script are provided.
- train.py - training and pretraining models
- test.py  - testing models

## Install

### set up virtual environment
For dependencies see requirements.txt
- $mkdir ../.venv
- $python3 -m venv ../.venv
- $source ../.venv/bin/activate
- $pip install -U pip
- $pip install -r requirements.txt 


## Train
train.py: arguments -
- --train_dataset: path to the train dataset
- --val_dataset: path to the val dataset
- --odel_path: path to directory where model is tb saved
- --tensorboard_dir: directory for tensorboard logging
- --loadpretrain: PATH to pre-trained model directory trained on pretext task
- --loadmodel: PATH of previously trained model directory  (same task) tb used run_i(_consec_j)?
- --trainmode: choices=["image_segmentation", "pretrain_auto_enc", "pretrain_jigsaw"], default="image_segmentation
- --crossval: Perform cross-validation
- --num_folds: Specify the number of fold to split the data
- --full_patience: Defines patience parameter for early stopping - if -1 no early stopping used df: -1
- --testing_interval: Defines the testing interval
- --model: model to be used, choices=['MedT', 'TransUNet'] 
- --dataset: Name of the Dataset being used, choices=['oct2015', 'oct2017', 'AROI']
- --permutation_per_axis: Switch to permutation_per_axis for trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'}
- --num_blocks: Number of blocks for trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'}
- --remaining_patience: Use to set a specific remaining_patience on consecutive training - requires --loadmodel
- --early_stopping: use early stopping - requires specifying full_patience as well
- --loss: Chose the type of loss, choices=['cross_entropy','dice','both']

### Train (default)
python train.py --batch_size={4} --train_dataset={train_dir} --val_dataset={val_dir} --model_path=model --trainmode=image_segmentation --epochs={1000} \
                --model={MedT} --dataset={oct2015} --testing_interval={50} 
### Train with early stopping
python train.py --batch_size={4} --train_dataset={train_dir} --val_dataset={val_dir} --model_path=model --trainmode=image_segmentation --epochs={1000} \
    --model={MedT} --dataset={oct2015} --testing_interval={50} --early_stopping --full_patience={200} 
### Train with cross validation 
python train.py --batch_size={4} --train_dataset={train_dir} --val_dataset={val_dir} --model_path=model --trainmode=image_segmentation --epochs={1000} \
    --model={MedT} --dataset={oct2015} --testing_interval={50} --crossval --num_folds={7}
## Pretrain 
Cross validation and early stopping are also available for pretraining.
### Pretrain Jigsaw Random Permutation
python3 train.py --batch_size={8} --train_dataset={train_dir} --val_dataset={val_dir} --model_path=model --trainmode=pretrain_jigsaw --epochs={1000} \
    --model={MedT} --dataset={oct2017} --testing_interval={50} --num_blocks={4}
### Pretrain Jigsaw  Permutation Per Axis (rows and columns)
python3 train.py --batch_size={8} --train_dataset={train_dir} --val_dataset={val_dir} --model_path=model --trainmode=pretrain_jigsaw --epochs={1000} \
    --model={MedT} --dataset={oct2017} --testing_interval={50} --num_blocks={4} --permutation_per_axis
### Pretrain Auto Encoder
python3 train.py --batch_size={8} --train_dataset={train_dir} --val_dataset={val_dir} --model_path=model --trainmode=pretrain_auto_enc --epochs={1000} \
    --model={MedT} --dataset={oct2017} --testing_interval={50} --num_blocks={4}
## Test on labelled dataset
python test.py --model=model/{runxx} --dataset={test_dir} --batch_size=1 --testmode=test_on_labelled 

## Inference on unlabelled dataset
python test.py --model=model/{runxx} --dataset={test_dir} --batch_size=1 --testmode=predict_unlabelled

## Visualization
tensorboard --logdir={log_dir}

## Data Preprocessing for oct2015 dataset
Preprocessing expected for use with implemented Dataset classes - see https://gitlab.lrz.de/mlmi-vit/oct2015chiudatapreprocessing.git

## Reformulation of the original Jigsaw Puzzle (see https://arxiv.org/abs/1603.09246)

### Reformulation as image segmentation task:

- jigsaw puzzle - modelled as image segmentation into the initial blocks ie. all pixels in the block in the block-permuted image corresponding to the block i in
original image should receive label (i-1) - labels e [0, N-1], where N=num blocks

- as each axis (H,W) is dissected into num_blocks we have num_blocks ** 2 blocks in total ~ num_classes

- Note: the reformulation is a harder task than classical jigsaw puzzle as the network needs to also learn the block-wise relationship between pixels,
ie. we can map the resulting permutated image mask to the blocksequence by majority vote on each block sequentially

### Deviations from classical Jigsaw puzzle apart from reformulation
- no fixed sequence of permutations - the permutation sequence is randomly generated for each sample
- first variant below only: permutation per axis

### Two implemented variants

#### Permutation per axis
-permutation is restricted to column-wise and row-wise permutations of an image (H,W) as this can be efficiently implemented with matrix multiplication
ie the rows of the image are block-wise permutated and then the columns of the image are block-wise permutated
- use --trainmode=pretrain_jigsaw --permutation_per_axis

#### Random Permutation (similar to original formulation)
- Permutation of all blocks randomly across the image, mask respectively.
- use --trainmode=pretrain_jigsaw 

