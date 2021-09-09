#!/bin/env python3

import torch
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import os
import signal
import sys

from tensorboardX import SummaryWriter
from functools import reduce

from sklearn.model_selection import KFold
from multiprocessing import Manager

from octModels.datasets import DatasetJigsawOctUnlbldFifteen, DatasetOct, DatasetJigsawOctUnlbld, DatasetAutoEncOctUnlbld, DatasetOctAROI, DatasetJigsawOctUnlbldAROI
from octModels.transforms import TransformOCTBilinear, JointTransformOCTRandomRotation, TransformGaussianWhiteNoise, TransformJigSawPuzzle
from octModels.utils import get_next_run, get_model_file
import octModels.models.MedT as models 
from octModels.models.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from octModels.models.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from octModels.train_utils import train

from octModels.diceloss import DiceLoss

def main(model_name, dataset, trainmode, crossval, num_folds, log_dir, model_base_path, loadmodel, loadpretrain, train_dataset_path,\
        val_dataset_path, num_epochs, batch_size, val_batch_size, early_stopping, full_patience, testing_interval, remaining_patience, num_blocks,\
        permutation_per_axis):
    """
    main function for training oct models
    """

    assert not (remaining_patience != -1 and (len(loadmodel) == 0 or crossval) or remaining_patience > full_patience), \
            "--remaining_patience can only be set for consecutive training with --loadmodel and cannot be used with --crossval" \
            + ". Also --remaining_patience must not exceed --full_patience - especially if no early stopping is used " \
            + "-> --early_stopping:False (df) and --full_patience:-1 (df) then so must remainig_patience:-1"
    
    assert not (num_blocks != -1 and trainmode == "image_segmentation"), \
            "--num_blocks is only meaningful in combination with trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'}" 
    assert num_blocks != -1 or trainmode == "image_segmentation", \
            "--num_blocks required when trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'} are set"
    assert not (permutation_per_axis and trainmode=="image_segmentation"), \
            "--permutation_per_axis is only meaningful in combination with trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'}" 
    assert not early_stopping and full_patience == -1 or early_stopping and full_patience > 0, \
            "if --early_stopping not set do not set --full_patience, if --early_stopping set --full_patience > 0 as well"

    # choose model instantiation according to train type 
    if model_name == "MedT":
        model_name = "MedT" if not trainmode == "pretrain_auto_enc" else "MedT_pretrain"

    # select datasets and model
    imgchannel = 1    

    if trainmode == "image_segmentation":

        #Dataset declaration
        if model_name == 'MedT':
            img_size = 128
        elif model_name == 'TransUNet':
            img_size = 224
        else:
            raise Exception("Please chose as a model either MedT or TransUNet")
            
        size_transform = TransformOCTBilinear(img_size=(img_size, img_size))
        joint_transform = torchvision.transforms.Compose([JointTransformOCTRandomRotation()])

        image_transform = torchvision.transforms.Compose([TransformGaussianWhiteNoise()])#, TransformJigSawPuzzle()]) 
        if dataset == 'AROI':
            train_dataset = DatasetOctAROI(train_dataset_path, size_transform=size_transform, joint_transform=joint_transform,\
                image_transform=image_transform, normalized=True)
            val_dataset = DatasetOctAROI(val_dataset_path, size_transform=size_transform, joint_transform=joint_transform, \
                    image_transform=image_transform, normalized=True) 
            num_classes = 8
        else:
            train_dataset = DatasetOct(train_dataset_path, size_transform=size_transform, joint_transform=joint_transform,\
                image_transform=image_transform, normalized=True)
            val_dataset = DatasetOct(val_dataset_path, size_transform=size_transform, joint_transform=joint_transform, \
                    image_transform=image_transform, normalized=True)  
            num_classes = 9

        if args.loss == 'cross_entropy':
            print("Using Cross-Entropy Loss")
            criterion = F.cross_entropy
        elif args.loss == 'dice':
            print("Using Dice Loss")
            criterion = DiceLoss(num_classes, with_CE=False)
        else:
            print("Using a combination of both Cross-Entropy and Dice equally")
            criterion = DiceLoss(num_classes, with_CE=True)

        #classes for image segmentation task in oct 2015 data
        #classes are altered by dataset object - the below /empty class is replaced with 0 and fluid 9 is moved down  to vacant class 8
        #aids in data augmentation flipping etc is simply 0 then
        #class 0: not class, classes 1-7: are retinal layers, class 8: fluid 

        #models

        if model_name == 'MedT':
            model = models.axialnet.MedT(img_size = img_size, imgchan = imgchannel, num_classes=num_classes)

        elif model_name == 'TransUNet':
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = num_classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(img_size / 16), int(img_size / 16))
            model = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes).cuda()

        # hyper-params

        learning_rate = 1e-3
        weight_decay = 1e-5
        #batch_size= 4

    elif trainmode == "pretrain_jigsaw":

        # classes for jigsaw puzzle - modelled as image segmentation into the initial blocks ie. all pixels in the block in the block-permuted image
        #                           corresponding to the block i in original image should receive label (i-1) - labels e [0, N-1], where N=num blocks

        num_classes = num_blocks ** 2

        if args.loss == 'cross_entropy':
            print("Using Cross-Entropy Loss")
            criterion = F.cross_entropy
        elif args.loss == 'dice':
            print("Using Dice Loss")
            criterion = DiceLoss(num_classes, with_CE=False)
        else:
            print("Using a combination of both Cross-Entropy and Dice equally")
            criterion = DiceLoss(num_classes, with_CE=True)
        
        if model_name == 'MedT':
            img_size = 128
            size_transform = TransformOCTBilinear(img_size=(img_size, img_size))

            model = models.axialnet.MedT(img_size = img_size, imgchan = imgchannel, num_classes=num_classes)

        elif model_name == 'TransUNet':
            img_size = 224
            size_transform = TransformOCTBilinear(img_size=(img_size, img_size))

            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = num_classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(img_size / 16), int(img_size / 16))
            model = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes).cuda()
        else:
            raise Exception("Please chose as a model either MedT or TransUNet")
        
        # added to keep logging neat as MedT_pretrain and MedT are logged together 
        joint_transform = None#torchvision.transforms.Compose([JointTransformOCTRandomRotation()])

        image_transform = torchvision.transforms.Compose([TransformGaussianWhiteNoise()]) 

        if dataset == "oct2015":
            train_dataset = DatasetJigsawOctUnlbldFifteen(train_dataset_path, size_transform=size_transform, joint_transform=joint_transform,\
                image_transform=image_transform, blocks_per_axis=num_blocks, normalized=True, permutation_per_axis=permutation_per_axis)
            val_dataset =  DatasetJigsawOctUnlbldFifteen(val_dataset_path, size_transform=size_transform, blocks_per_axis=num_blocks, normalized=True,\
                permutation_per_axis=permutation_per_axis)
        elif dataset == "oct2017":
            train_dataset = DatasetJigsawOctUnlbld(train_dataset_path, size_transform=size_transform, joint_transform=joint_transform,\
                image_transform=image_transform, blocks_per_axis=num_blocks, normalized=True, permutation_per_axis=permutation_per_axis)
            val_dataset =  DatasetJigsawOctUnlbld(val_dataset_path, size_transform=size_transform, blocks_per_axis=num_blocks, normalized=True,\
                permutation_per_axis=permutation_per_axis)
        elif dataset == "AROI":
            train_dataset = DatasetJigsawOctUnlbldAROI(train_dataset_path, size_transform=size_transform, joint_transform=joint_transform,\
                image_transform=image_transform, blocks_per_axis=num_blocks, normalized=True, permutation_per_axis=permutation_per_axis)
            val_dataset =  DatasetJigsawOctUnlbldAROI(val_dataset_path, size_transform=size_transform, blocks_per_axis=num_blocks, normalized=True,\
                permutation_per_axis=permutation_per_axis) 




        # hyper-params
        learning_rate = 1e-3
        weight_decay = 1e-5
        #batch_size= 4

    else: #pretrain_auto_enc
        img_size = 128
        size_transform = TransformOCTBilinear((img_size, img_size))
        image_transform = torchvision.transforms.Compose([torchvision.transforms.RandomRotation((0,360)), TransformGaussianWhiteNoise(), TransformJigSawPuzzle()])
        
        # added to keep logging neat as MedT_pretrain and MedT are logged together 
        joint_transform = None

        train_dataset = DatasetAutoEncOctUnlbld(train_dataset_path, size_transform=size_transform, image_transform=image_transform, normalized=True)
        val_dataset = DatasetAutoEncOctUnlbld(val_dataset_path, size_transform=size_transform, normalized=True)  

        # criterion for pretraining for oct unlabelled data set: difference between original and permutations
        criterion = F.mse_loss

        
        model = models.axialnet.MedT_pretrain(img_size = img_size, imgchan = imgchannel)

        #hyper params for optimizer
        learning_rate = 1e-4
        weight_decay = 1e-5
        #batch_size = 4



    
    #set up directories for model saving and logging
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    run_name = f"run_{get_next_run(log_dir)}"
    run_path = os.path.join(log_dir, run_name)
    model_path = os.path.abspath(os.path.join(model_base_path, run_name))
        

    assert len(loadpretrain) == 0 or len(loadmodel) == 0,\
    "Either choose a model for consecutive training (same task) - loadmodel or model trained on pretext task - loadpretrain"
    if len(loadpretrain) > 0:
        # pretrained model specified and tb loaded
        model_file = get_model_file(loadpretrain) 
    elif len(loadmodel) > 0:
        # pretrained model specified and tb loaded
        model_file = get_model_file(loadmodel) 
        prev_run = os.path.basename(loadmodel)
        if "consec" in prev_run:
            # remove _consec_i if this already is a prev run 
            prev_run = "_".join(prev_run.split("_")[0:-2])
        run_name = f"{prev_run}_consec"
        run_path = os.path.join(log_dir, run_name) 
        model_path = os.path.join(model_base_path, run_name)
    


    if os.path.isdir(model_path) or "consec" in model_path:
        # an old model already exists or consecutive models
        i = 0
        while os.path.isdir(f"{model_path}_{i}"):
            i +=1
        model_path = f"{model_path}_{i}" 
        os.makedirs(model_path)
    else:
        os.makedirs(model_path)
    print(f"creating model in {model_path}")

    if run_path.endswith("consec"):
        # used to handle consecutive runs ie when pre-trained model is used
        # every run has an _i suffix
        i = 0
        while os.path.isdir(f"{run_path}_{i}"):
            i +=1
        run_path = f"{run_path}_{i}" 
        os.makedirs(run_path)
    else:
        os.makedirs(run_path)
    print(f"creating logs in {run_path}")


    #cuda device
    device = torch.device("cuda")
    model.to(device)


    if len(loadpretrain) > 0:
        pretrained_dict = torch.load(model_file)
        model_dict = model.state_dict()
        if model_name == 'MedT':
	        model_dict.update({k:v for k,v in pretrained_dict.items() if k in model_dict.keys()\
	                and "adjust" not in k})
        elif model_name == 'TransUNet':
        	model_dict.update({k:v for k,v in pretrained_dict.items() if k in model_dict.keys()\
	                and "segmentation_head" not in k})
        model.load_state_dict(model_dict)
        model.eval()
        print(f"Pretraining with model {loadpretrain}") 
    # if consecutive run (pretraining) then load model weights
    elif len(loadmodel) > 0:
        model.load_state_dict(torch.load(model_file))
        model.eval()
        print(f"consecutively training with model from {model_file}")



    #optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                                weight_decay=weight_decay)


    num_model_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Model parameters: {num_model_params}")

    # setting seed for reproducibility: np, torch - https://discuss.pytorch.org/t/random-seed-initialization/7854/8



    #  hyper parameters

    hparams = {
        "model_name": model_name,
        "batch_size": batch_size,
        "optimizer": " ".join(reduce(lambda acc,x: acc + [x] if len(x) > 0 else acc, \
            str(optimizer).split(" "), [])).replace("\n", " - "),
        "size_transform": str(size_transform),
        "joint_transform": str(joint_transform),
        "image_transform": str(image_transform),
        "patience(early_stop)": full_patience,
        "testing_interval": testing_interval,
        "loadmodel": loadmodel,
        "loadpretrain": loadpretrain
    }
    hparams_metrics = {
        "hparam/loss": None,
        "hparam/acc": None
    }


    writer = SummaryWriter(run_path)

    #register signal with main process
    main_process = os.getpid()
    def signal_handler(sig, frame):
        if main_process == os.getpid():
            print(f"Saving best model and exiting gracefully.")
            if hparams_metrics["hparam/loss"] != None:
                writer.add_hparams(hparam_dict=hparams, metric_dict=hparams_metrics)
            writer.close()
        sys.exit(0)

    # register os signals for which model is tb saved
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    
    # Note on early stopping - only if active early_stopping==True and full_patience > 0 
    # patience in terms of iterations (optim steps / testing_interval) 
    #        ~ full_pat * testing_interval - optim steps of batch_size

    #TODO: figure out obscure errors from multiprocessing when running on instance - dataloader workers
    if crossval:
        joined_dataset = ConcatDataset([train_dataset, val_dataset])
        kfold = KFold(n_splits=num_folds, shuffle=True)
        print('joined_dataset: ', len(joined_dataset))
        its = 0
        for fold, (train_ids, val_ids) in enumerate(kfold.split(joined_dataset)):
            print(f'Fold {fold}')
            print('--------------------')

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            trainloader = DataLoader(joined_dataset, batch_size=batch_size, sampler=train_subsampler) #, num_workers=8, pin_memory=True)
            valloader = DataLoader(joined_dataset, batch_size=val_batch_size, sampler=val_subsampler) #, num_workers=8, pin_memory=True)

            print("trainloader: ", len(trainloader))
            print("valloader: ", len(valloader))

            its = train(trainloader, valloader, model, optimizer, criterion, num_epochs, trainmode, full_patience, testing_interval, writer,\
                hparams_metrics, model_path, model_name, remaining_patience, its)

    else:
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #, num_workers=8, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True) #, num_workers=8, pin_memory=True)
        
        train(trainloader, valloader, model, optimizer, criterion, num_epochs, trainmode, full_patience, testing_interval, writer,\
            hparams_metrics, model_path, model_name, remaining_patience)

                
    if hparams_metrics["hparam/loss"] != None:
        writer.add_hparams(hparam_dict=hparams, metric_dict=hparams_metrics)

    writer.close()


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='training script for oct models')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--model_path', default='../model', type=str,
                        help='directory to save model')
    parser.add_argument('--tensorboard_dir', default="./train_logs", type=str,
                        help='directory for tensorboard logging')
    parser.add_argument('--loadpretrain', default='', type=str, 
                        help='PATH to pre-trained model directory trained on pretext task')
    parser.add_argument('--loadmodel', default='', type=str, 
                        help='PATH of previously trained model directory  (same task) tb used run_i(_consec_j)?')
    parser.add_argument('--trainmode', choices=["image_segmentation", "pretrain_auto_enc", "pretrain_jigsaw"], default="image_segmentation") 
    parser.add_argument('--epochs', required=True, type=int)
    parser.add_argument('--crossval', help="Perform cross-validation", action="store_true")
    parser.add_argument('--num_folds', default=5, type=int, help="Specify the number of fold to split the data")
    parser.add_argument('--full_patience', default=-1, type=int, help="Defines patience parameter for early stopping - if -1 no early stopping used df: -1")
    parser.add_argument('--testing_interval', required=True, type=int, help="Defines the testing interval")
    parser.add_argument('--model', required=True, choices=['MedT', 'TransUNet'], help="model to be used")
    parser.add_argument('--dataset', required=True, choices=['oct2015', 'oct2017', 'AROI'], help="Name of the Dataset being used"\
        +" - essential for Dataloading")
    parser.add_argument('--permutation_per_axis', action="store_true", help="Switch to permutation_per_axis for trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'}")
    parser.add_argument('--num_blocks', default=-1, type=int, help="Number of blocks for trainmode={'pretrain_auto_enc, 'pretrain_jigsaw'}")
    parser.add_argument('--remaining_patience', default=-1, type=int, help="Use to set a specific remaining_patience on consecutive training - requires --loadmodel")
    parser.add_argument('--early_stopping',action="store_true", help="use early stopping - requires specifying full_patience as well")
    parser.add_argument('--loss',default='cross_entropy', choices=['cross_entropy','dice','both'], help='Chose the type of loss')
    args = parser.parse_args()


    model_name = args.model 
    dataset = args.dataset
    trainmode = args.trainmode 
    crossval = args.crossval
    num_folds = args.num_folds

    log_dir = os.path.abspath(args.tensorboard_dir)
    model_base_path = os.path.abspath(args.model_path)
    loadmodel = os.path.abspath(args.loadmodel) if len(args.loadmodel) > 0 else ""
    loadpretrain = os.path.abspath(args.loadpretrain) if len(args.loadpretrain) > 0 else ""
    train_dataset_path = os.path.abspath(args.train_dataset)
    val_dataset_path = os.path.abspath(args.val_dataset)

    # hyper parameters from script
    num_epochs = args.epochs
    batch_size = args.batch_size
    val_batch_size = batch_size

    early_stopping = args.early_stopping
    full_patience = args.full_patience 
    testing_interval = args.testing_interval 
    
    remaining_patience = args.remaining_patience
    num_blocks = args.num_blocks
    permutation_per_axis = args.permutation_per_axis
     
    main(model_name, dataset, trainmode, crossval, num_folds, log_dir, model_base_path, loadmodel, loadpretrain, train_dataset_path, val_dataset_path, num_epochs,\
            batch_size, val_batch_size, early_stopping, full_patience, testing_interval, remaining_patience, num_blocks, permutation_per_axis)
