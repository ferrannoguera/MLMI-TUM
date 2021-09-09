import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import os
import argparse
from tensorboardX import SummaryWriter

from octModels.datasets import DataSetFileOnlyOctUnlbld, DatasetOct, DatasetOctAROI
from octModels.transforms import TransformOCTBilinear
from octModels.utils import beautify_tensorboard_oct_ulbl, beautify_tensorboard_oct, get_model_file
import octModels.models.MedT as models

from octModels.models.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from octModels.models.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


import json

parser = argparse.ArgumentParser(description='training MedT')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--tensorboard_dir', default="", type=str,
                    help='directory for tensorboard logging')
parser.add_argument('--model', default='', type=str, 
                    help='path to directory of trained model to be used in prediciton') 
parser.add_argument('--testmode', choices=["test_on_labelled", "predict_unlabelled"], default="")
parser.add_argument('--modelname', default="MedT", type=str, help="name of the model to use")

args = parser.parse_args()


modelname = args.modelname
batch_size = args.batch_size
testmode = args.testmode
log_dir = args.tensorboard_dir
print(log_dir, testmode)
if log_dir == "":
    if testmode == "predict_unlabelled":
        log_dir = "./predict_logs"
    elif testmode == "test_on_labelled":
        log_dir = "./test_logs"
log_dir = os.path.abspath(log_dir)



model_file = get_model_file(args.model) 

run_name = f"{os.path.basename(args.model)}_{os.path.basename(model_file).replace('.pth','')}"
run_path = os.path.join(log_dir, run_name) 

rn = 0

while os.path.isdir(f"{run_path}_{rn}"):
    rn +=1
run_path = f"{run_path}_{rn}" 
os.makedirs(run_path)
    

imgchannel = 1   


#classes are altered by dataset - the below /empty class is replaced with 0 and fluid 9 is moved down  to vacant class 8
#aids in data augmentation flipping etc is simply 0 then
#class 0: not class, classes 1-7: are retinal layers, class 8: fluid 
if "AROI" in args.dataset:
    num_classes = 8
    print("USING AROI NUM CLASSES, aka 8!")
else:
    num_classes = 9

device = torch.device("cuda")

if modelname == "MedT":
    imgsize = 128 
    model = models.axialnet.MedT(img_size = imgsize, imgchan = imgchannel, num_classes=num_classes)
elif modelname == "TransUNet":
    imgsize = 224 
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(imgsize / 16), int(imgsize / 16))
    model = ViT_seg(config_vit, img_size=imgsize, num_classes=num_classes).cuda()

else:
    raise NotImplementedError(f"{modelname} not implemented. Please choose different model.")

model.to(device)

model.load_state_dict(torch.load(model_file))
model.eval()


writer = SummaryWriter(run_path)

if testmode == "predict_unlabelled":

    size_tf = TransformOCTBilinear((imgsize,imgsize))
    tf = None #TransformOCTRandomRotation()
    dataset = DataSetFileOnlyOctUnlbld(args.dataset, size_transform=size_tf, transform=tf)  

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            
            images = images.to(device='cuda')

            pred = model(images)

            # preprocess images and remove from gpu
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            pred = pred.int() 
            
            pred = pred.detach().cpu()
                    
            images = images.cpu()


            #save image
            writer.add_image('test/Image',  beautify_tensorboard_oct_ulbl(images), batch_idx )
            writer.add_image('test/Prediction', beautify_tensorboard_oct_ulbl(pred), batch_idx )



elif testmode == "test_on_labelled":

    size_tf = TransformOCTBilinear((imgsize,imgsize))
    tf = None #TransformOCTRandomRotation()
    if "AROI" in args.dataset:
        dataset = DatasetOctAROI(args.dataset, size_transform=size_tf, normalized=True)  
    else:
        dataset = DatasetOct(args.dataset, size_transform=size_tf, normalized=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 

    criterion = F.cross_entropy

    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():        
        #test epoch metrics
        test_running_loss = []
        test_num_samples = []
        test_running_correct_pixels = []
        test_running_pixels = []

        for batch_idx, (images, masks) in enumerate(loader): 

            images = images.to(device='cuda')
            masks = masks.to(device='cuda')

            pred = model(images)
            
            loss = criterion(pred, masks)
            #scale by batch size as df reduction for loss functions in pytorch is mean (eg CE, MSE)
            test_running_loss = [*test_running_loss, loss.item() * images.shape[0]][-len(loader):] 
            test_num_samples = [*test_num_samples, images.shape[0]][-len(loader):]

            
            #process prediction cf train loop
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            pred = pred.int() 

            #compute confusion mask for per-class accuracy
            #_, preds = torch.max(pred, 1)
            for t, p in zip(masks.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            #adapting mask dims cf train loop
            masks = masks.reshape(masks.shape[0],1,*masks.shape[1:])

            
            test_running_correct_pixels = [*test_running_correct_pixels, (pred == masks).sum().item()][-len(loader):]
            test_running_pixels = [*test_running_pixels, torch.ones_like(pred).sum().item()][-len(loader):]

            pred = pred.to(device="cpu")
            
            masks = masks.to(device="cpu")            
            images = images.to(device="cpu")

            
            writer.add_image('test/Image',  beautify_tensorboard_oct(images[0:16]), batch_idx)
            writer.add_image('test/Prediction', beautify_tensorboard_oct(pred[0:16]), batch_idx)
            writer.add_image('test/GroundTruth', beautify_tensorboard_oct(masks[0:16]), batch_idx)


        #test metrics computed over entire test set for each iteration
        test_loss =  sum(test_running_loss) / sum(test_num_samples)
        test_acc = sum(test_running_correct_pixels) / sum(test_running_pixels)
        writer.add_scalar('test/loss', test_loss, 0)
        writer.add_scalar('test/acc', test_acc, 0)

        per_class_acc = confusion_matrix.diag()/confusion_matrix.sum(1)

        x = per_class_acc
        y = x.tolist()
        file_name = '52_class_acc.json'
        with open(file_name, 'w') as file_object:
            json.dump(y, file_object)

        for i in range(len(per_class_acc)):
            writer.add_scalar('test/acc/class_' + str(i), per_class_acc[i], 0)


writer.close()

