from typing import Collection, Callable
import tensorboardX
import torch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

from octModels.utils import beautify_tensorboard_oct

def train(dataloader, valloader, model, optimizer, criterion, num_epochs, trainmode, full_patience, testing_interval, writer,\
    hparam_metrics, model_path, modelname, remaining_patience, its_run=-1):

    

    assert isinstance(dataloader, DataLoader), f"{dataloader.__class__} is not a subclass of {DataLoader}"
    assert isinstance(valloader, DataLoader), f"{valloader.__class__} is not a subclass of {DataLoader}"
    assert isinstance(model, torch.nn.Module), f"{model.__class__} is not a subclass of {torch.nn.Module}"
    assert isinstance(optimizer, torch.optim.Optimizer), f"{optimizer.__class__} is not a subclass of {torch.optim.Optimizer}"
    assert isinstance(criterion, torch.nn.modules.loss._Loss) or isinstance(criterion, Callable),\
        f"{criterion.__class__} is not a subclass of {torch.nn.modules.loss._Loss}"
    assert isinstance(num_epochs,int) and num_epochs > 0, f"num_epochs: {num_epochs} must be positive integer"
    assert isinstance(trainmode,str), f"trainmode:{trainmode} must be a string"
    assert isinstance(full_patience,int) and (full_patience > 0 or full_patience == -1), f"full_patience:{full_patience} must be positive integer or -1 ~ no early stopping"
    assert isinstance(testing_interval,int) and testing_interval > 0, f"testing_interval:{testing_interval} must be positive integer"
    assert isinstance(writer, tensorboardX.SummaryWriter), f"{writer.__class__} must be a subclass of {tensorboardX.SummaryWriter}"
    assert isinstance(hparam_metrics, dict), f"hparam_metrics:{hparam_metrics} must be a subtype of dictionary"
    assert os.path.isdir(model_path), f"model_path:{model_path} must be a valid path"
    assert isinstance(modelname, str), f"modelname:{modelname} must be a string"
    assert isinstance(remaining_patience,int) and (remaining_patience == -1 or remaining_patience > 0), \
            f"remaining_patience:{remaining_patience} must be positive integer or -1"
    assert not (remaining_patience != -1 and its_run != -1), "You cannot set remaining_patience for cross validation"
    assert isinstance(its_run,int) and its_run >= -1, \
            f"its_run:{its_run} must be [0,inf]  or -1"

    patience = full_patience

    # patience is either > 0 or -1 if >0 we allow reset to remaining patience 
    # which is also either > 0 or -1
    # -> patience > 0 : early stopping or patience == -1 : no early stopping
    if patience != -1:
        if remaining_patience != -1:
            patience = remaining_patience
        print(f"Training with early stopping and patience {patience}")
    else:
        print("Training without early stopping.")

    
    

    # no crossval - set to 0 - differentiate from fold 0
    its_run = its_run if its_run != -1 else 0
    its_cur = its_run

    #train running metrics over the number of iterations defined below
    train_running_loss = []
    train_num_samples = []
    train_running_correct_pixels = []
    train_running_pixels = []

    for epoch in range(num_epochs):

        # early stopping
        if patience == 0:
            break

        # num_batches == len(dataloader) == ceil(len(dataset) / batch_size)
        for batch_idx, (images, masks) in enumerate(dataloader):        

            #early stopping
            if patience == 0:
                break
            
            images = images.to(device ='cuda')
            masks = masks.to(device='cuda')

            # forward
            pred = model(images)
            
            loss = criterion(pred, masks)
            #scale by batch size as df reduction for loss functions in pytorch is mean (eg CE, MSE)
            train_running_loss = [*train_running_loss, loss.item() * images.shape[0]][-len(dataloader):]
            train_num_samples = [*train_num_samples, images.shape[0]][-len(dataloader):]
        
        
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if not trainmode == "pretrain_auto_enc":
                # process prediction for image segmentation (NxCxHxW) -> (Nx1xHxW) w/ elment e [0,C-1]
                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.argmax(pred, dim=1, keepdim=True)
                pred = pred.int() 

                # adapt mask to image dimension - is_pretrain != True then masks are actual masks of dim (N,H,W) 
                #   which we need to expand to (N,1,H,W) for futher processing and visualization
                # - is_pretain == True then masks are the original images (unaugmented) and of dim (N,C,H,W)
                masks = masks.reshape(masks.shape[0],1,*masks.shape[1:])

            #accurracy computation
            train_running_correct_pixels = [*train_running_correct_pixels, (pred == masks).sum().item()][-len(dataloader):] 
            train_running_pixels = [*train_running_pixels, torch.ones_like(pred).sum().item()][-len(dataloader):] 
            #print(f"acc: {(pred == masks).sum().item() / torch.ones_like(pred) .sum().item()}")

            # move images from gpu to cpu
            pred = pred.to(device="cpu")
            masks = masks.to(device="cpu")            
            images = images.to(device="cpu")            


            # every testing_interval batch iterations - shift by 1 to get proper batch number
            it = epoch * len(dataloader) + batch_idx + 1
            if it % testing_interval == 0 and it != 1:

                # testing iteration ~ testing_interval batch iterations /  optimizer steps (it)
                its = its_run + it // testing_interval
                # save 16 images of the train data
                writer.add_image('train/Image', beautify_tensorboard_oct(images[0:16]), its)
                writer.add_image('train/Prediction', beautify_tensorboard_oct(pred[0:16]), its)
                writer.add_image('train/GroundTruth', beautify_tensorboard_oct(masks[0:16]), its)
                
                #log train running metrics
                writer.add_scalar('train/loss', sum(train_running_loss) / sum(train_num_samples), its)
                writer.add_scalar('train/acc', sum(train_running_correct_pixels) / sum(train_running_pixels), its) 
                
                # test loop 
                with torch.no_grad():
                    
                    #test epoch metrics
                    test_running_loss = []
                    test_num_samples = []
                    test_running_correct_pixels = []
                    test_running_pixels = []

                    for batch_idx, (images, masks) in enumerate(valloader): 

                        images = images.to(device='cuda')
                        masks = masks.to(device='cuda')

                        pred = model(images)
                        
                        loss = criterion(pred, masks)
                        #scale by batch size as df reduction for loss functions in pytorch is mean (eg CE, MSE)
                        test_running_loss = [*test_running_loss, loss.item() * images.shape[0]][-len(valloader):] 
                        test_num_samples = [*test_num_samples, images.shape[0]][-len(valloader):]

                        
                        if not trainmode == "pretrain_auto_enc":
                            #process prediction cf train loop
                            pred = torch.nn.functional.softmax(pred, dim=1)
                            pred = torch.argmax(pred, dim=1, keepdim=True)
                            pred = pred.int() 
                            #adapting mask dims cf train loop
                            masks = masks.reshape(masks.shape[0],1,*masks.shape[1:])

                        #print(f"runing: cpx: {test_running_correct_pixels}, allpx: {test_running_pixels}")
                        
                        test_running_correct_pixels = [*test_running_correct_pixels, (pred == masks).sum().item()][-len(valloader):]
                        test_running_pixels = [*test_running_pixels, torch.ones_like(pred).sum().item()][-len(valloader):]

                        #print(f"acc: {(pred == masks).sum().item()}, pred:{torch.ones_like(pred) .sum().item()}, acc:{(pred == masks).sum().item()/torch.ones_like(pred) .sum().item()}")
                        #print(f"runing: cpx: {test_running_correct_pixels}, allpx: {test_running_pixels}, acc:{sum(test_running_correct_pixels) / sum(test_running_pixels)}")
                        pred = pred.to(device="cpu")
                        
                        masks = masks.to(device="cpu")            
                        images = images.to(device="cpu")

                        
                        #save image of last batch iteration
                        if batch_idx == len(valloader) - 1:
                            writer.add_image('test/Image',  beautify_tensorboard_oct(images[0:16]), its)
                            writer.add_image('test/Prediction', beautify_tensorboard_oct(pred[0:16]), its)
                            writer.add_image('test/GroundTruth', beautify_tensorboard_oct(masks[0:16]), its)


                    #test metrics computed over entire test set for each iteration
                    test_loss =  sum(test_running_loss) / sum(test_num_samples)
                    test_acc = sum(test_running_correct_pixels) / sum(test_running_pixels)
                    writer.add_scalar('test/loss', test_loss, its)
                    writer.add_scalar('test/acc', test_acc, its)
                    

                    #update patience if early stopping is used ~ patience > 0 
                    if patience > 0:
                        patience -= 1

                    #update the best model
                    if hparam_metrics["hparam/loss"] == None or hparam_metrics["hparam/loss"] > test_loss:
                        hparam_metrics["hparam/loss"] = test_loss
                        hparam_metrics["hparam/acc"] = test_acc
                        hparam_metrics[f"hparam/iter"] = its
                        
                        torch.save(model.state_dict(), os.path.join(model_path,f"{modelname}_best_model.pth")) 
                        patience = full_patience  
                    
                    #log patience if early stopping is used
                    if patience >= 0:
                        writer.add_scalar('patience', patience, its)
                
                print('its [{}/{}], loss:{:.4f}'.format(its, its_run + num_epochs * len(dataloader) // testing_interval, sum(train_running_loss) / sum(train_num_samples)))
                its_cur = its

    return its_cur 
