import torch
from torch.utils.data import DataLoader

import os
from tqdm import tqdm 

from .transforms import TransformOCTBilinear
from .datasets import DatasetOct, DataSetFileOnlyOctUnlbld, DataSetFileOnlyOctUnlbldFifteen
def compute_mean_and_std_oct_fifteen(data_path):
    # compute after size_transform -> when used must be on data after the size_transform
    size_transform = TransformOCTBilinear((128,128))
    dataset =  DatasetOct(data_path, size_transform=size_transform, normalized=False)
    return batched_calculate_mean_and_std(dataset)

def compute_mean_and_std_oct_eighteen(data_path):
    # compute after size_transform -> when used must be on data after the size_transform
    size_transform = TransformOCTBilinear((128,128))
    dataset = DataSetFileOnlyOctUnlbld(data_path, size_transform=size_transform, normalized=False)  
    return batched_calculate_mean_and_std(dataset)

def compute_mean_and_std_oct_fifteen_unlabelled(data_path):
    # compute after size_transform -> when used must be on data after the size_transform
    size_transform = TransformOCTBilinear((128,128))
    dataset = DataSetFileOnlyOctUnlbldFifteen(data_path, size_transform=size_transform, normalized=False)  
    return batched_calculate_mean_and_std(dataset)

def batched_calculate_mean_and_std(data_set, batch_size=1):
    """
    Calculate mean and std for training dataset to use in normalization
    images returned by load_fct should be of dims (H,W,C) or (H,W) for large datasets
    based on Welfords method

    dataset must return batches with dim (N,C,H,W)
    """
    mean = None
    std = None
    k = 1
    for i,data in tqdm(enumerate(DataLoader(data_set, batch_size=batch_size))):
        if len(data) > 1:
            # handle dataloader that return image,mask...
            data = data[0]
        if i == 0:
            mean = torch.zeros(data.shape[1])
            std = torch.zeros(data.shape[1])
        
        data = data.permute(1,0,*list(range(2,2+len(data.shape[2:]))))
        data = data.reshape(data.shape[0], -1)
        #print(data.shape)
        # (N,C,H,W) -> (C)
        for px in range(data.shape[1]):
            px = data[:,px]
            #print(f"px: {px}")
            current_mean = mean
            mean += (px - current_mean) / k
            std += (px - current_mean) * (px - mean)
            #print(f"std:{std}")
            k += 1
    return {"mean":mean, "std":torch.sqrt(std / (k-2))}



def calculate_mean_and_std(data_path, load_fct):
    """
    Calculate mean and std for training dataset to use in normalization
    images returned by load_fct should be of dims (H,W,C) or (H,W) 
    """
    # assuming small loadable dataset 
    images = torch.Tensor([])
    for f in tqdm(os.listdir(data_path)):
        img = torch.as_tensor(load_fct(os.path.join(data_path,f))).type(torch.uint8).squeeze()
        img = img.reshape(1,*img.shape) if len(img.shape) == 2 else img.squeeze().permute(2,0,1) 
        #print(f"{img.shape}")
        assert len(img.shape) == 3,\
            f"Dimensionality of image {f} is off should have 3 dimensions (C,H,W) (grayscale images are expanded to 3 automatically)- dims are {img.shape}"
        # flatten images along channels ((C,H,W)-> (C,H*W)) then stack along channels -> (C,N*H*W) 
        img = img.reshape(img.shape[0],-1)
        #print(f"{img.shape}")
        images = torch.hstack((images,img))
    assert images.shape[0] > 0, f"no images found in {data_path}"
    print("Tensor built from samples - now computing mean and std")
    #computed over all samples, height and width (N,H,W) 
    return {"mean": images.mean(axis=1), "std": images.std(axis=1)}


def naive_calculate_mean_and_std(data_path, load_fct):
    """
    Calculate mean and std for training dataset to use in normalization
    """
    # assuming small loadable dataset 
    images = [torch.Tensor(load_fct(os.path.join(data_path,f))).squeeze() for f in os.listdir(data_path)]
    assert len(images) > 0, f"no images found in {data_path}"
    # image dim should be (C,H,W)
    images = [img.reshape(1,*img.shape) if len(img.shape) == 2 else img for img in images]
    print(f"{[img.shape for img in images]}")
    assert torch.all(torch.Tensor([len(img.shape) == 3 for img in images])),\
        f"Dimensionality of images is off should have 3 dimensions (C,H,W) (grayscale images are expanded to 3 automatically)- dims are {images[0].shape}"
    # flatten images along channels ((C,H,W)-> (C,H*W)) then stack along channels -> (C,N*H*W) 
    images = torch.hstack([img.reshape(img.shape[0],-1) for img in images])
    #computed over all samples, height and width (N,H,W) 
    return {"mean": images.mean(axis=1), "std": images.std(axis=1)}


