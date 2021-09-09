import torchvision
import torch
from torch.utils.data import Dataset
import numpy as np

import cv2
from skimage import color

import os
from typing import Callable

from .transforms import JointTransformJigSawRandom, TransformOCTMaskAdjustment, TransformOCTMaskAdjustment_AROI, TransformStandardization, JointTransformJigSawPuzzle
from .utils import get_files


class DatasetOct(Dataset):
    """
    Map style dataset object for oct 2015 data
    - expects .npy files
    - assumes sliced images, masks - produced by our project: dataloading/preprocessing.py 
        (valid dims of images,masks and encoding: pixel label e [0..9], for every pixel)

    Parameters:
        dataset_path: path to the dataset path/{images,masks}
        size_transform: deterministic transformation for resizing applied to image and mask separately
        joint_transform: random transformations applied to image and mask jointly after size_transform 
        image_transform: transformation applied only to the image and after joint_transform
    _getitem__(): returns image and corresponding mask 
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, size_transform: Callable = None, image_transform: Callable =None, normalized=True) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = get_files(self.input_path, ".npy")

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.joint_transform = joint_transform

        self.mask_adjust = TransformOCTMaskAdjustment()

        self.image_transform = image_transform

        self.normalized = normalized
        #gray scale oct 2015: calculated with full tensor in memory {'mean': tensor([46.3758]), 'std': tensor([53.9434])}
        # calculated with batched method {'mean': tensor([46.3756]), 'std': tensor([53.9204])}
        self.normalize = TransformStandardization((46.3758), (53.9434))#torchvision.transforms.Normalize((46.3758), (53.9434)) 

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        img = np.load(os.path.join(self.input_path, image_filename))
        mask =  np.load(os.path.join(self.output_path, image_filename))
        
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()
        mask = mask.squeeze()
        

        img = torch.Tensor(img).reshape(1, 1, *img.shape)
        mask = torch.Tensor(mask).reshape(1, 1, *mask.shape).int()

        # adjust mask classes
        mask = self.mask_adjust(mask)


        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)
            mask = self.size_transform(mask)

        #normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.joint_transform:
            img, mask = self.joint_transform([img, mask])
        

        if self.image_transform:
            img = self.image_transform(img)
        

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze().long()
        
        #print(f"img: {torch.unique(img)}")

        return img, mask


class DatasetOctAROI(Dataset):
    """
    Map style dataset object for pretraining oct 2018 unlabeled data
    - expects .png files

    Parameters:
        dataset_path: path to the dataset path/{images,masks}
        size_transform: deterministic transformation for resizing applied to image and mask separately
        joint_transform: random transformations applied to image and mask jointly after size_transform 
        image_transform: transformation applied only to the image and after joint_transform
    _getitem__(): returns image and corresponding mask 
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, size_transform: Callable = None, image_transform: Callable =None, normalized=True) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = get_files(self.input_path, ".png")

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.joint_transform = joint_transform

        self.mask_adjust = TransformOCTMaskAdjustment_AROI()

        self.image_transform = image_transform

        self.normalized = normalized
        #gray scale oct 2015: calculated with full tensor in memory {'mean': tensor([46.3758]), 'std': tensor([53.9434])}
        # calculated with batched method {'mean': tensor([46.3756]), 'std': tensor([53.9204])}
        self.normalize = TransformStandardization((0.1396), (0.0879))#torchvision.transforms.Normalize((46.3758), (53.9434)) 

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        img = cv2.imread(os.path.join(self.input_path, image_filename))
        img = color.rgb2gray(img)
        mask = cv2.imread(os.path.join(self.output_path, image_filename))
        mask = color.rgb2gray(mask)

        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()
        mask = mask.squeeze()
        

        img = torch.Tensor(img).reshape(1, 1, *img.shape)
        mask = torch.Tensor(mask).reshape(1, 1, *mask.shape)

        # adjust mask classes
        mask = self.mask_adjust(mask)
        mask = mask.int()


        # note for some reason some masks differ in size from the actual image (dims)
        # ISSUE WITH BILINEAR INTERPOLATION: If e.g. class 3 hasn't existed, but class 2 and 4, this could result in class 3 being generated by interpolation
        if self.size_transform:
           img = self.size_transform(img)
           mask = self.size_transform(mask)

        # normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.joint_transform:  
           img, mask = self.joint_transform([img, mask])
        

        if self.image_transform:
          img = self.image_transform(img)
        

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze().long()
        
        #print(f"img: {torch.unique(img)}")

        return img, mask


class DatasetAutoEncOctUnlbld(Dataset):
    """
    Map style dataset object for pretraining oct 2018 unlabeled data
    - expects .jpeg files

    Parameters:
        dataset_path: path to the dataset path/  with image files
        size_transform: deterministic transformation for resizing applied to original image  
        image_transform: transformation applied only to the permuted image (copy of original) 
    _getitem__(): returns permuted image and corresponding original image 
    """

    def __init__(self, dataset_path: str, size_transform: Callable = None, image_transform: Callable =None, normalized=True) -> None:
        self.dataset_path = dataset_path
        self.images_list = get_files(self.dataset_path, ".jpeg")
        
        # adapt size 
        self.size_transform = size_transform
        # perform augmentation on original to derive img
        self.image_transform = image_transform

        self.normalized = normalized
        # gray scale oct 2018 lbld: DME training data set
        # Normalization after size_transform 
        # {'mean': tensor([0.2500]), 'std': tensor([0.1762])}
        self.normalize = TransformStandardization((0.25), (0.1762)) 

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        img = cv2.imread(os.path.join(self.dataset_path, image_filename))
        img = color.rgb2gray(img)
        
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()
        img = torch.Tensor(img).reshape(1, 1, *img.shape)

    
        if self.size_transform:
            img = self.size_transform(img)

        #normalize after size_transform
        if self.normalized:
            img = self.normalize(img)
        
        original = img.clone()
        if self.image_transform:
            img = self.image_transform(img)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        original = original.squeeze()
        original = original.reshape(1, *original.shape)

        return img, original

class DatasetJigsawOctUnlbld(Dataset):
    """
     Map style dataset object for pretraining oct 2018 unlabeled data
    - expects .jpeg files

    Parameters:
        dataset_path: path to the dataset path/  with image files
        size_transform: deterministic transformation for resizing applied to original image 
        image_transform: transformation applied only to the image and 
                         after the joint jigsaw transformation with the mask 
        per_axis_permutation: if True TransformJigSawPuzzle is used transforming columns and rows individually
                                 otherwise TransformJigSawRandom is used and blocks are arbitrarily randomly permutated
    _getitem__(): returns jontly permuted image and block-mask   
    """

    def __init__(self, dataset_path: str, size_transform: Callable = None, joint_transform: Callable = None,\
        image_transform: Callable =None, normalized=True, blocks_per_axis=4, permutation_per_axis=False) -> None:
        self.dataset_path = dataset_path
        self.images_list = get_files(self.dataset_path, ".jpeg")

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.joint_jigsaw_transform = JointTransformJigSawPuzzle() if permutation_per_axis else JointTransformJigSawRandom()

        if joint_transform:
            self.joint_jigsaw_transform = torchvision.transforms.Compose([self.joint_jigsaw_transform,\
                *joint_transform.transforms])\
                if joint_transform.__class__ == torchvision.transforms.Compose \
                else  torchvision.transforms.Compose([self.joint_jigsaw_transform, joint_transform])

        self.image_transform = image_transform
        
        self.normalized = normalized

        # gray scale oct 2018 lbld: DME training data set
        # Normalization after size_transform 
        # {'mean': tensor([0.2500]), 'std': tensor([0.1762])}
        self.normalize = TransformStandardization((0.25), (0.1762)) 

        self.blocks_per_axis = blocks_per_axis

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        img = cv2.imread(os.path.join(self.dataset_path, image_filename))

        img = color.rgb2gray(img)
        
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()        

        img = torch.Tensor(img).reshape(1, 1, *img.shape)


        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)

        #check whether dims are evenly divisible by self.blocks_per_axis and image is quadratic
        assert img.shape[-1] % self.blocks_per_axis == 0 and img.shape[-2] % self.blocks_per_axis == 0, \
            f"image is not scramblable into {self.blocks_per_axis} blocks per axis - wrong dims {img.shape}"
        assert img.shape[-1] == img.shape[-2], f"Image input must be quadratic {img.shape[-1]} vs {img.shape[-2]}" 

        #normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        mask = torch.ones_like(img, dtype=torch.uint8)
        block_length = img.shape[-1] // self.blocks_per_axis 
        for i in range(self.blocks_per_axis):
            for j in range(self.blocks_per_axis):
                mask[...,i*block_length:(i+1)*block_length,j*block_length:(j+1)*block_length] *= \
                    (self.blocks_per_axis * i + j) 

        img, mask = self.joint_jigsaw_transform([img, mask])
        

        if self.image_transform:
            img = self.image_transform(img)
        

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze().long()

        return img, mask


class DatasetJigsawOctUnlbldAROI(Dataset):
    """
    Map style dataset object for pretraining oct 2018 unlabeled data
    - expects .png files

    Parameters:
        dataset_path: path to the dataset path/  with image files
        size_transform: deterministic transformation for resizing applied to original image 
        image_transform: transformation applied only to the image and 
                         after the joint jigsaw transformation with the mask 
        per_axis_permutation: if True TransformJigSawPuzzle is used transforming columns and rows individually
                                 otherwise TransformJigSawRandom is used and blocks are arbitrarily randomly permutated
    _getitem__(): returns jontly permuted image and block-mask   
    """

    def __init__(self, dataset_path: str, size_transform: Callable = None, joint_transform: Callable = None,\
        image_transform: Callable =None, normalized=True, blocks_per_axis=4, permutation_per_axis=True) -> None:
        self.dataset_path = dataset_path
        self.images_list = get_files(self.dataset_path, ".png")

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.joint_jigsaw_transform = JointTransformJigSawPuzzle() if permutation_per_axis else JointTransformJigSawRandom()

        if joint_transform:
            self.joint_jigsaw_transform = torchvision.transforms.Compose([self.joint_jigsaw_transform,\
                *joint_transform.transforms])\
                if joint_transform.__class__ == torchvision.transforms.Compose \
                else  torchvision.transforms.Compose([self.joint_jigsaw_transform, joint_transform])

        self.image_transform = image_transform
        
        self.normalized = normalized

        # gray scale oct 2018 lbld: DME training data set
        # Normalization after size_transform 
        # {'mean': tensor([0.2500]), 'std': tensor([0.1762])}
        self.normalize = TransformStandardization((0.1417), (0.1046)) 

        self.blocks_per_axis = blocks_per_axis

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        img = cv2.imread(os.path.join(self.dataset_path, image_filename))

        img = color.rgb2gray(img)
        
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()        

        img = torch.Tensor(img).reshape(1, 1, *img.shape)


        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)

        #check whether dims are evenly divisible by self.blocks_per_axis and image is quadratic
        assert img.shape[-1] % self.blocks_per_axis == 0 and img.shape[-2] % self.blocks_per_axis == 0, \
            f"image is not scramblable into {self.blocks_per_axis} blocks per axis - wrong dims {img.shape}"
        assert img.shape[-1] == img.shape[-2], f"Image input must be quadratic {img.shape[-1]} vs {img.shape[-2]}" 

        #normalize after size_transform
        if self.normalized:
           img = self.normalize(img)

        mask = torch.ones_like(img, dtype=torch.uint8)
        block_length = img.shape[-1] // self.blocks_per_axis 
        for i in range(self.blocks_per_axis):
            for j in range(self.blocks_per_axis):
                mask[...,i*block_length:(i+1)*block_length,j*block_length:(j+1)*block_length] *= \
                    (self.blocks_per_axis * i + j) 

        img, mask = self.joint_jigsaw_transform([img, mask])
        
        if self.image_transform:
            img = self.image_transform(img)
        

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze().long()

        return img, mask


class DatasetJigsawOctUnlbldFifteen(Dataset):
    """
     Map style dataset object for pretraining oct 2015 unlabeled data 50/61
    - expects .jpeg files

    Parameters:
        dataset_path: path to the dataset path/  with image files
        size_transform: deterministic transformation for resizing applied to original image 
        image_transform: transformation applied only to the image and 
                         after the joint jigsaw transformation with the mask 
        per_axis_permutation: if True TransformJigSawPuzzle is used transforming columns and rows individually
                                 otherwise TransformJigSawRandom is used and blocks are arbitrarily randomly permutated
    _getitem__(): returns jontly permuted image and block-mask   
    """

    def __init__(self, dataset_path: str, size_transform: Callable = None, joint_transform: Callable = None,\
        image_transform: Callable =None, normalized=True, blocks_per_axis=4, permutation_per_axis=False) -> None:
        self.dataset_path = dataset_path
        self.images_list = get_files(self.dataset_path, ".npy")

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.joint_jigsaw_transform = JointTransformJigSawPuzzle() if permutation_per_axis else JointTransformJigSawRandom()

        if joint_transform:
            self.joint_jigsaw_transform = torchvision.transforms.Compose([self.joint_jigsaw_transform,\
                *joint_transform.transforms])\
                if joint_transform.__class__ == torchvision.transforms.Compose \
                else  torchvision.transforms.Compose([self.joint_jigsaw_transform, joint_transform])

        self.image_transform = image_transform
        
        self.normalized = normalized

        #gray scale oct 2015 unlabelled: calculated with full tensor in memory
        #{'mean': tensor([48.6212]), 'std': tensor([57.9843])}
        # batched calculation{'mean': tensor([48.3849]), 'std': tensor([56.8587])}

        self.normalize = torchvision.transforms.Normalize(48.6212, 57.9843) 

        self.blocks_per_axis = blocks_per_axis

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        img = np.load(os.path.join(self.dataset_path, image_filename))

        
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()        

        img = torch.Tensor(img).reshape(1, 1, *img.shape)


        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)

        #check whether dims are evenly divisible by self.blocks_per_axis and image is quadratic
        assert img.shape[-1] % self.blocks_per_axis == 0 and img.shape[-2] % self.blocks_per_axis == 0, \
            f"image is not scramblable into {self.blocks_per_axis} blocks per axis - wrong dims {img.shape}"
        assert img.shape[-1] == img.shape[-2], f"Image input must be quadratic {img.shape[-1]} vs {img.shape[-2]}" 

        #normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        mask = torch.ones_like(img, dtype=torch.uint8)
        block_length = img.shape[-1] // self.blocks_per_axis 
        for i in range(self.blocks_per_axis):
            for j in range(self.blocks_per_axis):
                mask[...,i*block_length:(i+1)*block_length,j*block_length:(j+1)*block_length] *= \
                    (self.blocks_per_axis * i + j) 

        img, mask = self.joint_jigsaw_transform([img, mask])
        

        if self.image_transform:
            img = self.image_transform(img)
        

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze().long()

        return img, mask

class DataSetFileOnlyOctUnlbldFifteen(Dataset):
    """
    Map style dataset object for oct 2015 data unlabelled data
    - expects .npy files
    - deals with a single folder of images
    Parameters:
        dataset_path: path/
        transform: 
        size_transform
    __getitem__(): returns single image only - no mask
    """

    def __init__(self, dataset_path: str, transform: Callable = None, size_transform: Callable = None, normalized=True) -> None:
        self.dataset_path = dataset_path
        self.images_list = get_files(dataset_path, ".npy")

        

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.transform = transform

        self.normalized = normalized
        #gray scale oct 2015 unlabelled: calculated with full tensor in memory
        #{'mean': tensor([48.6212]), 'std': tensor([57.9843])}
        # batched calculation {'mean': tensor([48.3849]), 'std': tensor([56.8587])}
        self.normalize = torchvision.transforms.Normalize(48.6212, 57.9843) 

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        img = np.load(os.path.join(self.dataset_path, image_filename))
        #np.array(cv2.imread(os.path.join(self.dataset_path, image_filename))) 
        #np.array(Image.open(os.path.join(self.dataset_path, image_filename))) #np.load()
      
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()
        img = torch.Tensor(img).reshape(1, 1, *img.shape)

        if self.size_transform:
            img = self.size_transform(img)

        #normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.transform:
            img = self.transform(img)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (N,H,W)  where value at (h,w) maps to class of corresponding pixel

        return img



class DataSetFileOnlyOctUnlbld(Dataset):
    """
    Map style dataset object for oct 2018 data unlabelled data
    - expects .npy files
    - deals with a single folder of images
    Parameters:
        dataset_path: path/
        transform: 
        size_transform
    __getitem__(): returns single image only - no mask
    """

    def __init__(self, dataset_path: str, transform: Callable = None, size_transform: Callable = None, normalized=True) -> None:
        self.dataset_path = dataset_path
        self.images_list = get_files(dataset_path, ".jpeg")

        

        # their size is 128 something we can try after 128 or 128 * 2^i multiples
        #self.center_crop = torchvision.transforms.CenterCrop(128)
        
        # size transform 
        self.size_transform = size_transform

        self.transform = transform

        self.normalized = normalized
        # Normalization after size_transform  application: DME training data set 
        #   {'mean': tensor([0.2500]), 'std': tensor([0.1762])}
        self.normalize = torchvision.transforms.Normalize(0.25, 0.1762) 

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        img = cv2.imread(os.path.join(self.dataset_path, image_filename))
        img = color.rgb2gray(img)
        #np.array(cv2.imread(os.path.join(self.dataset_path, image_filename))) 
        #np.array(Image.open(os.path.join(self.dataset_path, image_filename))) #np.load()
      
        # img_size 128 works - general transforms require (N,C,H,W) dims 
        img = img.squeeze()
        img = torch.Tensor(img).reshape(1, 1, *img.shape)

        if self.size_transform:
            img = self.size_transform(img)

        #normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.transform:
            img = self.transform(img)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (N,H,W)  where value at (h,w) maps to class of corresponding pixel

        return img




