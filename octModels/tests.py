import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from .transforms import TransformJigSawRandom, TransformOCTBilinear, JointTransformOCTRandomRotation, TransformJigSawPuzzle,\
    JointTransformGaussianDistortion, JointTransformRandomDistortion
from .datasets import DatasetOct
from .utils import beautify_tensorboard_oct



def test_normalization(path="/mnt/ben/Desktop/mlmi/code/data/2015_BOE_Chiu/2015_BOE_Chiu_sliced_bscans/train"):
    size_transform = TransformOCTBilinear((128,128))
    joint_transform = JointTransformOCTRandomRotation()
    train_dataset = DatasetOct(path,\
        size_transform=size_transform, joint_transform=joint_transform, normalized=True)
    dataloader = DataLoader(train_dataset, batch_size=10)
    for data, *_ in dataloader:
        for s in ["standardized image", "rescaled image"]:
            if s == "rescaled image":
                data = beautify_tensorboard_oct(data)
            plt.imshow(data[0].squeeze())
            plt.colorbar()
            plt.title(s)
            plt.show()


def test_jigsaw(blocks_per_axis):
    tf = TransformJigSawPuzzle()
    x = create_mask(16,16, blocks_per_axis).reshape(-1, 16, 16)
    y = tf(x)
    return x,y 

def test_jigsaw_random(blocks_per_axis):
    tf = TransformJigSawRandom()
    x = create_mask(16,16, blocks_per_axis).reshape(-1, 16, 16)
    y = tf(x)
    return x,y 
def test_gaussian_distortion(blocks_per_axis, **kwargs):
    tf = JointTransformGaussianDistortion(**kwargs)
    h,w = blocks_per_axis * 4, blocks_per_axis * 4
    x = create_mask(h, w, blocks_per_axis).reshape(-1, h, w)
    y = x.clone() 
    print(x.size,y.size)
    x_t, y_t = tf(x, y)

    f,ax = plt.subplots(1,2)
    mesh_0 = ax[0].pcolormesh(x.squeeze())
    f.colorbar(mesh_0, ax=ax[0])
    #ax[0].imshow(x.squeeze())

    mesh_1 = ax[1].pcolormesh(x_t.squeeze())
    f.colorbar(mesh_1, ax=ax[1])
    #ax[1].imshow(x_t.squeeze())

    #for h in range(x.shape[0]):
        #for w in range(x.shape[1]):
            # set_text: x,y,s -> height = y, width=x
            #ax[1].text(w,h,round(x_t.squeeze()[h,w].item(),2))
    plt.show()
    return x,x_t

def test_random_distortion(blocks_per_axis, **kwargs):
    tf = JointTransformRandomDistortion(**kwargs)
    h,w = blocks_per_axis * 4, blocks_per_axis * 4
    x = create_mask(h, w, blocks_per_axis).reshape(-1, h, w)
    y = x.clone() 
    print(x.size,y.size)
    x_t, y_t = tf(x, y)

    f,ax = plt.subplots(1,2)
    mesh_0 = ax[0].pcolormesh(x.squeeze())
    f.colorbar(mesh_0, ax=ax[0])
    #ax[0].imshow(x.squeeze())

    mesh_1 = ax[1].pcolormesh(x_t.squeeze())
    f.colorbar(mesh_1, ax=ax[1])
    #ax[1].imshow(x_t.squeeze())

    #for h in range(x.shape[0]):
        #for w in range(x.shape[1]):
            # set_text: x,y,s -> height = y, width=x
            #ax[1].text(w,h,round(x_t.squeeze()[h,w].item(),2))
    plt.show()
    return x, x_t



def create_mask(height, width, blocks_per_axis):
    mask = torch.ones(height, width, dtype=torch.uint8)
    block_length = width // blocks_per_axis 
    for i in range(blocks_per_axis):
        for j in range(blocks_per_axis):
            #print(f"i:{i}, j:{j}, block_length:{block_length}")
            mask[...,i*block_length:(i+1)*block_length,j*block_length:(j+1)*block_length] *= \
                (blocks_per_axis * i + j)
    return mask

def plot_permutations(x):
    x = x.squeeze()
    plt.imshow(x)
    plt.colorbar()
    for h in range(x.shape[0]):
        for w in range(x.shape[1]):
            #plt.text: x,y,s -> height = y, width = x
            plt.text(w,h,round(x[h,w].item(),2))

    plt.show()


def test_jigsaw_ds(permutation_per_axis=True):
    from .datasets import DatasetJigsawOctUnlbld
    from .transforms import TransformOCTBilinear, TransformGaussianWhiteNoise
    import torchvision
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    size_transform = TransformOCTBilinear((128,128))
    image_transform = torchvision.transforms.Compose([TransformGaussianWhiteNoise()]) 
    train_dataset_path = "/mnt/ben/Desktop/mlmi/code/data/2018_oct_data/OCT2017/train/DME"
    train_dataset = DatasetJigsawOctUnlbld(train_dataset_path, size_transform=size_transform,\
        image_transform=image_transform, normalized=True, permutation_per_axis=permutation_per_axis)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for x,l in dataloader:
        f,ax = plt.subplots(1,2)
        ax[0].imshow(x.squeeze())
        ax[1].imshow(l.squeeze())
        for h in range(x.shape[0]):
            for w in range(x.shape[1]):
                # set_text: x,y,s -> height = y, width=x
                ax[1].text(w,h,l.squeeze()[h,w].item())
        plt.show()



