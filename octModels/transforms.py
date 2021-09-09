import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import Augmentor 


class CropToSquareFromTop(object):
    def __init__(self, output_size, input_size):
        assert isinstance(output_size[0],int) and isinstance(output_size[1],int), "output size must be a tuple of ints"
        assert isinstance(input_size[0],int) and isinstance(input_size[1],int), "input size must be a tuple of ints"
        self.output_size = output_size
        left_offset = (input_size[1] - self.output_size[1]) // 2
        #(... x H x W) 
        self.crop = torchvision.transforms.Lambda(lambda x: x[...,0:self.output_size[1],left_offset:left_offset + self.output_size[1]])
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape 

        print("image.shape: ", image.shape)
        print("label.shape: ", label.shape)
        
        image = image.reshape(1, *image.shape)
        label = label.reshape(1, *label.shape)

        print("image.shape: ", image.shape)
        print("label.shape: ", label.shape)

        image = self.crop(image)
        label = self.crop(label)

        print("postcrop image.shape: ", image.shape)
        print("postcrop label.shape: ", label.shape)
        sample = {'image': image, 'label': label.long()}
        return sample



class CropToSquareFromTopOCT(CropToSquareFromTop):
    def __init__(self, output_size):
        input_size=(480,480)# 768 x (510 +-20) - to be on the sage side width wise-
                            # used to compute offset for width from the left border
        super(CropToSquareFromTopOCT, self).__init__(output_size, input_size)

class TransformOCTCenterTopCropBilinear(object):
    def __new__(cls,*args, **kwargs):
        return torchvision.transforms.Compose([
            CropToSquareFromTopOCT(384,384),
            torchvision.transforms.Resize((128,128), interpolation=F.InterpolationMode.BILINEAR)
            ])
class TransformOCTBilinear(object):
    def __new__(cls, img_size=(128,128),*args, **kwargs):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size, interpolation=F.InterpolationMode.BILINEAR)
            ])

class TransformOCTRandomRotation(object):
    def __new__(cls, *args, **kwargs):
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation((0,360), interpolation=F.InterpolationMode.BILINEAR, fill=0)
            ])

class OldJointTransformOCTRandomRotation(object):
    """
    Jointly apply Random Rotation to an image and a mask filling the image with max(image) and mask with 0 the NOT class
    degrees: (min,max) - df:(0,360)
    """
    def __init__(self, degrees:tuple=(0,360)):
        self.degrees = degrees
    def __call__(self, image, mask):
        degrees = torchvision.transforms.RandomRotation.get_params(self.degrees)
        return F.rotate(image, degrees, F.InterpolationMode.BILINEAR, fill=torch.max(image).item()), \
            F.rotate(mask, degrees, F.InterpolationMode.BILINEAR, fill=0) 
    def __repr__(self):
        return self.__class__.__name__ + f": degrees {self.degrees}"


class JointTransformGaussianDistortion(object):
    """
    Wrapper Class for https://augmentor.readthedocs.io/en/master/code.html?highlight=GaussianDistortion#Augmentor.Operations.GaussianDistortion
    Jointly apply Gaussian Distortion to an image and a mask 
    parameters:
        - probability : probability of applying the transformation
        - grid_width: width of the grid used in gaussian distortion
        - grid_height: height of the grid used in gaussian distortion
        - magnitude: controls strength of distortion
        - corner: corner of picture to distort
    """
    def __init__(self, probability=1., grid_width=8, grid_height=8, magnitude=8, corner="bell", method="in"):
        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude
        self.corner = corner
        self.method = method
        # mex,stdx... - parameters for scaling surface - computed from x,y: exp(- ((x-mex)**2/sdx+(y-mey)**2/sdy))
        #               based on a normal distribution
        self.transform = Augmentor.Operations.GaussianDistortion(probability=probability, grid_width=grid_width,
        grid_height=grid_height, magnitude=magnitude, corner=corner, method=method, mex=0, sdx=1, mey=0, sdy =1)

        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
    def __call__(self, images):
        image, mask = images
        print(image.shape)
        print(f"object:{[image, mask]}, access {[image, mask][0]}")
        image, mask = self.transform.perform_operation([self.to_pil(image), self.to_pil(mask)])
        return self.to_tensor(image), self.to_tensor(mask)
    def __repr__(self):
        return self.__class__.__name__ + f": probability {self.probability}, grid_width {self.grid_width}, grid_height {self.grid_height}, "\
            + f"magnitude {self.magnitude}, corner {self.corner}, method {self.method}"


class JointTransformRandomDistortion(object):
    """
    Wrapper Class for https://augmentor.readthedocs.io/en/master/code.html?highlight=GaussianDistortion#Augmentor.Operations.Distort
    Jointly apply Random Distortion to an image and a mask 
    parameters:
        - probability : probability of applying the transformation
        - grid_width: width of the grid used in random distortion
        - grid_height: height of the grid used in random distortion
        - magnitude: controls strength of distortion
    """
    def __init__(self, probability=1., grid_width=8, grid_height=8, magnitude=8):
        self.probability = probability
        self.grid_width = grid_width
        self.magnitude = magnitude

        # mex,stdx... - parameters for scaling surface - computed from x,y: exp(- ((x-mex)**2/sdx+(y-mey)**2/sdy))
        #               based on a normal distribution
        self.transform = Augmentor.Operations.Distort(probability=probability, grid_width=grid_width,
        grid_height=grid_height, magnitude=magnitude)

        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_pil = torchvision.transforms.ToPILImage()
    def __call__(self, image, mask):
        print(f"object:{[image, mask]}, access {[image, mask][0]}")
        image, mask = self.transform.perform_operation([self.to_pil(image), self.to_pil(mask)])
        return self.to_tensor(image), self.to_tensor(mask)
    def __repr__(self):
        return self.__class__.__name__ + f": probability {self.probability}, grid_width {self.grid_width}, grid_height {self.grid_height}, "\
            + f"magnitude {self.magnitude}"


class TransformJigSawPuzzle(object):
    """
    Random Blockwise permutation of the image only  
    requiring model to provide unscrambled mask

    limited by separation into permutations of blocked rows and bloked columns 
    (ie same col/row permutations applied to all blocked rows/ columns)
    """
    rng = np.random.default_rng()
    def __init__(self, blocks_per_axis=4):
        self.blocks_per_axis = blocks_per_axis
    def __call__(self, image):
        
        row_perm_idx,  col_perm_idx = TransformJigSawPuzzle.get_params(self.blocks_per_axis)
        image = TransformJigSawPuzzle.permutate(image, row_perm_idx, col_perm_idx, self.blocks_per_axis)
        
        return image  

    def __repr__(self):
        return self.__class__.__name__ + f": blocks_per_axis {self.blocks_per_axis}"
    
    @classmethod
    def get_params(cls, blocks_per_axis):
        """
        generate permutations i,p in enumerate(perm_idx) - gives actual permutation: i->p
        returns row_permutation_idx, col_permutation_idx
        """
        return cls.rng.permutation(blocks_per_axis), cls.rng.permutation(blocks_per_axis)
    
    @staticmethod
    def permutate(image, row_permutation_idx, col_permutation_idx, blocks_per_axis):
        assert image.shape[-1] % blocks_per_axis == 0 and image.shape[-2] % blocks_per_axis == 0, \
            f"image is not scramblable into {blocks_per_axis} blocks per axis - wrong dims {image.shape}"
        assert image.shape[-1] == image.shape[-2], f"Image input must be quadratic {image.shape[-1]} vs {image.shape[-2]}" 
        
        block_length = image.shape[-1] // blocks_per_axis 
        
        row_perm = torch.zeros_like(image)
        # entry of identity at block (i,j) ~ swap column i and j when image @ col_perm
        for row,col in enumerate(row_permutation_idx):
            row_perm[...,row*block_length:(row+1)*block_length,col*block_length:(col+1)*block_length] = \
                torch.eye(block_length)
        col_perm = torch.zeros_like(image)
        # entry of identity at block (i,j) ~ swap row i and j when row_perm @ image
        for row,col in enumerate(col_permutation_idx):
            col_perm[...,row*block_length:(row+1)*block_length,col*block_length:(col+1)*block_length] = \
                torch.eye(block_length)
        #note that the difference in permutation arises only from the order of operands in multiplication (rows or columns)
        image = row_perm @ image
        image = image @ col_perm
        
        return image




class JointTransformJigSawPuzzle(object):
    """
    Random Blockwise permutation of N images jointly 

    limited by separation into permutations of blocked rows and bloked columns 
    (ie same col/row permutations applied to all blocked rows/ columns)
    """
    def __init__(self, blocks_per_axis=4):
        self.blocks_per_axis = blocks_per_axis
    def __call__(self, images):
        permutations = []

        row_perm_idx,  col_perm_idx = TransformJigSawPuzzle.get_params(self.blocks_per_axis)
        
        for img in images:
            perm_img = TransformJigSawPuzzle.permutate(img, row_perm_idx, col_perm_idx, self.blocks_per_axis)
            permutations.append(perm_img)
        
        return permutations

    def __repr__(self):
        return self.__class__.__name__ + f": blocks_per_axis {self.blocks_per_axis}"




class TransformJigSawRandom(object):
    """
    Random Blockwise permutation of the image only  
    requiring model to provide unscrambled mask

    Random permutation of blocks.
    """
    rng = np.random.default_rng()
    def __init__(self, blocks_per_axis=4):
        self.blocks_per_axis = blocks_per_axis
    def __call__(self, image):
        
        permutation_idx = TransformJigSawRandom.get_params(self.blocks_per_axis)
        image = TransformJigSawRandom.permutate(image, permutation_idx, self.blocks_per_axis)
        
        return image  

    def __repr__(self):
        return self.__class__.__name__ + f": blocks_per_axis {self.blocks_per_axis}"
    
    @classmethod
    def get_params(cls, blocks_per_axis):
        """
        generate permutations i,p in enumerate(perm_idx) - gives actual permutation: i->p
        args:
            blocks_per_axis: number of blocks per axis
        returns:
            permutation_idx: index sequence of permutations of the blocks, with i e [0, blocks_per_axis ** 2 - 1]
        """
        return cls.rng.permutation(blocks_per_axis ** 2)
    
    @staticmethod
    def permutate(image, permutation_idx, blocks_per_axis):
        assert image.shape[-1] % blocks_per_axis == 0 and image.shape[-2] % blocks_per_axis == 0, \
            f"image is not scramblable into {blocks_per_axis} blocks per axis - wrong dims {image.shape}"
        assert image.shape[-1] == image.shape[-2], f"Image input must be quadratic {image.shape[-1]} vs {image.shape[-2]}" 
        
        block_length = image.shape[-1] // blocks_per_axis 
        
        permutation = torch.zeros_like(image)
        # entry of identity at block (i,j) ~ swap column i and j when image @ col_perm
        for pos,prm in enumerate(permutation_idx):
            row_prm, col_prm = prm // blocks_per_axis, prm % blocks_per_axis
            row, col = pos // blocks_per_axis, pos % blocks_per_axis
            #print(f"perm row/col: {row_prm, col_prm}, original row/col:{row, col}")
            permutation[...,row_prm*block_length:(row_prm+1)*block_length,col_prm*block_length:(col_prm+1)*block_length] = \
                image[...,row*block_length:(row+1)*block_length,col*block_length:(col+1)*block_length]
        
        return permutation


class JointTransformJigSawRandom(object):
    """
    Random Blockwise permutation of N images jointly 

    Blocks are permutated randomly 
    """
    def __init__(self, blocks_per_axis=4):
        self.blocks_per_axis = blocks_per_axis
    def __call__(self, images):
        permutations = []

        permutation_idx = TransformJigSawRandom.get_params(self.blocks_per_axis)
        
        for img in images:
            perm_img = TransformJigSawRandom.permutate(img, permutation_idx, self.blocks_per_axis)
            permutations.append(perm_img)
        
        return permutations

    def __repr__(self):
        return self.__class__.__name__ + f": blocks_per_axis {self.blocks_per_axis}"



class TransformGaussianWhiteNoise(object):
    """
    Gaussian White Noise
    """
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        return image +  torch.randn(image.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + f"mean {self.mean}, std {self.std}"

class TransformOctGaussianWhiteNoise(TransformGaussianWhiteNoise):
    def __init__(self, *args, **kwargs):
        # set std yet enable deliberate overwriting
        super(TransformOctGaussianWhiteNoise, self).__init__(*args, std=53.9434, **kwargs)

class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        return (image - self.mean) / self.std
    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"

class TransformOCTMaskAdjustment(object):
    """
    Adjust OCT 2015 Mask 
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid 
    """
    def __call__(self, mask):
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        return mask

class TransformOCTMaskAdjustment_AROI(object):
    """
    Adjust OCT AROI Mask 
    from: greyscale values [0.     0.0721 0.2125 0.2846 0.7154 0.7875 0.9279 1.    ]
    to: classes [0,1,2,3,4,5,6,7]
    """
    def __call__(self, mask):
        mask[mask == 0] = 0
        mask[mask == 1] = 7
        mask[mask == 0.0721] = 1
        mask[mask == 0.2125] = 2
        mask[mask == 0.2846] = 3
        mask[mask == 0.7154] = 4
        mask[mask == 0.7875] = 5
        mask[mask == 0.9279] = 6
        return mask

class JointTransformOCTRandomRotation(object):
    """
    Jointly apply Random Rotation to arbitrarily many images and or labels
    degrees: (min,max) - df:(0,360)
    """
    def __init__(self, degrees:tuple=(0,360)):
        self.degrees = degrees
    def __call__(self, images):
        degrees = torchvision.transforms.RandomRotation.get_params(self.degrees)
        return [F.rotate(img, degrees, F.InterpolationMode.BILINEAR) for img in images]
