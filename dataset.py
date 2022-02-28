import os
import glob

import paddle
import numpy as np

from transforms import Compose, Compose_test

class Dataset(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=None):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root
        self.mask_root = '/home/aistudio/work/'
        self.transforms = Compose(transforms)
       

        self.input_img = glob.glob(os.path.join(self.dataset_root, "images", "*.jpg"))
        self.gt_img = glob.glob(os.path.join(self.dataset_root, "gts", "*.png"))
        self.mask_img = glob.glob(os.path.join(self.mask_root, "mask", "*.png"))

        self.input_img.sort()
        self.gt_img.sort()
        self.mask_img.sort()

        assert len(self.input_img) == len(self.gt_img) 

    def __getitem__(self, index):
        img_org = self.input_img[index]
        gt = self.gt_img[index]
        mask = self.mask_img[index]
        
        return self.transforms(img_org, gt, mask)
        

    def __len__(self):
        return len(self.input_img)

class Dataset_test(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=None):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root
        self.transforms = Compose_test(transforms)
        self.input_img = glob.glob(os.path.join(self.dataset_root, "images", "*.jpg"))

        self.input_img.sort()

 

    def __getitem__(self, index):
        input_path = self.input_img[index]  
        input, h, w= self.transforms(input_path)
        
        return input, h, w, input_path

    def __len__(self):
        return len(self.input_img)

if __name__ == '__main__':
    dataset = Dataset(dataset_root="/media/kyt/kyt_mobile/algorithm/EraseNet/dehw_train_dataset")

    