import argparse
import glob
import os.path
from dataset import Dataset_test
from transforms import Normalize, Resize
import paddle
import paddle.nn as nn
import cv2
import numpy as np
from models.pert import PERT
import time
from utils import load_pretrained_model
from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser(description='Model testing')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default=None)

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=1
    )

    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='save_path',
        type=str,
        default='test_result'
    )

    return parser.parse_args()

def main(args):
    model = PERT(3)
    mask_path = 'mask_result'
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)


    transforms = [
        Resize(target_size=(512, 512)),
        Normalize()
    ]
    dataset = Dataset_test(dataset_root=args.dataset_root, transforms=transforms)
    dataloader = paddle.io.DataLoader(dataset, 
                                    batch_size = args.batch_size, 
                                    num_workers = 0,
                                    shuffle = True,
                                    return_list = True)
    model.eval()
    for i, (img, h, w, path) in enumerate(dataloader):
        
        # inference
        start = time.time()
        #out1, out2, out3, img_out,mm = model(img)
        img_out, mask_out, p1_out, p2_out= model(img, img)# where is generator
        for t in range(2):
            img_out, mask_out, p1_out, p2_out= model(img, img_out)
        end = time.time()
        time_one = end - start
        print('The running time of an image is : {:2f} s'.format(time_one))

        img_out = nn.functional.interpolate(img_out, size = [h,w], mode = 'bilinear')
        img_out = img_out.squeeze(0)

        img_out = paddle.clip(img_out* 255.0, 0, 255)
        img_out = paddle.transpose(img_out, [1,2,0])
        img_out = img_out.numpy()

        mask_out = nn.functional.interpolate(mask_out, size = [h,w], mode = 'bilinear')
        mask_out = mask_out.squeeze(0)

        mask_out = paddle.clip(mask_out* 255.0, 0, 255)
        mask_out = paddle.transpose(mask_out, [1,2,0])
        mask_out = mask_out.numpy()

        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        Image.fromarray(np.uint8(img_out)).convert("RGB").resize((w,h),Image.BILINEAR).save(os.path.join(save_path, path[0].split('/')[-1].replace('jpg','png')))
        Image.fromarray(np.uint8(mask_out)).convert("RGB").resize((w,h),Image.NEAREST).save(os.path.join(mask_path, path[0].split('/')[-1].replace('jpg','png')))



if __name__=='__main__':
    args = parse_args()
    main(args)