import paddle 
import paddle.nn as nn
import numpy as np
from paddle.vision.transforms import resize
#from skimage.metrics import structural_similarity as compare_ssim
from typing import ClassVar

import vgg

class pre_network(nn.Layer):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """
    def __init__(self, pretrained: str = None):
        super(pre_network, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=pretrained).features
        self.layer_name_mapping = {
            '3':'relu1',
            '8':'relu2',
            '13':'relu3',
            # '22':'relu4',
            # '31':'relu5',
        }
    def forward(self, x):
        output = {}
        
        for name, module in self.vgg_layers._sub_layers.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output

def RS_loss(p1_out, p2_out, img_out, mask_gt, img_gt, alpha, alpha1, alpha2, beta, beta1, beta2):

    resize_128 = nn.Upsample(size=[128,128],mode='bilinear')
    resize_256 = nn.Upsample(size=[256,256],mode='bilinear')
    mask1_gt = resize_128(mask_gt)
    mask2_gt = resize_256(mask_gt)

    img1_gt = resize_128(img_gt)
    img2_gt = resize_256(img_gt)

    loss = alpha1*paddle.norm((p1_out-img1_gt)*mask1_gt, p=1) + alpha2*paddle.norm((p2_out-img2_gt)*mask2_gt, p=1) + beta1*paddle.norm((p1_out-img1_gt)*(1-mask1_gt), p=1) + beta2*paddle.norm((p2_out-img2_gt)*(1-mask2_gt), p=1) + alpha*paddle.norm((img_out-img_gt)*mask_gt, p=1) + beta*paddle.norm((img_out-img_gt)*(1-mask_gt), p=1)

    return loss


# def ssim_loss(img_out, img_gt):

#     loss = compare_ssim(img_out, img_gt)

#     return loss

def compute_l1_loss(input, output):
    return paddle.mean(paddle.abs(input - output))

def preceptual_loss(lossnet, fake_B, real_B, tensor_c):
    loss_fake_B = lossnet(fake_B * 255 - tensor_c)
    loss_real_B = lossnet(real_B * 255 - tensor_c)
    p1 = compute_l1_loss(loss_fake_B['relu1'], loss_real_B['relu1']) / 2.6
    p2 = compute_l1_loss(loss_fake_B['relu2'], loss_real_B['relu2']) / 4.8
    p3=  compute_l1_loss(loss_fake_B['relu3'],loss_real_B['relu3'])/3.7
    # p4=compute_l1_loss(loss_fake_B['relu4'],loss_real_B['relu4'])/5.6
    # p5=compute_l1_loss(loss_fake_B['relu5'],loss_real_B['relu5'])/5.6
    loss_p = p1 + p2 + p3
    return loss_p