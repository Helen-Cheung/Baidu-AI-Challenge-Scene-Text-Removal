import random 
import numpy as np
import cv2
import paddle
from PIL import Image
import os
class Compose:
    def __init__(self, transforms=None, to_rgb= True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.to_rgb = to_rgb
        self.transforms = transforms
    def __call__(self, input, gt, mask):

        if isinstance(input, str):
            input = cv2.imread(input).astype('float32')
 
        if isinstance(gt, str):
            gt = cv2.imread(gt).astype('float32')

        if isinstance(mask, str):
            mask = cv2.imread(mask).astype('float32')

        if input is None or gt is None or mask is None:
            raise ValueError("Can't read The image file")
        if self.to_rgb:
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            for op in self.transforms:
                outputs = op(input, gt, mask)
                input = outputs[0]
                gt = outputs[1]
                mask = outputs[2]
        else:
            pass
        input = np.transpose(input, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        # print(input.shape)
        return input, gt, mask


class Compose_test:
    def __init__(self, transforms=None, to_rgb= True):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.to_rgb = to_rgb
        self.transforms = transforms
    def __call__(self, input):

        if isinstance(input, str):
            input = cv2.imread(input).astype('float32')
            h,w = input.shape[0], input.shape[1]
        if input is None:
            raise ValueError("Can't read The image file")
        if self.to_rgb:
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            for op in self.transforms:
                outputs = op(input,input,input)
                input = outputs[0]
        else:
            pass
        input = np.transpose(input, (2, 0, 1))

        return input, h, w

def sample_images(epoch, i, output, gt, input, args):
    output, gt, input = output*255, gt*255, input*255
    output = paddle.clip(output.detach(), 0, 255)

    output = output.cast('int64')
    gt = gt.cast('int64')
    input = input.cast('int64')
    h,w = output.shape[-2], output.shape[-1]
    img = np.zeros((h, 3*w, 3))
    for idx in range(0,1):
        row = idx * h
        tmplist = [input[idx], gt[idx], output[idx]]
        for k in range(3):
            col = k * w 
            tmp = np.transpose(tmplist[k], (1, 2, 0))
            img[row:row + h, col:col + w] = np.array(tmp)
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    img.save(os.path.join(args.save_path, '%03d_%06d.png'%(epoch,i)))


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im



class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = horizontal_flip(im1)
            im2 = horizontal_flip(im2)
        return im1, im2


def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


class Normalize:

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im1, im2, im3):

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im1 = normalize(im1, mean, std)
        im2 = normalize(im2, mean, std)
        im3 = normalize(im3, mean, std)

        return im1, im2, im3


class Resize:

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im1, im2, im3):

        if not isinstance(im1, np.ndarray) or not (im2, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im1.shape) != 3 or len(im2.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im1 = resize(im1, self.target_size, self.interp_dict[interp])

        im2 = resize(im2, self.target_size, self.interp_dict[interp])

        im3 = resize(im3, self.target_size, self.interp_dict[interp])
        return im1, im2, im3
