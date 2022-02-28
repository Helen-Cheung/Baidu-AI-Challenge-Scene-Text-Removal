import argparse
import os.path
import random
import time
import datetime
import sys

import numpy as np


import paddle

from transforms import sample_images, Resize, Normalize
from dataset import Dataset
from models.sa_gan import STRNet 
from losses import pre_network
from utils import load_pretrained_model
from loss.Loss import LossWithGAN_STE
def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
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
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=100
    )

    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='save_path',
        type=str,
        default='train_result'
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=100
    )

    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='save_interval',
        type=int,
        default=10
    )

    parser.add_argument(
        '--sample_interval',
        dest='sample_interval',
        help='sample_interval',
        type=int,
        default=100
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='load pretrained model',
        type=str,
        default=None
    )

    return parser.parse_args()


def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    #data load
    transforms = [
        Resize(target_size=(512,512)),
        Normalize()
    ]
    dataset = Dataset(dataset_root = args.dataset_root, transforms=transforms)

    dataloader = paddle.io.DataLoader(dataset, 
                                    batch_size = args.batch_size, 
                                    num_workers = 0,
                                    shuffle = True,
                                    return_list = True)
    
    # loss functions

    pre_loss = pre_network(pretrained='./vgg.pdparams')
    criterion = LossWithGAN_STE(pre_loss, lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0)
    # model

    generator = STRNet(3)
    params_info = paddle.summary(generator, (1,3,512,512))
    print(params_info)
    if args.pretrained != '':
        load_pretrained_model(generator, args.pretrained)

    # optimizer 

    # optimizer = paddle.optimizer.Adam(parameters = generator.parameters)
    G_optimizer = paddle.optimizer.Adam(learning_rate=0.0001, 
                                                    parameters = generator.parameters(), 
                                                    beta1 = 0.5,
                                                    beta2 = 0.9,
                                                    weight_decay=0.01)
    count = 1
    
    prev_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        for i, data_batch in enumerate(dataloader):
            input_img = data_batch[0]
            gt_img = data_batch[1]
            mask_img = data_batch[2]

            # model inference
            
            x_o1,x_o2,x_o3,fake_images,mm = generator(input_img)# where is generator


            # loss backward
            G_loss = criterion(input_img, mask_img, x_o1, x_o2, x_o3, fake_images, mm, gt_img, count, i)
            G_loss = G_loss.sum()
            G_optimizer.clear_grad()
            G_loss.backward()
            G_optimizer.step()
            
            # loss = a+b+c
            # loss.backward()
            # optimizer.step()
            # generator.clear_gradients()


            # determine approximate time left
            batches_done = epoch * len(dataloader) + 1
            batches_left = args.max_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if i % args.log_iters == 0:
                sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s" %
                                                (epoch, args.max_epochs,
                                                i, len(dataloader),
                                                G_loss.item(),
                                                time_left))
            if i % args.sample_interval == 0:
                sample_images(epoch, i, fake_images, gt_img, input_img, args)
        # if epoch % args.sample_interval == 0:
        if epoch % 10 == 0:
            current_save_dir = os.path.join(args.save_path,
            "model", f"epoch_{epoch}")
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(generator.state_dict(),
                                os.path.join(current_save_dir, "model.pdparams"))
            paddle.save(G_optimizer.state_dict(),
                                os.path.join(current_save_dir, "model.pdopt"))
if __name__=="__main__":
    args = parse_args()
    main(args)