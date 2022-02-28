import cv2
import numpy as np
import glob
import os


# src_image为RGB三通道源图片，gt_image为RGB三通道GT图片
threshold = 25
dataset_root = '/home/aistudio/data/dehw_train_dataset/'
input_img = glob.glob(os.path.join(dataset_root, "images", "*.jpg"))
gt_img = glob.glob(os.path.join(dataset_root, "gts", "*.png"))

input_img.sort()
gt_img.sort()
assert len(input_img) == len(gt_img) 

for i in range(len(input_img)):
    src_image = cv2.imread(input_img[i])
    gt_image = cv2.imread(gt_img[i])
    diff_image = np.abs(src_image.astype(np.float32) - gt_image.astype(np.float32))
    mean_image = np.mean(diff_image, axis=-1)
    mask = np.greater(mean_image, threshold).astype(np.uint8)
    mask = mask*255
    mask = np.array([mask for i in range(3)]).transpose(1,2,0)
    # mask_test = mask/255
    # new_img = mask_test*gt_image + (1-mask_test)*src_image
    # print(mask.shape)
    cv2.imwrite("/home/aistudio/work/mask/dehw_train_%06d.png" % i, mask)
    # cv2.imwrite("/home/aistudio/work/test/dehw_train_%06d.png" % i, new_img)
    print(i)



