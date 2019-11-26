# encoding: utf-8

"""
Read images and corresponding labels.
"""
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


def load_and_resize_img(path):
    """
    Load and convert the full resolution images on CodaLab to
    low resolution used in the small dataset.
    """
    img = cv2.imread(path, 0)

    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    if max_ind == 1:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size = (hsize, 320)

    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size = (320, wsize)

    resized_img = cv2.resize(img, new_size)

    return resized_img


class CheXpertDataSet_GCN(Dataset):
    def __init__(self, image_list_file, transform=None):
        df = pd.read_csv(image_list_file)
        self.transform = transform
        self.imagePaths = []
        for i, row in df.iterrows():
            # self.imagePaths.append(os.path.join(data_dir,row['Path']))
            self.imagePaths.append(row[0])
        self.inp = np.identity(14)#one-hot

    def __getitem__(self, index):
        image = Image.open(self.imagePaths[index]).convert('RGB') # pre-trained models on ImageNet are 'RGB'

        if self.transform is not None:
            image = self.transform(image)
        #return image, torch.LongTensor(self.labels[index]) #for 3-output, softmax, CE
        return image,self.inp #for 1-output, sigmoid, BCE

    def __len__(self):
        return len(self.imagePaths)


class CheXpertDataSet14(Dataset):
    def __init__(self,image_list_file, transform=None):
        df = pd.read_csv(image_list_file)
        self.transform = transform
        self.imagePaths = []
        for i, row in df.iterrows():
            # self.imagePaths.append(os.path.join(data_dir,row['Path']))
            self.imagePaths.append(row[0])

    def __getitem__(self, index):
        image = Image.open(self.imagePaths[index]).convert('RGB') # pre-trained models on ImageNet are 'RGB'\
        # image = cv2.imread(self.imagePaths[index])
        # image = data_augmentation(image)
        if self.transform is not None:
            image = self.transform(image)
        #return image, torch.LongTensor(self.labels[index]) #for 3-output, softmax, CE
        return image #for 1-output, sigmoid, BCE

    def __len__(self):
        return len(self.imagePaths)


# from albumentations import Resize
class CheXpert(Dataset): #14类加载
    def __init__(self, image_list_file, transform=None):
        df = pd.read_csv(image_list_file)
        self.transform = transform
        self.imagePaths = []
        for i, row in df.iterrows():
            # self.imagePaths.append(os.path.join(data_dir,row['Path']))
            self.imagePaths.append(row[0])

    
    def __getitem__(self, index):
        self.image_size = 680
        # print(self.imagePaths[index])
        image = cv2.imread(self.imagePaths[index])
        y, x, z = image.shape
        if x > y:
            resized_x, resized_y = self.image_size, int(y*self.image_size/x)
        else:
            resized_x, resized_y = int(x*self.image_size/y), self.image_size
        # resize = Resize(resized_y, resized_x)
        # image = resize(image=image)["image"]
        image = cv2.resize(image, (resized_x, resized_y), interpolation=cv2.INTER_LINEAR)
        image = np.pad(image, [[0, self.image_size - resized_y], [0, self.image_size - resized_x], [0, 0]], 'constant', constant_values=0)
        image =self.transform(image)
        image = (image - 0.5) / 0.5

        return image #for 1-output, sigmoid, BCE
    def __len__(self):
        return len(self.imagePaths)


# from scipy.ndimage.interpolation import zoom, rotate
# from scipy.ndimage.filters import gaussian_filter
# import scipy
# from skimage import exposure

# def data_augmentation(image):
#     # Input should be ONE image with shape: (L, W, CH)
#     # print(image.shape)
#     options = ["gaussian_smooth", "rotate", "adjust_gamma","no_aug"]
#     # Probabilities for each augmentation were arbitrarily assigned
#     which_option = np.random.choice(options)
#     # print(which_option)
#     if which_option == "gaussian_smooth":
#         sigma = np.random.uniform(0.2, 1.0)
#         image = gaussian_filter(image, sigma)

#     # elif which_option == "randomzoom":
#     #     # Assumes image is square
#     #     min_crop = int(image.shape[1] * 0.85)
#     #     max_crop = int(image.shape[1] * 0.95)
#     #     crop_size = np.random.randint(min_crop, max_crop)
#     #     crop = crop_center(image, crop_size, crop_size)
#     #     # crop = transforms.CenterCrop(crop_size)
#     #     if crop.shape[-1] == 1: crop = crop[:, :, 0]  # for grayscale
#     #     image = scipy.misc.imresize(crop, image.shape)
#     elif which_option == "no_aug":
#         pass

#     elif which_option == "rotate":
#         angle = np.random.uniform(-30, 30)
#         image = rotate(image, angle, reshape=False)

#     elif which_option == "adjust_gamma":
#         # image = image / 255.
#         image = exposure.adjust_gamma(image, np.random.uniform(0.75, 1.25))
#         # image = image * 255.
#     if len(image.shape) == 2: image = np.expand_dims(image, axis=2)
#     # print(image.shape)
#     return Image.fromarray(image.astype('uint8')).convert('RGB')


