import sys
import math
import numpy as np
import re
# import time
# import os
# from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import  roc_curve, auc

from misc.read_data_challenge import CheXpertDataSet14, CheXpertDataSet_GCN, CheXpert
from misc.se_dense_gcn import SE_GCN_DenseNet121
from arch.models import Xception, NasNet, pNasNet, SEnet, InceptionV4
from arch.models import Xception_SA, Pnasnet_SA, Xception_FPN, Xception_GCN


def get_model():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Xception(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
   
    return model


def get_NASmodel():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NasNet(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    
    return model


def get_pNASmodel():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = pNasNet(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_pNASSAmodel():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Pnasnet_SA(14).cuda()
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
        
    return model


def get_SENETmodel():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SEnet(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_IncepV4model():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionV4(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_xCeptionSAmodel():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Xception_SA(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    
    return model


def get_xception_gcn_model():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Xception_GCN(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_xCeptionFPNmodel():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Xception_FPN(14)
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_dataload(test_list,size=448):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    test_dataset = CheXpertDataSet14(image_list_file=test_list,
                                    transform=transforms.Compose([
                                        transforms.Resize([size, size]),
                                        transforms.ToTensor(),
                                        normalize
                                    ])
                                    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def get_dataset(test_list,size=680):
    test_dataset = CheXpert(image_list_file=test_list,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ])
                                    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def compute_roc(targets, probs, label_names):
    # PATH_OUTPUT = '../output/'
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    mean_auc = []
    count = 0
    for i, label_name in enumerate(label_names):  # i th observation
        y_true = targets[:, i]
        y_score = probs[:, i]
        # drop uncertain
        iwant = y_true < 2
        y_true = y_true[iwant]
        y_score = y_score[iwant]
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
        if math.isnan(roc_auc[i]) != True:
            count += 1
            mean_auc.append(roc_auc[i])
        print(label_name, ":", roc_auc[i])
    if len(label_names)==14:
        class_mean=((mean_auc[2]+mean_auc[8]+mean_auc[6]+mean_auc[5]+mean_auc[10])/5)
    else:
        class_mean=np.mean(np.array(mean_auc))
    print("AUC mean:", np.mean(np.array(mean_auc)))
    print(len(mean_auc))
    print("5label auc:",class_mean)
    return class_mean

# def data_augmentation(image):
#     # Input should be ONE image with shape: (L, W, CH)
#     #print(image.shape)
#     options = ["gaussian_smooth", "rotate", "randomzoom", "adjust_gamma"]
#     # Probabilities for each augmentation were arbitrarily assigned
#     which_option = np.random.choice(options)
#     #print(which_option)
#     if which_option == "gaussian_smooth":
#         sigma = np.random.uniform(0.2, 1.0)
#         image = gaussian_filter(image, sigma)
#
#     elif which_option == "randomzoom":
#         # Assumes image is square
#         min_crop = int(image.shape[1]*0.85)
#         max_crop = int(image.shape[1]*0.95)
#         crop_size = np.random.randint(min_crop, max_crop)
#         crop = crop_center(image, crop_size, crop_size)
#         #crop = transforms.CenterCrop(crop_size)
#         if crop.shape[-1] == 1: crop = crop[:,:,0] # for grayscale
#         image = scipy.misc.imresize(crop, image.shape)
#
#     elif which_option == "rotate":
#         angle = np.random.uniform(-15, 15)
#         image = rotate(image, angle, reshape=False)
#
#     elif which_option == "adjust_gamma":
#         #image = image / 255.
#         image = exposure.adjust_gamma(image, np.random.uniform(0.75,1.25))
#         #image = image * 255.
#     if len(image.shape) == 2: image = np.expand_dims(image, axis=2)
#
#     return Image.fromarray(image.astype('uint8')).convert('RGB')
#
#
# def state_fusion(models):
#
#     models_num = len(models)
#
#     n0_state = models.pop(0).state_dict()
#
#     n1_state = models.pop(0).state_dict()
#     n2_state = models.pop(0).state_dict()
#     #print(len(models))
#     #exit(0)
#     for k in n0_state:
#         print('0',n0_state[k])
#         print('1',n1_state[k])
#         print('2',n2_state[k])
#     exit(0)
#     for k in empty_net_state:
#         empty_net_state[k] *= (1.0/float(models_num))
#
#     while models:
#         net = models.pop(0)
#         for k in empty_net_state:
#             print('before',empty_net_state[k])
#             empty_net_state[k] += net.state_dict()[k]*(1.0/float(models_num))
#             print('after',empty_net_state[k])
#         del net
#         time.sleep(5)
#
#     exit(0)