# encoding: utf-8
import sys
import os
import timeit

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from misc.functions import get_all_models, get_all_data
from misc.functions import pre_single_model, predict_csvfile


def main(argv):
    # TEST_IMAGE_LIST = "valid_image_paths.csv"
    # output_file_path = "output.csv"
    TEST_IMAGE_LIST = argv[1]
    output_file_path = argv[2]
    
    cudnn.benchmark = True
    models_dict = "all_models"

    print('loading models & data')
    
    model, pNas_model, SA_model, pNasSA_model, XcepFPN_model, xception_gcn = get_all_models()

    valid_dataloder, Nas_dataloder, Atel_dataloader, dataset_loader = get_all_data(TEST_IMAGE_LIST)
    
    model_list = {  'u2_file': [model,         valid_dataloder, '0.8990708733759138.pth'],
                    'u5_file': [pNas_model,    Nas_dataloder,   '0.8933122711473276.pth'],
                    'u6_file': [pNas_model,    Nas_dataloder,   '0.8918464376816251.pth'],
                    'u7_file': [pNas_model,    Nas_dataloder,   '0.8685262236170553.pth'],
                    'u8_file': [SA_model,      valid_dataloder, 'xcepdualexpsimat.pth'],
                    'u9_file': [SA_model,      valid_dataloder, 'xcepdualsumsimat.pth'],
                    'u10_file':[SA_model,      valid_dataloder, '0.8960423659716028.pth'],
                    'u11_file':[SA_model,      valid_dataloder, '0.8901343376820208.pth'],
                    'u12_file':[pNasSA_model,  Nas_dataloder,   '0.885028055543075.pth'],
                    'u13_file':[SA_model,      valid_dataloder, 'xcupdualesimscaleat.pth'],
                    'u14_file':[SA_model,      Atel_dataloader, 'xcupdualesimscaleat.pth'],
                    'u15_file':[model,         valid_dataloder, '0.8877043778295596.pth'],
                    'u16_file':[XcepFPN_model, valid_dataloder, '0.8743647271509183.pth'],
                    'u17_file':[XcepFPN_model, valid_dataloder, '0.8932593746041071.pth'],
                    'u18_file':[SA_model,      valid_dataloder, '0.8997462889772881.pth'],
                    'u19_file':[SA_model,      valid_dataloder, '0.8988146931955356.pth'],
                    'u20_file':[XcepFPN_model, valid_dataloder, '0.8952064212757506.pth'],
                    'u21_file':[SA_model,      dataset_loader,  'DR_cls_index_227.pth'],
                    'u22_file':[SA_model,      dataset_loader,  'DR_cls_index_113.pth'],
                    'u23_file':[xception_gcn,  dataset_loader,  'DR_cls_0.869_0.909.pth']}


    print('load models & data success!')
    print('predicting')

    y_predU_list = {}

    start = timeit.default_timer()
    for k,v in model_list.items():
        value = k.split("_")[0][1:]
        model_file = os.path.join(models_dict, v[2])
        y_predU = pre_single_model(v[0], v[1], model_file)  # (234,14)
        y_predU_list.update({f'y_predU{value}': y_predU})
    stop = timeit.default_timer()
    print('Prediction Time: ', stop - start) 


    y_predU_list['y_predU21'][:, 2] = (y_predU_list['y_predU21'][:, 2] + y_predU_list['y_predU21'][:, 1]) / 2
    y_predU_list['y_predU22'][:, 2] = (y_predU_list['y_predU22'][:, 2] + y_predU_list['y_predU22'][:, 1]) / 2 
    
    Cardiom = np.concatenate((y_predU_list['y_predU2'][:, 2, np.newaxis], y_predU_list['y_predU16'][:, 2, np.newaxis], 
                                y_predU_list['y_predU19'][:, 2, np.newaxis], y_predU_list['y_predU21'][:, 2, np.newaxis], 
                                y_predU_list['y_predU22'][:, 2, np.newaxis], y_predU_list['y_predU23'][:, 2, np.newaxis]), axis=1)

    std_car = np.std(Cardiom, axis=1)
    mean_car = np.mean(Cardiom, axis=1)
    Cardiomegaly_mean = []
    
    for i in range(Cardiom.shape[0]):
        if std_car[i] > 0.1:
            mean_car[i] = (Cardiom[i, :].sum() - Cardiom[i, :].max() - Cardiom[i, :].min()) / 4
            Cardiomegaly_mean.append(mean_car[i])
        else:
            Cardiomegaly_mean.append(mean_car[i])
    
    Edema_mean = np.concatenate((y_predU_list['y_predU17'][:, 5, np.newaxis], y_predU_list['y_predU11'][:, 5, np.newaxis],y_predU_list['y_predU10'][:, 5, np.newaxis],
                                 y_predU_list['y_predU18'][:, 5, np.newaxis], y_predU_list['y_predU20'][:, 5, np.newaxis]), axis=1)
    Edema_mean = np.mean(Edema_mean, axis=1)
    
    Consolidation_mean = np.concatenate(
        (y_predU_list['y_predU6'][:, 6, np.newaxis], y_predU_list['y_predU7'][:, 6, np.newaxis], y_predU_list['y_predU15'][:, 6, np.newaxis]), axis=1)
    Consolidation_mean = np.mean(Consolidation_mean, axis=1)
    
    Atelectasis_mean = np.concatenate((y_predU_list['y_predU8'][:, 8, np.newaxis],y_predU_list['y_predU14'][:, 8, np.newaxis],
                                      y_predU_list['y_predU9'][:, 8, np.newaxis], y_predU_list['y_predU13'][:, 8, np.newaxis]), axis=1)
    Atelectasis_mean = np.mean(Atelectasis_mean, axis=1)
    
    Pleural_Effusion_mean = np.concatenate((y_predU_list['y_predU5'][:, 10, np.newaxis],y_predU_list['y_predU12'][:, 10, np.newaxis]), axis=1)
    Pleural_Effusion_mean = np.mean(Pleural_Effusion_mean, axis=1)
    
    ensemble_prediction = np.zeros_like(y_predU_list['y_predU2'],dtype=np.float32)
    ensemble_prediction[:, 2] = np.array(Cardiomegaly_mean)
    ensemble_prediction[:, 8] = Atelectasis_mean  # y_predU3[:,8]
    ensemble_prediction[:, 5] = Edema_mean  # y_predU4[:,5]
    ensemble_prediction[:, 6] = Consolidation_mean  # y_predU4[:,6]
    ensemble_prediction[:, 10] = Pleural_Effusion_mean  # y_predU5[:,10]

    print('predict_file')
    predict_csvfile(ensemble_prediction, TEST_IMAGE_LIST, output_file_path)
    print('predict_file success!')


if __name__ == '__main__':
    main(sys.argv)