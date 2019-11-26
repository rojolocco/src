# encoding: utf-8
import sys
import os

import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from misc.util import get_dataload, get_dataset
from misc.util import get_model
from misc.util import get_pNASmodel, get_pNASSAmodel
from misc.util import get_xCeptionSAmodel, get_xCeptionFPNmodel, get_xception_gcn_model
from misc.functions import predict_positive, pre_single_model, predict_csvfile



def main(argv):
    # TEST_IMAGE_LIST = "valid_image_paths.csv"
    # output_file_path = "output.csv"
    TEST_IMAGE_LIST = argv[1]
    output_file_path = argv[2]
    
    cudnn.benchmark = True
    models_dict = "all_models"

    print('loading models & data')
    
    model = get_model()
    pNas_model = get_pNASmodel()
    SA_model = get_xCeptionSAmodel()
    pNasSA_model = get_pNASSAmodel()
    XcepFPN_model = get_xCeptionFPNmodel()
    xception_gcn = get_xception_gcn_model()
    
    valid_dataloder = get_dataload(TEST_IMAGE_LIST,size=680)
    Nas_dataloder = get_dataload(TEST_IMAGE_LIST, size=331)
    Atel_dataloader = get_dataload(TEST_IMAGE_LIST, size=800)
    dataset_loader = get_dataset(TEST_IMAGE_LIST, size=680)

    print('load models & data success!')
    print('predicting')
    
    u2_file = '0.8990708733759138.pth'
    model2_file = os.path.join(models_dict, u2_file)
    y_predU2 = pre_single_model(model, valid_dataloder, model2_file)  # (234,14)
    
    u5_file = '0.8933122711473276.pth'
    model5_file = os.path.join(models_dict, u5_file)
    y_predU5 = pre_single_model(pNas_model, Nas_dataloder, model5_file)
    
    u6_file = '0.8918464376816251.pth'
    model6_file = os.path.join(models_dict, u6_file)
    y_predU6 = pre_single_model(pNas_model, Nas_dataloder, model6_file)
    
    u7_file = '0.8685262236170553.pth'
    model7_file = os.path.join(models_dict, u7_file)
    y_predU7 = pre_single_model(pNas_model, Nas_dataloder, model7_file)

    u8_file = 'xcepdualexpsimat.pth'
    model8_file = os.path.join(models_dict, u8_file)
    y_predU8 = pre_single_model(SA_model, valid_dataloder, model8_file)
    
    u9_file = 'xcepdualsumsimat.pth'
    model9_file = os.path.join(models_dict, u9_file)
    y_predU9 = pre_single_model(SA_model, valid_dataloder, model9_file)

    u10_file = '0.8960423659716028.pth'
    model10_file = os.path.join(models_dict, u10_file)
    y_predU10 = pre_single_model(SA_model, valid_dataloder, model10_file)
    
    u11_file = '0.8901343376820208.pth'
    model11_file = os.path.join(models_dict, u11_file)
    y_predU11 = pre_single_model(SA_model, valid_dataloder, model11_file)
    
    u12_file = '0.885028055543075.pth'
    model12_file = os.path.join(models_dict, u12_file)
    y_predU12 = pre_single_model(pNasSA_model, Nas_dataloder, model12_file)

    u13_file = 'xcupdualesimscaleat.pth'
    model13_file = os.path.join(models_dict, u13_file)
    y_predU13 = pre_single_model(SA_model, valid_dataloder, model13_file)
    
    u14_file = 'xcupdualesimscaleat.pth'
    model14_file = os.path.join(models_dict, u14_file)
    y_predU14 = pre_single_model(SA_model, Atel_dataloader, model14_file)

    u15_file = '0.8877043778295596.pth'
    model15_file = os.path.join(models_dict, u15_file)
    y_predU15 = pre_single_model(model, valid_dataloder, model15_file)
   
    u16_file = '0.8743647271509183.pth'
    model16_file = os.path.join(models_dict, u16_file)
    y_predU16 = pre_single_model(XcepFPN_model, valid_dataloder, model16_file)
    
    u17_file = '0.8932593746041071.pth'
    model17_file = os.path.join(models_dict, u17_file)
    y_predU17 = pre_single_model(XcepFPN_model, valid_dataloder, model17_file)
    
    u18_file = '0.8997462889772881.pth'
    model18_file = os.path.join(models_dict, u18_file)
    y_predU18 = pre_single_model(SA_model, valid_dataloder, model18_file)

    u19_file = "0.8988146931955356.pth"
    model19_file = os.path.join(models_dict, u19_file)
    y_predU19 = pre_single_model(SA_model, valid_dataloder, model19_file)

    u20_file = "0.8952064212757506.pth"
    model20_file = os.path.join(models_dict, u20_file)
    y_predU20 = pre_single_model(XcepFPN_model, valid_dataloder, model20_file)

    u21_file = "DR_cls_index_227.pth"
    model21_file = os.path.join(models_dict, u21_file)
    y_predU21 = pre_single_model(SA_model, dataset_loader, model21_file)
    y_predU21[:, 2] = (y_predU21[:, 2] + y_predU21[:, 1]) / 2

    u22_file = "DR_cls_index_113.pth"
    model22_file = os.path.join(models_dict, u22_file)
    y_predU22 = pre_single_model(SA_model, dataset_loader, model22_file)

    u23_file = "DR_cls_0.869_0.909.pth"
    model23_file = os.path.join(models_dict, u23_file)
    y_predU23 = pre_single_model(xception_gcn, dataset_loader, model23_file)


    y_predU21[:, 2] = (y_predU21[:, 2] + y_predU21[:, 1]) / 2
    y_predU22[:, 2] = (y_predU22[:, 2] + y_predU22[:, 1]) / 2 
    
    Cardiom = np.concatenate((y_predU2[:, 2, np.newaxis], y_predU16[:, 2, np.newaxis], 
                                y_predU19[:, 2, np.newaxis], y_predU21[:, 2, np.newaxis], 
                                y_predU22[:, 2, np.newaxis], y_predU23[:, 2, np.newaxis]), axis=1)

    std_car = np.std(Cardiom, axis=1)
    mean_car = np.mean(Cardiom, axis=1)
    Cardiomegaly_mean = []
    
    for i in range(Cardiom.shape[0]):
        if std_car[i] > 0.1:
            mean_car[i] = (Cardiom[i, :].sum() - Cardiom[i, :].max() - Cardiom[i, :].min()) / 4
            Cardiomegaly_mean.append(mean_car[i])
        else:
            Cardiomegaly_mean.append(mean_car[i])
    
    Edema_mean = np.concatenate((y_predU17[:, 5, np.newaxis], y_predU11[:, 5, np.newaxis], y_predU10[:, 5, np.newaxis],
                                 y_predU18[:, 5, np.newaxis], y_predU20[:, 5, np.newaxis]), axis=1)
    Edema_mean = np.mean(Edema_mean, axis=1)
    
    Consolidation_mean = np.concatenate(
        (y_predU6[:, 6, np.newaxis], y_predU7[:, 6, np.newaxis], y_predU15[:, 6, np.newaxis]), axis=1)
    Consolidation_mean = np.mean(Consolidation_mean, axis=1)
    
    Atelectasis_mean = np.concatenate((y_predU8[:, 8, np.newaxis], y_predU14[:, 8, np.newaxis],
                                       y_predU9[:, 8, np.newaxis], y_predU13[:, 8, np.newaxis]), axis=1)
    Atelectasis_mean = np.mean(Atelectasis_mean, axis=1)
    
    Pleural_Effusion_mean = np.concatenate((y_predU5[:, 10, np.newaxis], y_predU12[:, 10, np.newaxis]), axis=1)
    Pleural_Effusion_mean = np.mean(Pleural_Effusion_mean, axis=1)
    
    ensemble_prediction = np.zeros_like(y_predU2,dtype=np.float32)
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