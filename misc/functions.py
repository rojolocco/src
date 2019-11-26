import os 

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from misc.util import get_dataload, get_dataset
from misc.util import get_model
from misc.util import get_pNASmodel, get_pNASSAmodel
from misc.util import get_xCeptionSAmodel, get_xCeptionFPNmodel, get_xception_gcn_model
from misc.functions import predict_positive, pre_single_model, predict_csvfile

def predict_positive(model, device, data_loader):
    model.eval()
    # return a List of probabilities
    # input, target = zip(*data_loader)
    probas = np.array([])
    with torch.no_grad():
        for i, input in enumerate(data_loader):
            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            output = model(input)  # num_batch x 14 x 3
            sigmoid = nn.Sigmoid()
            output = sigmoid(output)
            y_pred = output.detach().to('cpu').numpy()
            # y_pred = y_pred[:,:,:2] # drop uncertain
            # y_pred = softmax(y_pred, axis = -1)
            # y_pred = y_pred[:,:,1] # keep positive only
            probas = np.concatenate((probas, y_pred), axis=0) if len(probas) > 0 else y_pred

    return probas

def pre_single_model(model,test_loader,model_file):
    best_model = torch.load(model_file)
    model.load_state_dict(best_model)
    device = torch.device("cuda")
    probas=predict_positive(model,device,test_loader)
    return probas

def ensemble_models(models_dict,model,test_loader):
    ensemble_prediction=np.array([])
    for model_file in os.listdir(models_dict):
        model_file = os.path.join(models_dict,model_file)
        y_pred = pre_single_model(model,test_loader,model_file)
        # targets = targets[...,np.newaxis]
        y_pred = y_pred[..., np.newaxis]
        ensemble_prediction = np.concatenate((ensemble_prediction, y_pred), axis=2) if len(ensemble_prediction) > 0 else y_pred
    ensemble_score = np.mean(ensemble_prediction,axis=2)
    return ensemble_score

def predict_csvfile(prediction_np, input_file, output_file):
    df_test = pd.read_csv(input_file)
    paths = df_test['Path'].copy().values
    ids = df_test['Path'].copy().values
    for i, id in enumerate(ids):
        ids[i] = id[:-17]  # include patient and study

    test_probs_studies, studies_list = [], []
    i = 0
    while i < len(ids):
        j = i + 1
        while (j < len(ids)) and (ids[i] == ids[j]):
            j += 1
        y_pred = np.mean(prediction_np[i:j], axis=0)
        y_pred[2] = prediction_np[i, 2]
        y_pred[5] = np.max(prediction_np[i:j, 5], axis=0)
        test_probs_studies.append(y_pred)
        i = j
        studies_list.append(paths[i - 1].split('/view')[0])
    test_probs_studies = np.array(test_probs_studies)
    test_studies_list = np.array(studies_list)
    
    
    Atelectasis = test_probs_studies[:, 8, np.newaxis]
    Cardiomegaly = test_probs_studies[:, 2, np.newaxis]
    Consolidation = test_probs_studies[:, 6, np.newaxis]
    Edema = test_probs_studies[:, 5, np.newaxis]
    Pleural_Effusion = test_probs_studies[:, 10, np.newaxis]

    prediction_np = np.concatenate(
        [test_studies_list[:, np.newaxis], Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural_Effusion], axis=1)

    pred_df = pd.DataFrame(prediction_np)
    pred_df.columns = ['Study', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    pred_df.to_csv(output_file, index=False)

def get_all_models():
    model = get_model()
    pNas_model = get_pNASmodel()
    SA_model = get_xCeptionSAmodel()
    pNasSA_model = get_pNASSAmodel()
    XcepFPN_model = get_xCeptionFPNmodel()
    xception_gcn = get_xception_gcn_model()

    return model, pNas_model, SA_model, pNasSA_model, XcepFPN_model, xception_gcn

def get_all_data(TEST_IMAGE_LIST):
    valid_dataloder = get_dataload(TEST_IMAGE_LIST,size=680)
    Nas_dataloder = get_dataload(TEST_IMAGE_LIST, size=331)
    Atel_dataloader = get_dataload(TEST_IMAGE_LIST, size=800)
    dataset_loader = get_dataset(TEST_IMAGE_LIST, size=680)

    return valid_dataloder, Nas_dataloder, Atel_dataloader, dataset_loader