import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as Dataset_n
from torch.utils.data import DataLoader as DataLoader_n
import numpy as np
import pandas as pd
import random
from lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta as Model
import argparse

from lm_gnn_model_jz import Adaptive_Regulariz
from dataset_sars import Ab_Dataset  # Ab_Dataset_augment, Ab_Dataset_augment_cross, Ab_Dataset_wo_label
from metrics import *
from losses import *

parser = argparse.ArgumentParser()

# HYPER PARAM
_train_mode = 'neu'  # 'sars+neu', 'sars', 'neu'
_model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'
_device = torch.device('cuda')  # train params
_batch_sz = 4096
_use_onehot = False
# _use_onehot = True
_lr = 0.00001
_print_step = 20
_regul_step = 100
_regul_v = [0.02, 0.001]
_regul_tgt_dev_rat = 0.08
_best_val_epoch = 125
_rand_seed = 2023

_root_dir = './'

_RESULT_DIR = _root_dir + '240619_rbd_bind_results/'  # output dir
if not os.path.exists(_RESULT_DIR):
    print('Cannot find [RESULT DIR], created a new one.')
    os.makedirs(_RESULT_DIR)
print('[RESULT DIR]: {}'.format(_RESULT_DIR))


def implement(model, dataloader, wolabel=False):
    '''
    Model implement function.
        input: model & dataloader
        output: prediction, labels, and loss weights
    '''
    predictions_main_tr = []
    labels_main_tr = []
    lossweight_main_tr = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # data pre-process
            input_ids_ab_v = data['input_ids_ab_v'].to(_device)
            attention_mask_ab_v = data['attention_mask_ab_v'].to(_device)
            input_ids_ab_l = data['input_ids_ab_l'].to(_device)
            attention_mask_ab_l = data['attention_mask_ab_l'].to(_device)
            input_ids_ag = data['input_ids_ag'].to(_device)
            # prediction
            outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v,
                            attention_mask_ab_v=attention_mask_ab_v,
                            ab_l=input_ids_ab_l,
                            attention_mask_ab_l=attention_mask_ab_l, )
            # loss
            predictions_main_tr.extend(outputs[0].cpu().view(-1).tolist())
            if wolabel == False:
                labels_main_tr.extend(data['label'].view(-1).tolist())
                lossweight_main_tr.extend(data['loss_main'].view(-1).tolist())
    if wolabel:
        return predictions_main_tr
    return predictions_main_tr, labels_main_tr, lossweight_main_tr


def font(out, gt):
    # this is for visualizing the correctness of predictions,
    # 'o' for correct predictions, 'x' for false predictions.
    ff = []
    for i in range(len(out)):
        fff = 'o' if out[i] == gt[i] else 'x'
        ff.append(fff)
    return ff


bert_name = 'prot_bert'
model = Model(extra_dense=True, block_num=8,
                     freeze_bert=True, ab_freeze_layer_count=30,
                     bert=bert_name
                     )
# model = nn.DataParallel(model)
model.to(_device)

## BCR
# _fdir_tst_neu = _root_dir + "1222_merged_A1-A11_BCR.xlsx"
# test_name = '1222_merged_A1-A11_BCR'
_fdir_tst_neu = _root_dir + "1222-add_merged_A1-A11_BCR.xlsx"
test_name = '1222-add_merged_A1-A11_BCR'
# _fdir_tst_neu = _root_dir + "1030_WU368_kim_et_al_nature_2022_igblast.xlsx"
# test_name = '1030_WU368_kim_et_al_nature_2022_BCR'
# _fdir_tst_neu = _root_dir + "1030_WU368_kim_et_al_nature_2022_igblast_2.xlsx"
# test_name = '1030_WU368_kim_et_al_nature_2022-2_BCR'
# _fdir_tst_neu = _root_dir + "WU382_alsoussi_after_qc_bcrs_to_pred-1.xlsx"
# test_name = '1212_WU382_alsoussi_BCR'
# _fdir_tst_neu = _root_dir + "WU382_alsoussi_after_qc_bcrs_to_pred-2.xlsx"
# test_name = '1212_WU382_alsoussi-2_BCR'
data_test_neu = pd.DataFrame()
data_test = pd.read_excel(_fdir_tst_neu)
ags = pd.read_excel('./SARS-example-antigen_to_predict2_0903.xlsx')
# for i in ['SARS-CoV-2','SARS-CoV-2_Omicron-BA1','SARS-CoV-2_Omicron-BA4','XBB.1.5']:
for i in ags['variant_name'].values:
    data_test['Antig Name'] = i
    data_test['variant_seq'] = ags.loc[ags['variant_name'] == i, 'variant_seq'].values[0]
    data_test_neu = pd.concat([data_test_neu, data_test], ignore_index=True)
data_test_neu['rbd'] = 0
test_set_rbd = Ab_Dataset(datalist=[data_test_neu], proportions=[None], sample_func=['sample'],
                              n_samples=data_test_neu.shape[0], is_rand_sample=False,
                              onehot=_use_onehot, rand_shift=False)
test_loader_rbd = DataLoader_n(dataset=test_set_rbd, batch_size=_batch_sz, num_workers=0, shuffle=False)
model_dirs = [#"rslt-meta_XBCR_ACNN_0618-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert30-4231_sars-bind_[4, 2, 3, 1]_fold3_metaacc/epoch_79.pth",
# 'rslt-meta_XBCR_ACNN_0612-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert-1221_sars-bind_[1, 2, 2, 1]_fold2_metaacc/epoch_20.pth',
# 'rslt-meta_XBCR_ACNN_0612-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert-A1A11-1221_sars-bind_[1, 2, 2, 1]_fold3_metaacc/epoch_22.pth'
    
# 'rslt-meta_XBCR_ACNN_0618-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert15-4231_sars-bind_[4, 2, 3, 1]_fold4_metaacc/epoch_90.pth'
# 'rslt-meta_XBCR_ACNN_0618-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert0-4231_sars-bind_[4, 2, 3, 1]_fold4_metaacc/epoch_30.pth'
# 'rslt-meta_XBCR_ACNN_0618-abag3-finetunedbatch-extradense-noclamp-acc-unfrozenbert5-4231_sars-bind_[4, 2, 3, 1]_fold4_metaacc/epoch_48.pth'

# 'rslt-meta_XBCR_ACNN_0627-abag3-finetuned-extradense-noclamp-acc-2211_sars-bind_[2, 2, 1, 1]_fold0_metaacc/epoch_11.pth',
# "rslt-meta_XBCR_ACNN_0627-abag3-finetuned-extradense-noclamp-acc-2211_sars-bind_[2, 2, 1, 1]_fold3_metaacc/epoch_4.pth"
# "rslt-meta_XBCR_ACNN_0702-abag3-finetuned-extradense-noclamp-acc-2211_sars-bind_[2, 2, 1, 1]_fold3_metaacc/epoch_98.pth",
# "rslt-meta_XBCR_ACNN_0702-abag3-finetuned-extradense-noclamp-acc-2211_sars-bind_[2, 2, 1, 1]_fold0_metaacc/epoch_79.pth",

# "rslt-meta_XBCR_ACNN_0702-abag3-finetuned-extradense-noclamp-acc-1121_sars-bind_[1, 1, 2, 1]_fold2_metaacc/epoch_93.pth"
# "rslt-meta_XBCR_ACNN_0702-abag3-finetuned-extradense-noclamp-acc-1121_sars-bind_[1, 1, 2, 1]_fold3_metaacc/epoch_84.pth"

# "rslt-meta_XBCR_ACNN_0706-abag3-finetuned-extradense-noclamp-acc-1121_sars-bind_[1, 1, 2, 1]_fold0_metaacc/epoch_64.pth"
# "rslt-meta_XBCR_ACNN_0706-abag3-finetuned-extradense-noclamp-acc-1121_sars-bind_[1, 1, 2, 1]_fold3_metaacc/epoch_75.pth"
"rslt-meta_XBCR_ACNN_0708-abag3-finetuned-extradense-noclamp-acc-11221_sars-bind_[1, 1, 2, 2, 1]_fold3_metaacc/epoch_79.pth"
# "rslt-meta_XBCR_ACNN_0708-abag3-finetuned-extradense-noclamp-acc-11221_sars-bind_[1, 1, 2, 2, 1]_fold2_metaacc/epoch_96.pth"
            ]
model_names = [#'0618MetaBCR-unfrozenbert30-4231_fold3'
# '0612MetaBCR-unfrozenbert-1221_20_fold2'
# '0612MetaBCR-A1A11-1221_22_fold3'
    
# '0618MetaBCR-unfrozenbert15-4231_fold4'
# '0618MetaBCR-unfrozenbert0-4231_fold4'
# '0618MetaBCR-unfrozenbert5-4231_fold4'
# '0627MetaBCR-unfrozenbert30-2211_fold0',
# '0627MetaBCR-unfrozenbert30-2211_fold3'
# '0702MetaBCR-unfrozenbert30-2211_fold3',
# '0702MetaBCR-unfrozenbert30-2211_fold0',
# '0702MetaBCR-unfrozenbert30-1121_fold2',
# '0702MetaBCR-unfrozenbert30-1121_fold3',

# '0706MetaBCR-unfrozenbert30-1121_fold0',
# '0706MetaBCR-unfrozenbert30-1121_fold3',
'0708MetaBCR-unfrozenbert30-11221_fold3',
# '0708MetaBCR-unfrozenbert30-11221_fold2',
    ]

for model_name, model_dir in zip(model_names, model_dirs):
    print('Testing `{}`'.format(model_dir))
    # test implement
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    predictions_tst, labels_tst, lossweight_tst = implement(model, test_loader_rbd, )

    data_test_neu['output'] = np.around(np.array(predictions_tst)).tolist()
    data_test_neu['predict'] = predictions_tst
    data_test_neu.to_csv(f"{_RESULT_DIR}{test_name}_{model_name}_rbd_binding_test.csv")