'''
Config file: one file to manage them all
Yu Chen
2024.06.12
'''
import torch
import os

################## import model ####################
model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'

meta_update_iter = 5
num_epochs = 100
unsup_loss_weight = 1
prop = [ 2, 2, 1, 1, 1.2]
benchmark = 'acc'

date_step = f'0905-updt_itr={meta_update_iter}-epoch={num_epochs}-{benchmark}-w_unsup_loss={unsup_loss_weight}'
date = f'{date_step}-abag3-semi' #-newneg

device = torch.device('cuda')  # train params

os.makedirs(f'{date_step}_flu_neu_results_semi', exist_ok=True)

use_onehot = False


if 'Influenza' in date:
    bert_name = '240612_Influenza_epoch10'
else:
    bert_name = 'prot_bert'

#################### data path #####################
root_dir = '/fs1/home/caolab/bcr_semi_supervise/'
hiv_split_seed = date.split('abag')[1].split('-')[0]

#################### dataloader ####################
train_mode = 'flu-neu'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng

#################### train ####################
rand_seed = 2023

# model
freeze_layer_count = 100
freeze_bert = True

saveaft = 5 
lr =0.00001

print_step = 20
best_val_epoch = 125

regul_step = 100
regul_v = [0.02, 0.001]  # [0.0002, 0.00001]
regul_tgt_dev_rat = 0.08

############# sh small test ###############
# pretrain_model_dir = '/fs1/home/caolab/yuchen_deepBCR/rslt-meta_XBCR_ACNN_0620-abag3-nopretrain-noclamp-extradense-acc2131_flu-neu_[2, 1, 3, 1]_fold2_metaacc/epoch_65.pth'
pretrain_model_dir = '/fs1/home/caolab/yuchen_deepBCR/0530-flu-neu/fold{}.pth'
batch_sz = 32
