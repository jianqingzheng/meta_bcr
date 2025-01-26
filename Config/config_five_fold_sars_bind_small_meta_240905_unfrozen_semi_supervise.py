'''
Config file: one file to manage them all
Yu Chen
2024.06.12
'''
import torch
import os
################## import model ####################
prop = [4, 2, 3, 1, 2]
model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'
benchmark = 'acc'
date_step = '0905-small'
unsup_loss_weight = 1
meta_update_iter = 5

date = f'{date_step}-abag3-{benchmark}-{prop}-semi'
use_onehot = False

os.makedirs(f'{date_step}_rbd_bind_results_semi', exist_ok=True)

bert_name = '20240603_A1-A11_epoch10'

#################### data path #####################
# root_dir = '/fs1/home/caolab/bcr_semi_supervise/'
root_dir = '/home/data/jzheng/bcr_semi_supervise_alpha/'
hiv_split_seed = date.split('abag')[1].split('-')[0]

#################### dataloader ####################

train_mode = 'sars-bind'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng
batch_sz = 128

#################### train ####################
rand_seed = 2023

# model
freeze_layer_count = 100
freeze_bert = True

# device
# device = torch.device('cuda')  # train params
device = torch.device('cuda:3')  # train params

# pretrain model
pretrain_model_dir = root_dir + '0611-sars-bind/fold{}.pth'

saveaft = 0
lr = 0.00001  # 0.000001
num_epochs = 100


print_step = 20
best_val_epoch = 0

regul_step = 100
regul_v = [0.02, 0.001]  # [0.0002, 0.00001]
regul_tgt_dev_rat = 0.08

