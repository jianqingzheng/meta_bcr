'''
Config file: one file to manage them all
Yu Chen
2024.06.12
'''
import torch
import os
################## import model ####################
prop = [1, 5, 3, 1, 1.2]
model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'
benchmark = 'acc'
date_step = '1026'
unsup_loss_weight = 1
meta_update_iter = 50

date = f'{date_step}-abag3-{benchmark}-{prop}-semi_unfrozenbert'
use_onehot = False

os.makedirs(f'{date_step}_rbd_neu_results_semi', exist_ok=True)

# bert_name = '20240531_BNT_epoch5'
if 'bnt' in date:
    bert_name = '20240531_BNT_epoch5'
elif 'A1A11' in date:
    bert_name = '20240603_A1-A11_epoch10'
else:
    bert_name = 'prot_bert'
#################### data path #####################
# root_dir = '/fs1/home/caolab/bcr_semi_supervise/'
root_dir = '/home/data/jzheng/bcr_semi_supervise_alpha/'
hiv_split_seed = date.split('abag')[1].split('-')[0]

#################### dataloader ####################

train_mode = 'sars-neu'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng
batch_sz = 128

#################### train ####################
rand_seed = 2023

## model
# freeze_layer_count = 100
# freeze_bert = True

if 'unfrozenbert' in date:
    batch_sz = 64
    freeze_bert = False
    freeze_layer_count = 20
else:
    batch_sz = 128
    freeze_bert = True
    freeze_layer_count = 100

# device
# device = torch.device('cuda')  # train params
device = torch.device('cuda:3')  # train params

# pretrain model
pretrain_model_dir = root_dir + '0530-sars-neu/fold{}.pth'

saveaft = 0
lr = 0.00001  # 0.000001
# num_epochs = 100
num_epochs = 50

print_step = 20
best_val_epoch = 0

regul_step = 100
regul_v = [0.02, 0.001]  # [0.0002, 0.00001]
regul_tgt_dev_rat = 0.08

