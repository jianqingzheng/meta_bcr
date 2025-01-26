'''
Config file: one file to manage them all
Yu Chen
2024.06.12
'''
import torch
import os

################## import model ####################
model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'

meta_update_iter = 10
num_epochs = 100
unsup_loss_weight = 1
benchmark = 'acc'
prop = [2, 4, 4, 1, 2]
lr = 0.00001
date_step = f'0903-updt_itr={meta_update_iter}-{lr}-{benchmark}-w_unsup_loss={unsup_loss_weight}'
date = f'{date_step}-abag3-{prop}-semi' #-newneg

device = torch.device('cuda')  # train params

os.makedirs(f'{date_step}_BNT_neu_results_semi', exist_ok=True)

use_onehot = False


bert_name = '20240531_BNT_epoch5'

#################### data path #####################
root_dir = '/fs1/home/caolab/bcr_semi_supervise/'
hiv_split_seed = date.split('abag')[1].split('-')[0]

#################### dataloader ####################
train_mode = 'sars-neu'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng


#################### train ####################
rand_seed = 2023

# model
freeze_layer_count = 100
freeze_bert = True

# device
# device = torch.device('cuda')  # train params

# pretrain model
# pretrain_model_dir = root_dir + 'rslt-meta_XBCR_ACNN_0331_sars+flu+hiv_[2, 3, 2, 1, 2, 1, 2, 2, 4, 2, 1, 1]_[fold2aug]_[valaug1^10f1]_[metaf1wobert]/epoch_52_hiv.pth'

saveaft = 0


print_step = 20
best_val_epoch =0

regul_step = 100
regul_v = [0.02, 0.001]  # [0.0002, 0.00001]
regul_tgt_dev_rat = 0.08


############# sh small test ###############
pretrain_model_dir = root_dir + '0530-sars-neu/fold{}.pth'
# pretrain_model_dir = None
batch_sz = 128
print_step = 20
