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

date_step = f'0822-bigdata-updt_itr={meta_update_iter}-epoch={num_epochs}-f1-w_unsup_loss={unsup_loss_weight}'
date = f'{date_step}-abag3-nopretrain-extradense-noclamp-acc-2211(1.2)-semi'

device = torch.device('cuda')  # train params

os.makedirs(f'{date_step}_flu_bind_results_semi', exist_ok=True)

use_onehot = False


if 'Influenza' in date:
    bert_name = '240612_Influenza_epoch10'
else:
    bert_name = 'prot_bert'

#################### data path #####################
root_dir = '/fs1/home/caolab/bcr_semi_supervise/'
hiv_split_seed = date.split('abag')[1].split('-')[0]

#################### dataloader ####################
train_mode = 'flu-bind'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng
prop = [ 2, 2, 1, 1, 1.2]

#################### train ####################
rand_seed = 2023

# model
freeze_layer_count = 30 if 'unfrozenbert' in date else 100
freeze_bert = False if 'unfrozenbert' in date else True

# device
# device = torch.device('cuda')  # train params

# pretrain model
# pretrain_model_dir = root_dir + 'rslt-meta_XBCR_ACNN_0331_sars+flu+hiv_[2, 3, 2, 1, 2, 1, 2, 2, 4, 2, 1, 1]_[fold2aug]_[valaug1^10f1]_[metaf1wobert]/epoch_52_hiv.pth'

saveaft = 5 if 'nopretrain' in date else 0
lr = 0.00001 if 'nopretrain' in date else 0.00001


print_step = 20
best_val_epoch = 125

regul_step = 100
regul_v = [0.02, 0.001]  # [0.0002, 0.00001]
regul_tgt_dev_rat = 0.08
benchmark = 'acc'

############# sh small test ###############

pretrain_model_dir = root_dir + '0612-flu-bind/fold{}.pth'
batch_sz = 32
print_step = 20
