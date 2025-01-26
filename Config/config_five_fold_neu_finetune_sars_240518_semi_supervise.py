'''
Config file: one file to manage them all
Yu Chen
2024.06.12
'''
import torch



################## import model ####################
model = 'XBCR_ACNN'  # 'XBCR_ACNN', 'DeepBCR_ACEXN_protbert'
date = '0530-abag3-nopretrain-noclamp-extradense-acc'
use_onehot = False

if model == 'XBCR_ACNN':
    if use_onehot:
        pass
    else:
        # from lm_gnn_model_jz import XBCR_ACNN_meta as Model
        if 'noclamp' in date:
            pass
        else:
            pass
elif model == 'DeepBCR_ACEXN_protbert':
    pass
else:
    Model = None
    print('Wrong model {}'.format(model))
    raise ValueError

#################### data path #####################
root_dir = './'
hiv_split_seed = date.split('abag')[1].split('-')[0]

#################### dataloader ####################
train_mode = 'hiv-neu'  # 'flu'  # , 'sars', 'flu'        #0330 by jzheng
prop = [ 6, 5, 0, 0, 11]
batch_sz = 64 if 'unfrozenbert' in date else 128

#################### train ####################
rand_seed = 2023

# device
device = torch.device('cuda')  # train params

# pretrain model
pretrain_model_dir = root_dir + 'rslt-meta_XBCR_ACNN_0331_sars+flu+hiv_[2, 3, 2, 1, 2, 1, 2, 2, 4, 2, 1, 1]_[fold2aug]_[valaug1^10f1]_[metaf1wobert]/epoch_52_hiv.pth'

saveaft = 5
fold_list = [4]
lr = 0.0001  # 0.000001
num_epochs = 100

print_step = 20
best_val_epoch = 125

meta_update_iter = 5
regul_step = 100
regul_v = [0.02, 0.001]  # [0.0002, 0.00001]
regul_tgt_dev_rat = 0.08
benchmark = 'acc'

############# sh small test ###############
root_dir = '/home/data/jzheng/bcr_semi_supervise_alpha/'
torch.cuda.set_device(3)
device = torch.device('cuda:3')  # train params
pretrain_model_dir = None
batch_sz = 64
print_step = 1