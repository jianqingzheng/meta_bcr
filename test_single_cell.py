import os
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd

from Config.config import get_config, combine_configs
from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta
from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta
from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert
from MetaBCR.dataset_flu import Ab_Dataset
from MetaBCR.losses import *
import utils


def test_single_cell(_cfg_=None):
   
    if _cfg_.model == 'XBCR_ACNN':
        if _cfg_.use_onehot:
            Model = XBCR_ACNN_woBERT_meta
        else:
            Model = XBCR_ACNN_dense_meta
    elif _cfg_.model == 'DeepBCR_ACEXN_protbert':
        Model = DeepBCR_ACEXN_protbert
    else:
        print(f'ERROR [ test ] : Wrong model {_cfg_.model}')
        raise ValueError
        
    result_path = f'Results/{_cfg_.train_mode}/rslt-meta_{_cfg_.model}_{_cfg_.date}_{_cfg_.train_mode}_{_cfg_.prop}_fold{_cfg_.test_fold}_meta{_cfg_.benchmark}-semi/'
    if not os.path.exists(result_path):
        print('INFO [ test ] : Cannot find <RESULT DIR>, created a new one.')
        os.makedirs(result_path)
    print('INFO [ test ] : <RESULT DIR>: {}'.format(result_path))

    print('INFO [ test ] : Start testing ...')
    model = utils.get_model(Model=Model, _cfg_=_cfg_)
    model.to(_cfg_.device)
    
    data_test = pd.DataFrame()
    data_test_abs = utils.read_table(_cfg_.fdir_tst_flu)
    data_test_ags = utils.read_table(_cfg_.fdir_ags)

    for i in data_test_ags['variant_name'].values:
        data_test_abs['Antig Name'] = i
        data_test_abs['variant_seq'] = data_test_ags.loc[data_test_ags['variant_name'] == i, 'variant_seq'].values[0]
        data_test = pd.concat([data_test, data_test_abs], ignore_index=True)
    data_test['rbd'] = 0

    test_set = Ab_Dataset(datalist=[data_test], proportions=[None], sample_func=['sample'],
                                n_samples=data_test.shape[0], is_rand_sample=False, onehot=_cfg_.use_onehot,
                                rand_shift=False)
    test_loader = DataLoader_n(dataset=test_set, batch_size=_cfg_.batch_sz, shuffle=False)
    
    print('INFO [ test ] : Testing `{}`'.format(_cfg_.dir_model_to_test))
    # test implement
    model.load_state_dict(torch.load(_cfg_.dir_model_to_test), strict=False)
    model.eval()
    predictions_tst, labels_tst, lossweight_tst = utils.implement(model, test_loader, _cfg_)

    data_test['output'] = np.around(np.array(predictions_tst)).tolist()
    data_test['predict'] = predictions_tst
    data_test.to_csv(f"{result_path}{_cfg_.test_name}_test.csv")
    print('INFO [ test ] : Results saved to `{}`'.format(f"{result_path}{_cfg_.test_name}_test.csv"))

if __name__ == '__main__':
    configure_train = get_config("Config/config_five_fold_flu_bind_meta_240621_semi_supervise.json")
    configure_test = get_config("Config/config_test_single_cell_flu_bind.json")
    configure = combine_configs(configure_train, configure_test)

    test_single_cell(_cfg_=configure)
