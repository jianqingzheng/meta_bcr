import os
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader as DataLoader_n
import pandas as pd
import random

from Config.config import get_config
from MetaBCR.lm_gnn_model_jz import XBCR_ACNN_woBERT_meta
from MetaBCR.lm_gnn_model_jz0508_unfrozen import XBCR_ACNN_dense_meta
from MetaBCR.lm_gnn_model_jz import DeepBCR_ACEXN_protbert
from MetaBCR.lm_gnn_model_jz import Adaptive_Regulariz
from MetaBCR.dataset_flu import Ab_Dataset, Ab_Dataset_mean_teacher
import MetaBCR.metrics as metrics
from MetaBCR.losses import *
from MetaBCR.lm_gnn_model_jz0508_unfrozen import Adaptive_Regulariz
import utils

##################################################################

def train(num_fold=None, _cfg_=None):
    if num_fold == None:
        all_folds = [0, 1, 2, 3, 4]
        print("INFO [ train ] : Got num_folds=None, will train all folds")
    else:
        all_folds = [num_fold]
        print(f"INFO [ train ] : Train fold #{num_fold}")

    if _cfg_.model == 'XBCR_ACNN':
        if _cfg_.use_onehot:
            Model = XBCR_ACNN_woBERT_meta
        else:
            Model = XBCR_ACNN_dense_meta
    elif _cfg_.model == 'DeepBCR_ACEXN_protbert':
        Model = DeepBCR_ACEXN_protbert
    else:
        print(f'ERROR [ train ] : Wrong model {_cfg_.model}')
        raise ValueError

    for fold in all_folds:
        
        result_path = f'Results/{_cfg_.train_mode}/rslt-meta_{_cfg_.model}_{_cfg_.date}_{_cfg_.train_mode}_{_cfg_.prop}_fold{fold}_meta{_cfg_.benchmark}-semi/'
        if not os.path.exists(result_path):
            print('INFO [ train ] : Cannot find <RESULT DIR>, created a new one.')
            os.makedirs(result_path)
        print('INFO [ train ] : <RESULT DIR>: {}'.format(result_path))

        random.seed(_cfg_.rand_seed)

        #################### dataloader ####################

        data_train_1 = utils.read_table(_cfg_.fdir_train_1.format(fold))
        data_train_0 = utils.read_table(_cfg_.fdir_train_0.format(fold))

        data_train_non_experiment_list = []
        for fdir_train_non_exp in _cfg_.fdir_train_non_experiment_list:
            data_train_non_experiment_list.append(utils.read_table(fdir_train_non_exp))

        data_train_nolabel = utils.read_table(_cfg_.fdir_train_nolabel)

        train_set = Ab_Dataset_mean_teacher(datalist=[data_train_1,
                                                      data_train_0,
                                                      [data_train_1, data_train_non_experiment_list[0]],
                                                      [data_train_1, data_train_non_experiment_list[1]],
                                                      [data_train_nolabel, data_train_1]
                                                      ],
                                            proportions=_cfg_.prop,
                                            sample_func=['rand_sample',
                                                         'rand_sample',
                                                         'rand_sample_rand_combine',
                                                         'rand_sample_rand_combine',
                                                         'no_label'
                                                         ],
                                            n_samples=max(data_train_1.shape[0] + data_train_0.shape[0], 1024),
                                            is_rand_sample=True, onehot=_cfg_.use_onehot, rand_shift=True)

        train_loader = DataLoader_n(dataset=train_set, batch_size=_cfg_.batch_sz, shuffle=False)

        data_val_1 = utils.read_table(_cfg_.fdir_val_1.format(fold))
        data_val_0 = utils.read_table(_cfg_.fdir_val_0.format(fold))

        data_val = pd.concat([data_val_1, data_val_0], ignore_index=True)

        val_set = Ab_Dataset(datalist=[data_val], proportions=[None], sample_func=['sample'],
                             n_samples=data_val.shape[0], is_rand_sample=False, onehot=_cfg_.use_onehot,
                             rand_shift=False)
        val_loader = DataLoader_n(dataset=val_set, batch_size=_cfg_.batch_sz, shuffle=False)
        meta_loader = DataLoader_n(dataset=val_set, batch_size=_cfg_.batch_sz, shuffle=False)

        tst_loader_dict = {}
        for k,v in _cfg_.fdir_tst_dict.items():
            data_tst = utils.read_table(v)
            tst_set = Ab_Dataset(datalist=[data_tst], proportions=[None], sample_func=['sample'],
                                n_samples=data_tst.shape[0], is_rand_sample=False,
                                onehot=_cfg_.use_onehot, rand_shift=False)
            tst_loader = DataLoader_n(dataset=tst_set, batch_size=_cfg_.batch_sz, num_workers=0,
                                        shuffle=False)
            tst_loader_dict[k] = [data_tst, tst_loader]

        #################### train ####################

        print('INFO [ train ] : Start training ...')
        model = utils.get_model(Model=Model, _cfg_=_cfg_)
        model.to(_cfg_.device)

        # load state dict
        if _cfg_.pretrain_model_dir is not None:
            pretrain_model_dir = os.path.join(_cfg_.pretrain_model_dir, f'fold{fold}.pth')
            model.load_state_dict(torch.load(pretrain_model_dir), strict=False)  # 0107
            print(f'INFO [ train ] : Loaded pretrain model params from {pretrain_model_dir}')
        else:
            model.apply(utils.init_weights)  # Yu 2024/1/4
            print('INFO [ train ] : No pretrained models, initialize model weights')

        supervise_criterion = nn.BCELoss(reduction='none')
        val_criterion = nn.BCELoss()
        unsupervise_criterion = nn.MSELoss(reduction='none')
        optimizer = torch.optim.SGD(model.parameters(), lr=_cfg_.lr)
        # adaptive regularize
        adaptive_regular = Adaptive_Regulariz(velocity=_cfg_.regul_v,
                                              target_deviation_ratio=_cfg_.regul_tgt_dev_rat)  # velocity=[0.02,0.001], target_deviation_ratio=0.08
        weight_regulariz_neu = adaptive_regular.weight

        # train the network
        if _cfg_.train_mode == 'sars+flu':
            _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = {'sars': 0, 'flu': 0}, {'sars': 0, 'flu': 0}, {
                'sars': 0, 'flu': 0}, {'sars': 0, 'flu': 0}
        elif _cfg_.train_mode == 'sars+flu+hiv':  ### 240314
            _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = {'sars': 0, 'flu': 0, 'hiv': 0}, {'sars': 0,
                                                                                                          'flu': 0,
                                                                                                          'hiv': 0}, {
                                                                            'sars': 0, 'flu': 0, 'hiv': 0}, {'sars': 0,
                                                                                                             'flu': 0,
                                                                                                             'hiv': 0}
        else:
            _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = 0, 0, 0, 0
        global_iter = 0

        # fast model
        fast_model = model.clone_meta(device=_cfg_.device)

        fast_optimizer = utils.get_optimizer(fast_model, _cfg_, None)
        train_acc = []
        train_loss = []
        train_epoch = []
        train_precision = []
        train_recall = []

        val_loss = []
        val_acc = []
        val_precision = []
        val_recall = []
        for epoch in range(_cfg_.num_epochs):
            running_loss = 0.0  # for loss printing
            predictions_tr, labels_tr, has_label_mask_tr = [], [], []
            weight_regulariz = [weight_regulariz_neu, 0.]

            meta_lr = _cfg_.lr * 1e4 * (1.01 - epoch / float(_cfg_.num_epochs))
            utils.set_learning_rate(optimizer, meta_lr)

            for i, data in enumerate(train_loader, 0):
                global_iter += 1
                # data pre-process
                input_ids_ab_v = data['input_ids_ab_v'].to(_cfg_.device)
                attention_mask_ab_v = data['attention_mask_ab_v'].to(_cfg_.device)
                input_ids_ab_l = data['input_ids_ab_l'].to(_cfg_.device)
                attention_mask_ab_l = data['attention_mask_ab_l'].to(_cfg_.device)
                input_ids_ag = data['input_ids_ag'].to(_cfg_.device)
                labels = data['label'].to(_cfg_.device)
                has_label_mask = data['has_label'].unsqueeze(1).to(_cfg_.device)
                input_ids_ab_v_origin = data['input_ids_ab_v_origin'].to(_cfg_.device)
                attention_mask_ab_v_origin = data['attention_mask_ab_v_origin'].to(_cfg_.device)
                input_ids_ab_l_origin = data['input_ids_ab_l_origin'].to(_cfg_.device)
                attention_mask_ab_l_origin = data['attention_mask_ab_l_origin'].to(_cfg_.device)

                # fast learning
                fast_loss = utils.fast_learn(fast_model, supervise_criterion, fast_optimizer, _cfg_,
                                       model, unsupervise_criterion,
                                       X=[input_ids_ab_v, attention_mask_ab_v,
                                          input_ids_ab_l, attention_mask_ab_l,
                                          input_ids_ag,
                                          input_ids_ab_v_origin, attention_mask_ab_v_origin,
                                          input_ids_ab_l_origin, attention_mask_ab_l_origin],
                                       Y=labels, mask=has_label_mask)
                # update slow model
                if i % _cfg_.meta_update_iter == 0:
                    # zero the meta-parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v, attention_mask_ab_v=attention_mask_ab_v,
                                    ab_l=input_ids_ab_l, attention_mask_ab_l=attention_mask_ab_l)

                    # regularize
                    loss_regulariz = model.get_variables()
                    loss = sum([w * x for w, x in zip(weight_regulariz, loss_regulariz)])
                    # backward and optimize
                    loss.backward()
                    optimizer.step()
                    # fast MAML
                    optimizer.zero_grad()
                    model.point_grad_to(fast_model, _cfg_.device)  ####
                    optimizer.step()

                    fast_model.load_state_dict(model.state_dict())

                    fast_state = fast_optimizer.state_dict()  # save fast optimizer state
                    fast_optimizer = utils.get_optimizer(fast_model, _cfg_, state=fast_state)  ####

                    # store predictions and labels
                    predictions_tr.extend(outputs[0].cpu().view(-1).tolist())
                    labels_tr.extend(labels.view(-1).tolist())
                    has_label_mask_tr.extend(has_label_mask.view(-1).tolist())
                    # print statistics
                    running_loss += loss.item()

                running_loss += fast_loss.item()

                if i % _cfg_.print_step == (_cfg_.print_step - 1):
                    print(
                        f'INFO [ train ] : Training '
                        f'Epoch {epoch} Iter {i + 1:5d}, '
                        f'loss: {running_loss / _cfg_.print_step:.6f}, '
                        f'wregul: {weight_regulariz_neu:.3f}')
                    running_loss = 0.0

                # meta
                if global_iter % _cfg_.regul_step == (_cfg_.regul_step - 1):
                    # get train confusion matrix
                    confusion_mat_tr = metrics.get_confusion_mat(predictions_tr, labels_tr, has_label_mask_tr)
                    eval_confusion_tr = metrics.eval_confusion(confusion_mat_tr)
                    # validation
                    model.eval()
                    predictions_val, labels_val, lossweight_val = utils.implement(model, meta_loader, _cfg_)
                    model.train()
                    # get validate confusion matrix
                    confusion_mat_val = metrics.get_confusion_mat(
                        [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
                        [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
                    eval_confusion_val = metrics.eval_confusion(confusion_mat_val)
                    # update regularize weight
                    if _cfg_.benchmark == 'acc':
                        weight_regulariz_neu = adaptive_regular.update_weight(-eval_confusion_tr[0],
                                                                              -eval_confusion_val[0])  ### 240314
                    else:
                        weight_regulariz_neu = adaptive_regular.update_weight(-eval_confusion_tr[6],
                                                                              -eval_confusion_val[6])  ### 240314

            # get train confusion matrix
            confusion_mat_tr = metrics.get_confusion_mat(predictions_tr, labels_tr, has_label_mask_tr)
            eval_confusion_tr = metrics.eval_confusion(confusion_mat_tr)
            train_loss.append(running_loss / len(train_loader))
            train_epoch.append(epoch)
            train_acc.append(eval_confusion_tr[0])
            train_precision.append(eval_confusion_tr[3])
            train_recall.append(eval_confusion_tr[1])
            # validation
            model.eval()
            predictions_val, labels_val, lossweight_val = utils.implement(model, val_loader, _cfg_)
            model.train()
            # get validate confusion matrix
            confusion_mat_val = metrics.get_confusion_mat(
                [predictions_val[i] for (i, v) in enumerate(lossweight_val) if v == 1],
                [labels_val[i] for (i, v) in enumerate(lossweight_val) if v == 1])
            eval_confusion_val = metrics.eval_confusion(confusion_mat_val)

            val_loss.append(
                val_criterion(torch.Tensor(predictions_val), torch.Tensor(labels_val)).item())
            val_acc.append(eval_confusion_val[0])
            val_precision.append(eval_confusion_val[3])
            val_recall.append(eval_confusion_val[1])

            print(f"INFO [ train ] : Validation "
                  f"Epoch {epoch}, acc: {eval_confusion_val[0]:.2f}, "
                  f"ppv: {eval_confusion_val[3]:.2f}, sns: {eval_confusion_val[1]:.2f}")
            if _cfg_.benchmark == 'acc':
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[0] > best_val_acc):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{result_path}epoch_{epoch}.pth')

                    for tst_name,tst_v in tst_loader_dict.items():
                        data_tst, tst_ldr = tst_v
                        model_name = f'{_cfg_.date}_{epoch}_fold{fold}-maml'
                        model.eval()
                        predictions_tst, labels_tst, lossweight_tst = utils.implement(model, tst_ldr, _cfg_)
                        model.train()
                        data_tst['output'] = np.around(np.array(predictions_tst)).tolist()
                        data_tst['predict'] = predictions_tst
                        data_tst.to_excel(
                            f"{result_path}{tst_name}_{model_name}_flu_binding_test.xlsx",
                            index=False, header=True)

            if _cfg_.benchmark == 'f1':
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[6] > best_val_f1):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{result_path}epoch_{epoch}.pth')

                    for tst_name, tst_v in tst_loader_dict.items():
                        data_tst, tst_ldr = tst_v
                        model_name = f'{_cfg_.date}_{epoch}_fold{fold}-maml'
                        model.eval()
                        predictions_tst, labels_tst, lossweight_tst = utils.implement(model, tst_ldr, _cfg_)
                        model.train()
                        data_tst['output'] = np.around(np.array(predictions_tst)).tolist()
                        data_tst['predict'] = predictions_tst
                        data_tst.to_excel(
                            f"{result_path}{tst_name}_{model_name}_flu_binding_test.xlsx",
                            index=False, header=True)

            else:
                if (epoch >= _cfg_.saveaft or epoch == 0) and (eval_confusion_val[-1] > best_val_f1):
                    _best_val_epoch, best_val_acc, best_val_prec, best_val_f1 = epoch, eval_confusion_val[0], \
                                                                                eval_confusion_val[3], \
                                                                                eval_confusion_val[6]
                    torch.save(model.state_dict(), f'{result_path}epoch_{epoch}.pth')

            train_file = pd.DataFrame({'epoch': train_epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                                       'train_precision': train_precision, 'train_recall': train_recall,
                                       'val_loss': val_loss, 'val_acc': val_acc,
                                       'val_precision': val_precision, 'val_recall': val_recall, })
            train_file.to_excel(f'{result_path}{model_name}_train_file.xlsx', index=False)
        print(f'INFO [ train ] : Finished Training, '
              f'best model from epoch {_best_val_epoch}, '
              f'f1 {best_val_f1:.2f}, acc {best_val_acc:.2f}, ppv {best_val_prec:.2f}')

if __name__ == '__main__':
    # flu bind
    # configure = get_config("Config/config_five_fold_flu_bind_meta_240621_semi_supervise.json")

    # flu neutral
    # configure = get_config("Config/config_five_fold_flu_neu_meta_240903_unfrozen_semi_supervise.json")
    
    # sars bind
    # configure = get_config("Config/config_five_fold_sars_bind_meta_240612_unfrozen_semi_supervise.json")

    # sars neutral
    configure = get_config("Config/config_five_fold_sars_neu_meta_new_data_241019_unfrozen_semi_supervise.json")

    train(num_fold=None, _cfg_=configure)
