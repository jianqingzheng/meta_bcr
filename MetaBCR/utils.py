import os
import torch.nn as nn
import torch.nn.init as init
import pandas as pd

from MetaBCR.losses import *


def read_table(file):
    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_excel(file)
    return data


def read_tables(files):
    if isinstance(files, str):
        data = read_table(files)
    else:
        data = pd.concat([read_table(f) for f in files], ignore_index=True)
    return data


def get_model(Model, _cfg_):
    return Model(extra_dense=True, block_num=8,
                 freeze_bert=_cfg_.freeze_bert,
                 ab_freeze_layer_count=_cfg_.freeze_layer_count,
                 bert=_cfg_.bert_name)


def init_weights(m):
    print('Initializing model weight ...')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)  # he ini
        # init.orthogonal_(m.weight.data)  # orthogonal inic
        # init.orthogonal_(m.weight.data)  # xaiver / glorot ini
        # init.normal_(m.weight.data, mean=0, std=0.01)  # normal distribution ini
        if m.bias is not None: init.constant_(m.bias.data, 0.01)  # preventing zero bias


def font(out, gt):
    # this is for visualizing the correctness of predictions,
    # 'o' for correct predictions, 'x' for false predictions.
    ff = []
    for i in range(len(out)):
        fff = 'o' if out[i] == gt[i] else 'x'
        ff.append(fff)
    return ff


def implement(model, dataloader, _cfg_, wolabel=False):
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
            input_ids_ab_v = data['input_ids_ab_v'].to(_cfg_.device)
            attention_mask_ab_v = data['attention_mask_ab_v'].to(_cfg_.device)
            input_ids_ab_l = data['input_ids_ab_l'].to(_cfg_.device)
            attention_mask_ab_l = data['attention_mask_ab_l'].to(_cfg_.device)
            input_ids_ag = data['input_ids_ag'].to(_cfg_.device)
            # prediction
            outputs = model(ag_x=input_ids_ag, ab_x=input_ids_ab_v,
                            attention_mask_ab_v=attention_mask_ab_v,
                            ab_l=input_ids_ab_l,
                            attention_mask_ab_l=attention_mask_ab_l)
            # loss
            predictions_main_tr.extend(outputs[0].cpu().view(-1).tolist())
            if wolabel == False:
                labels_main_tr.extend(data['label'].view(-1).tolist())
                lossweight_main_tr.extend(data['loss_main'].view(-1).tolist())
    if wolabel: return predictions_main_tr
    return predictions_main_tr, labels_main_tr, lossweight_main_tr


def get_optimizer(net, _cfg_, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=_cfg_.lr, betas=(0, 0.999))
    if state is not None: optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def model_deep_duplicate(model, Model, _cfg_):
    dup_model = get_model(Model=Model, _cfg_=_cfg_)
    dup_model.to(_cfg_.device)
    dup_model.load_state_dict(model.state_dict())
    return dup_model


def fast_learn(model, sup_criterion, fast_opt, _cfg_, teacher_model=None, unsup_criterion=None, X=None, Y=None,
               mask=None):
    model.train()
    fast_opt.zero_grad()
    outputs = model(ag_x=X[4], ab_x=X[0], attention_mask_ab_v=X[1],
                    ab_l=X[2], attention_mask_ab_l=X[3])
    intermediate_loss = sup_criterion(outputs[0], Y.view(-1, 1).float())

    if mask is None:
        loss = torch.mean(intermediate_loss)
    else:
        supervised_loss = mask * intermediate_loss
        inverse_mask = torch.ones_like(mask).to(_cfg_.device) - mask

        teacher_model.eval()
        teacher_outputs = teacher_model(ag_x=X[4], ab_x=X[5], attention_mask_ab_v=X[1],
                                        ab_l=X[6], attention_mask_ab_l=X[3])
        teacher_model.train()

        unsupervised_loss = inverse_mask * unsup_criterion(outputs[0], teacher_outputs[0])

        loss = torch.mean(supervised_loss + _cfg_.unsup_loss_weight * unsupervised_loss)

    loss.backward()
    fast_opt.step()

    return loss

