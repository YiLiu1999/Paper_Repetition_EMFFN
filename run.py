import torch
import torch.nn as nn
import os
import scipy.io as io
import configs.configs as cfg
import torch.optim as optim

from data.HSI_data import HSIdata as fun_data
from data.get_train_test_set import get_train_test_set as fun_get_set
from tools.train import train as fun_train
from tools.test import test as fun_test
from tools.show import Predict_Label2Img
from matplotlib import pyplot as plt
from Paper_Repetition.EMFFNHSI.model.PMN import PMN
from Paper_Repetition.EMFFNHSI.model.CDCN import CDCN
from model.EMFFN import EMFFN as fun_model


def main():
    cfg_data = cfg.data
    cfg_model = cfg.model
    cfg_train = cfg.train['train_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test

    data_sets = fun_get_set(cfg_data)
    # print(data_sets.keys())
    # print(data_sets['img_pca_pad'])
    # print('img_pca_pad' in data_sets)
    train_data = fun_data(data_sets, cfg_data['train_data'])
    test_data = fun_data(data_sets, cfg_data['test_data'])
    no_gt_data = fun_data(data_sets, cfg_data['no_gt_data'])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:3' if use_cuda else 'cpu')
    model = fun_model(cfg_model['inCDCN_fea_num'], cfg_model['inPMN_fea_num'], cfg_model['out_fea_num']).to(device)
    model0 = CDCN(cfg_model['inCDCN_fea_num'], cfg_model['out_fea_num']).to(device)
    model1 = PMN(cfg_model['inPMN_fea_num'], cfg_model['out_fea_num']).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=cfg_optim['lr'],
                          weight_decay=cfg_optim['weight_decay'])

    fun_train(train_data, model, loss_fun, optimizer, device, cfg_train)
    pred_train_label = fun_test(train_data, data_sets['ori_gt'], model, device, cfg_test)  # torch.Size([270, 4])
    pred_test_label = fun_test(test_data, data_sets['ori_gt'], model, device, cfg_test)  # torch.Size([42506, 4])
    pred_no_gt_label = fun_test(no_gt_data, data_sets['ori_gt'], model, device, cfg_test)  # torch.Size([164624, 4])
    predict_label = torch.cat([pred_train_label, pred_test_label, pred_no_gt_label], dim=0)  # torch.Size([207400, 4])
    # predict_label = pred_test_label
    HSI = Predict_Label2Img(predict_label)
    plt.imshow(HSI)
    plt.show()
    save_folder = cfg_test['save_folder']
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    io.savemat(save_folder + '/classification_label.mat', {'predict_label_CNN2D': predict_label})


if __name__ == '__main__':
    main()
