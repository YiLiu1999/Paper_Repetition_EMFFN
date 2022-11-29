import torch
import time
import datetime
import math
import os
import visdom
import numpy as np
from Paper_Repetition.EMFFNHSI.data.HSI_data import batch_collate as collate_fn
from torch.utils.data import DataLoader
viz = visdom.Visdom(env='Liuy_MSCNN')


# 学习率调整策略
def adjust_lr(lr_init, lr_gamma, optimizer, epoch, step_index):
    if epoch < 1:
        lr = 0.0001 * lr_init
    else:
        lr = lr_init * lr_gamma ** ((epoch - 1) // step_index)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# 训练
def train(train_data, model, loss_fun, optimizer, device, cfg):
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    save_folder = cfg['save_folder']
    save_name = cfg['save_name']
    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']
    epoch_size = cfg['epoch']
    batch_size = cfg['batch_size']

    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    model.train()

    # 是否使用上次训练的模型
    if cfg['reuse_model']:
        print('loading model,please wait a moment')
        print('Load complete')
        checkpoint = torch.load(cfg['reuse_file'], map_location=device)

        start_epoch = checkpoint['epoch']  # 10
        # 浅拷贝 获取模型中的所有参数
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint['model'].items()
                           if k in model_dict}

        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)
    else:
        start_epoch = 0

    batch_num = math.ceil(len(train_data) / batch_size)
    print('start training...')
    for epoch in range(start_epoch + 1, epoch_size + 1):

        epoch_time0 = time.time()

        batch_data = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=True)
        if lr_adjust:
            lr = adjust_lr(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        epoch_loss = 0
        predict_correct = 0
        label_num = 0

        for batch_idx, batch_sample in enumerate(batch_data):
            iteration = (epoch - 1) * batch_num + batch_idx + 1
            batch_time0 = time.time()

            if len(batch_sample) > 3:
                img, target, indices, img_pca = batch_sample
                img_pca = img_pca.to(device)
            else:
                img, target, indices = batch_sample

            img = img.to(device)
            target = target.to(device)
            joint, spatial, spectral = model(img, img_pca)

            loss_joint = loss_fun(joint, target.long())
            loss_spatial = loss_fun(spatial, target.long())
            loss_spectral = loss_fun(spectral, target.long())
            loss = loss_spectral+loss_spatial+loss_joint
            optimizer.zero_grad()
            # loss_spectral.backward()
            # loss_spatial.backward()
            loss.backward()
            optimizer.step()

            viz.line(
                Y=[loss.item()],
                X=[iteration/100],
                win="line",
                opts={
                    'showlegend': True,  # 显示网格
                    'title': "loss in train",
                    'xlabel': "x",  # x轴标签
                    'ylabel': "y",  # y轴标签
                    'fillarea': False,
                },
                update='append',
            )

            batch_time1 = time.time()
            batch_time = batch_time1 - batch_time0

            batch_eta = batch_time * (batch_num - batch_idx)
            epoch_eta = int(batch_time * (epoch_size - epoch) * batch_num + batch_eta)

            epoch_loss += loss.item()

            predict_label = joint.detach().softmax(dim=-1).argmax(dim=1, keepdim=True)
            predict_correct += predict_label.eq(target.view_as(predict_label)).sum().item()
            label_num += len(target)

        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        print('Epoch:{}/{} || lr:{} || loss:{} || Train acc:{:.2f}% ||'
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss/batch_num, 100*predict_correct/label_num,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta))
                      )
              )

        viz.line(
            Y=[predict_correct/label_num],
            X=[epoch],
            win='line1',
            opts={
                'showlegend': True,  # 显示网格
                'title': "acc in train",
                'xlabel': "x",  # x轴标签
                'ylabel': "y",  # y轴标签
                'fillarea': False,
            },
            update='append',
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + '_Final.pth'))
