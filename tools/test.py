import torch
from Paper_Repetition.EMFFNHSI.data.HSI_data import batch_collate as collate_fn
from torch.utils.data import DataLoader
import Paper_Repetition.EMFFNHSI.data.data_preprocess as pre_fun
import visdom
viz = visdom.Visdom(env='Liuy')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}', format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'

    return True


def remove_prefix(state_dict, prefix):
    # print('remove prefix \'{}\''.format(prefix))

    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))

    if load_to_cpu == torch.device('cpu'):
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']

    else:
        device = torch.device('cuda:0')
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))['model']

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def test(test_data, origin_gt, model, device, cfg):
    num_workers = cfg['workers_num']
    gpu_num = cfg['gpu_num']
    batch_size = cfg['batch_size']

    model = load_model(model, cfg['model_weights'], device)
    model.eval()
    model = model.to(device)

    if gpu_num > 1 and cfg['gpu_train']:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    batch_data = DataLoader(test_data, batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=True)

    predict_correct = 0
    label_num = 0
    predict_label = []


    for batch_idx, batch_sample in enumerate(batch_data):

        if len(batch_sample) > 3:
            img, target, indices, img_pca = batch_sample
            img_pca = img_pca.to(device)
        else:
            img, target, indices = batch_sample

        img = img.to(device)

        with torch.no_grad():
            joint, spatial, spectral = model(img, img_pca)

        joint = spectral

        label = joint.softmax(dim=1).cpu().argmax(dim=1, keepdim=True)

        if target.sum() > 0:
            predict_correct += label.eq(target.view_as(label)).sum().item()
            label_num += len(target)
        if label_num > 0:

            acc = predict_correct / label_num
            viz.line(
                Y=[acc],
                X=[batch_idx],
                win="line2",
                opts={
                    'showlegend': True,  # 显示网格
                    'title': "acc in test",
                    'xlabel': "x",  # x轴标签
                    'ylabel': "y",  # y轴标签
                    'fillarea': False,
                },
                update='append',
            )

        label = pre_fun.label_inverse_transform(label, origin_gt.long())
        predict_label.append(torch.cat([indices, label], dim=1))

    predict_label = torch.cat(predict_label, dim=0)
    # print(len(test_data))

    # indian_pines=1765,8484
    # paciau=1350, 41426
    # KSC=459,4752
    if len(test_data) == 459:
        print('Train_data:0A {:.2f}%'.format(100 * predict_correct / label_num))
    elif len(test_data) == 4752:
        print('Test_data:0A {:.2f}%'.format(100 * predict_correct / label_num))

    return predict_label
