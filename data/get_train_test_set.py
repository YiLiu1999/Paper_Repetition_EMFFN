import torch
import scipy.io as io
import Paper_Repetition.EMFFNHSI.data.data_preprocess as pre_fun


def get_train_test_set(cfg):
    # 从cfg中导入设定好的参数
    data_path = cfg['data_path']
    image_name = cfg['image_name']
    gt_name = cfg['gt_name']
    train_set_num = cfg['train_set_num']
    patch_size = cfg['patch_size']

    # 加载数据
    data = io.loadmat(data_path)

    img = data[image_name].astype('float32')
    gt = data[gt_name].astype('float32')
    img = torch.from_numpy(img)
    gt = torch.from_numpy(gt)
    img = img.permute(2, 0, 1)
    img = pre_fun.std_norm(img)

    # label_transform
    img_gt = pre_fun.label_transform(gt)
    # construct_sample:切分patch，储存每个patch的坐标值
    img_pad, img_pad_indices = pre_fun.construct_sample(img, patch_size)
    # select_sample:用img_gt的标签信息划分样本
    data_sample = pre_fun.select_sample(img_gt, train_set_num)

    data_sample['pad_img'] = img_pad
    data_sample['pad_img_indices'] = img_pad_indices
    data_sample['img_gt'] = img_gt
    data_sample['ori_gt'] = gt

    if cfg['pca'] > 0:
        img_pca = pre_fun.extract_pc(img, cfg['pca'])
        img_pca = pre_fun.one_zero_norm(img_pca)
        img_pca = pre_fun.std_norm(img_pca)

        img_pca_pad, _ = pre_fun.construct_sample(img_pca, patch_size)
        data_sample['img_pca_pad'] = img_pca_pad

    return data_sample
