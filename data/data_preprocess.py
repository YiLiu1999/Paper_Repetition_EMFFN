import torch
import torch.nn as nn
from torchvision import transforms


def extract_pc(image, pc=3):
    channel, height, width = image.shape  # input float tensor image with CxHxW
    # data = image.view(channel, height*width)
    data = image.contiguous().view(channel, height * width)
    data_c = data - data.mean(dim=1).unsqueeze(1)
    u, s, vt = torch.svd(data_c.matmul(data_c.T))
    sorted_data, indices = s.sort(descending=True)

    image_pc = u[:, indices[0:pc]].T.matmul(data)

    return image_pc.view(pc, height, width)


def std_norm(image):
    image = image.permute(1, 2, 0).numpy()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(torch.tensor(image).mean(dim=[0, 1]), torch.tensor(image).std(dim=[0, 1]))
    ])
    return trans(image)


def one_zero_norm(image):
    channel, height, width = image.shape
    data = image.view(channel, height*width)
    data_max = data.max(dim=1)[0]
    data_min = data.min(dim=1)[0]
    data = (data-data_min.unsqueeze(1))/(data_max.unsqueeze(1) - data_min.unsqueeze(1))
    return data.view(channel, height, width)


def construct_sample(image, windows=27):
    _, height, width = image.shape
    half_window = int(windows/2)
    pad = nn.ReplicationPad2d(half_window)
    pad_image = pad(image.unsqueeze(0)).squeeze(0)
    batch_image_indices = torch.zeros((height*width, 4), dtype=torch.long)
    t = 0
    for h in range(height):
        for w in range(width):
            batch_image_indices[t, :] = torch.tensor([h, h+windows, w, w+windows])
            t += 1
    return pad_image, batch_image_indices


def label_transform(gt):
    label = torch.unique(gt)
    gt_new = torch.zeros_like(gt)

    for each in range(len(label)):
        indices = torch.where(gt == label[each])
        if label[0] == 0:
            gt_new[indices] = each - 1
        else:
            gt_new[indices] = each
    label_new = torch.unique(gt_new)
    return gt_new


def label_inverse_transform(predict_result, gt):
    label_origin = torch.unique(gt)
    label_predict = torch.unique(predict_result)
    predict_result_origin = torch.zeros_like(predict_result)
    for each in range(len(label_predict)):
        indices = torch.where(predict_result == label_predict[each])
        if len(label_predict) != len(label_origin):
            predict_result_origin[indices] = label_origin[each + 1]
        else:
            predict_result_origin[indices] = label_origin[each]
    return predict_result_origin


def select_sample(gt, ntr):
    gt_vector = gt.reshape(-1, 1).squeeze(1)
    label = torch.unique(gt)

    for each in range(len(label)):
        indices_vector = torch.where(gt_vector == label[each])  # 1 tuple 返回一维的索引
        indices = torch.where(gt == label[each])  # 2 tuple 返回二维的索引

        indices_vector = indices_vector[0]
        indices_row = indices[0]
        indices_column = indices[1]

        # 背景 -1
        # 10776
        if label[each] == -1:
            no_gt_indices = torch.cat([indices_vector.unsqueeze(1),
                                       indices_row.unsqueeze(1),
                                       indices_column.unsqueeze(1)],
                                      dim=1
                                      )
            no_gt_num = torch.tensor(len(indices_vector))
        else:
            class_num = torch.tensor(len(indices_vector))
            rand_indices0 = torch.randperm(class_num)
            rand_indices = indices_vector[rand_indices0]
            # indian_pines
            # num_workers = [30, 150, 150, 100, 150, 150, 20, 150, 15, 150, 150, 150, 150, 150, 50, 50]
            # paviau
            # num_workers = [150, 150, 150, 150, 150, 150, 150, 150, 150]
            # KSC
            num_workers = [33, 23, 24, 24, 15, 22, 9, 38, 51, 39, 41, 49, 91]
            sel_num = num_workers[each - 1]

            sel_num = torch.tensor(sel_num)
            tr_ind0 = rand_indices0[0:sel_num]
            te_ind0 = rand_indices0[sel_num:]

            tr_ind = rand_indices[0:sel_num]
            te_ind = rand_indices[sel_num:]

            # 训练集:索引+坐标
            sel_tr_ind = torch.cat([
                tr_ind.unsqueeze(1),
                indices_row[tr_ind0].unsqueeze(1),
                indices_column[tr_ind0].unsqueeze(1)
            ], dim=1)
            # 测试集：
            sel_te_ind = torch.cat([
                te_ind.unsqueeze(1),
                indices_row[te_ind0].unsqueeze(1),
                indices_column[te_ind0].unsqueeze(1)
            ], dim=1)

            if each == 1:
                train_indices = sel_tr_ind
                train_num = sel_num.unsqueeze(0)
                test_indices = sel_te_ind
                test_num = (class_num - sel_num).unsqueeze(0)
            else:
                train_indices = torch.cat([train_indices, sel_tr_ind], dim=0)
                train_num = torch.cat([train_num, sel_num.unsqueeze(0)])

                test_indices = torch.cat([test_indices, sel_te_ind], dim=0)
                test_num = torch.cat([test_num, (class_num - sel_num).unsqueeze(0)])

    # 训练集
    rand_tr_ind = torch.randperm(train_num.sum())
    train_indices = train_indices[rand_tr_ind, ]
    # 测试集
    rand_te_ind = torch.randperm(test_num.sum())  # torch.Size([42506])
    test_indices = test_indices[rand_te_ind, ]
    # 背景图
    rand_no_gt_ind = torch.randperm(no_gt_num.sum())
    no_gt_indices = no_gt_indices[rand_no_gt_ind, ]

    data_sample = {'train_indices': train_indices, 'train_num': train_num,
                   'test_indices': test_indices, 'test_num': test_num,
                   'no_gt_indices': no_gt_indices, 'no_gt_num': no_gt_num.unsqueeze(0)}

    return data_sample
