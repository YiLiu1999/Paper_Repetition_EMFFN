import torch


def Predict_Label2Img(predict_label):
    predict_img = torch.zeros([512, 614])
    num = predict_label.shape[0]

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = int(predict_label[i][3])
        predict_img[x][y] = l

    return predict_img
