import torch
from torch import nn
from torch.nn import functional as F
device = torch.device('cuda:3')


class basic(nn.Module):
    def __init__(self, in_fea_num, out_fea_num, k, s, p):
        super(basic, self).__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_fea_num, out_fea_num, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_fea_num),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.b(x)


class PMN(nn.Module):
    def __init__(self, inPMN_fea_num, out_fea_num):
        super(PMN, self).__init__()
        self.b11 = basic(inPMN_fea_num, 16, 7, 1, 3)
        self.b12 = basic(inPMN_fea_num, 32, 5, 1, 2)
        self.b13 = basic(inPMN_fea_num, 64, 3, 1, 1)
        self.b21 = basic(16, 16, 7, 1, 3)
        self.b22 = basic(48, 32, 5, 1, 2)
        self.b23 = basic(112, 64, 3, 1, 1)
        self.b31 = basic(16, 16, 7, 1, 3)
        self.b32 = basic(48, 32, 5, 1, 2)
        self.b33 = basic(112, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Flatten()
        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(1584, out_fea_num)
        )

    def forward(self, x0, x):
        s11 = self.b11(x)
        h11 = self.pool(s11)
        s12 = self.b12(x)
        c12 = torch.cat([s11, s12], dim=1)
        h12 = self.pool(c12)
        s13 = self.b13(x)
        c13 = torch.cat([c12, s13], dim=1)
        h13 = self.pool(c13)
        s21 = self.b21(h11)
        h21 = self.pool(s21)
        s22 = self.b22(h12)
        c22 = torch.cat([s21, s22], dim=1)
        h22 = self.pool(c22)
        s23 = self.b23(h13)
        c23 = torch.cat([c22, s23], dim=1)
        h23 = self.pool(c23)
        s31 = self.b31(h21)
        h31 = self.pool(s31)
        s32 = self.b32(h22)
        c32 = torch.cat([s31, s32], dim=1)
        h32 = self.pool(c32)
        s33 = self.b33(h23)
        c33 = torch.cat([c32, s33], dim=1)
        h33 = self.pool(c33)
        c1 = self.fc(h31)
        c2 = self.fc(h32)
        c3 = self.fc(h33)
        a0 = torch.cat([c1, c2, c3], dim=1)
        a1 = torch.sigmoid(F.relu(a0))
        a2 = a0.mul(a0)
        a3 = self.final(a2)

        return a2, a3


def main():
    tmp = torch.randn(32, 5, 25, 25)
    net = PMN(5, 16)
    out1, out2 = net(tmp, tmp)
    print(out1.shape, out2.shape)


if __name__ == '__main__':
    main()
