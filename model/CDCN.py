import torch
from torch import nn
device = torch.device('cuda:6')


class Residual(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Residual, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm1d(ch_out)
        )

    def forward(self, x):
        return self.residual(x)


class Dilated(nn.Module):
    def __init__(self, ch_in, ch_out, k):
        super(Dilated, self).__init__()
        self.d = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=6, dilation=k, padding='same'),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=6, dilation=k, padding='same'),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.d(x)


class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y
        x = x.view(x.size(0), -1)
        return x


class CDCN(nn.Module):
    def __init__(self, in_fea_num, out_fea_num):
        super(CDCN, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=1),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True)
        )
        self.r1 = Residual(4, 32)
        self.d1 = Dilated(4, 32, 1)
        self.r2 = Residual(32, 32)
        self.d2 = Dilated(32, 32, 2)
        self.r3 = Residual(32, 32)
        self.d3 = Dilated(32, 32, 4)
        self.r4 = Residual(32, 32)
        self.d4 = Dilated(32, 32, 8)
        self.attention = se_block(128)
        # indian_pines=25600
        # paciau=13184
        # black=22528
        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(22528, out_fea_num)
        )

    def forward(self, x, x1):
        n, c, w, h = x.shape
        new_x = x[:, :, w//2, h//2]
        new_x = new_x.unsqueeze(1)
        # 128, 1, 220
        x0 = self.b1(new_x)
        # 128, 32, 220
        xr1 = self.r1(x0)
        # 128, 32, 220
        xd1 = self.d1(x0)
        # 128, 32, 220
        xdr1 = xr1 + xd1
        # 128, 32, 220
        xr2 = self.r2(xdr1)
        # 128, 32, 220
        xd2 = self.d2(xdr1)
        # 128, 32, 220
        xdr2 = xr2 + xd2
        # 128, 32, 220
        xr3 = self.r3(xdr2)
        # 128, 32, 220
        xd3 = self.d3(xdr2)
        # 128, 32, 220
        xdr3 = xr3 + xd3
        # 128, 32, 220
        xr4 = self.r4(xdr3)
        # 128, 32, 220
        xd4 = self.d4(xdr3)
        # 128, 32, 220
        xdr4 = xr4 + xd4
        # 128, 128, 220
        x = torch.cat([xd1, xd2, xd3, xdr4], dim=1)
        x0 = self.attention(x)
        x1 = self.final(x0)
        return x0, x1


def main():
    tmp = torch.randn(128, 176, 25, 25)
    net = CDCN(200, 9)
    out1, out2 = net(tmp, tmp)
    print(out1.shape, out2.shape)


if __name__ == '__main__':
    main()
