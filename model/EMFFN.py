import torch
from torch import nn

from Paper_Repetition.EMFFNHSI.model.PMN import PMN
from Paper_Repetition.EMFFNHSI.model.CDCN import CDCN

# indian_pines=27184
# paciau=14768
# black=24112



class EMFFN(nn.Module):
    def __init__(self, inCDCN_fea_num, inPMN_fea_num, out_fea_num):
        super(EMFFN, self).__init__()
        self.cdcn = CDCN(inCDCN_fea_num, out_fea_num)
        self.pmn = PMN(inPMN_fea_num, out_fea_num)
        self.linear = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(24112, 30000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(30000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_fea_num),

        )
        for m in self.modules():
            for name, parameter in m.named_parameters():
                if parameter.dim() > 1:
                    # nn.init.normal_(parameter, mean=0, std=0.1)
                    # nn.init.kaiming_uniform_(parameter, mode='fan_out', nonlinearity='relu')
                    nn.init.kaiming_normal_(parameter, mode='fan_out', nonlinearity='relu')
                elif parameter.dim() == 1:
                    if name.split('.')[-1] == 'weight':
                        # nn.init.normal_(parameter, mean=0, std=0.1)
                        nn.init.constant_(parameter, 0.1)
                    elif name.split('.')[-1] == 'bias':
                        nn.init.constant_(parameter, 0)
            break

    def forward(self, a1, a2):
        # torch.Size([32, 186624])
        # print(a1.is_cuda)
        c1, c2 = self.cdcn(a1, a1)
        # print(a1.is_cuda)
        # torch.Size([32, 1584])
        # print(a2.is_cuda)
        p1, p2 = self.pmn(a2, a2)
        # print(a2.is_cuda)
        a = torch.cat([c1, p1], dim=1)
        a = self.linear(a)
        return a, c2, p2


def main():
    a1 = torch.randn(128, 176, 25, 25)
    a2 = torch.randn(128, 5, 25, 25)
    net = EMFFN(200, 5, 9)
    out0, out1, out2 = net(a1, a2)
    print(out0.shape, out1.shape, out2.shape)


if __name__ == '__main__':
    main()
