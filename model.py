import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, base=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(input_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = nn.Sequential(nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, True))
        self.e3 = nn.Sequential(nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, True))
        self.e4 = nn.Sequential(nn.Conv2d(base*4, base*8, 4, 2, 1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, True))
        self.e5 = nn.Sequential(nn.Conv2d(base*8, base*8, 4, 2, 1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, True))
        self.e6 = nn.Sequential(nn.Conv2d(base*8, base*8, 4, 2, 1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, True))
        self.e7 = nn.Sequential(nn.Conv2d(base*8, base*8, 4, 2, 1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, True))
        self.e8 = nn.Sequential(nn.Conv2d(base*8, base*8, 4, 2, 1))

        def up(in_c, out_c, dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.d1 = up(base*8,   base*8, dropout=True)
        self.d2 = up(base*8*2, base*8, dropout=True)
        self.d3 = up(base*8*2, base*8, dropout=True)
        self.d4 = up(base*8*2, base*8)
        self.d5 = up(base*8*2, base*4)
        self.d6 = up(base*4*2, base*2)
        self.d7 = up(base*2*2, base)
        self.d8 = nn.Sequential(nn.ConvTranspose2d(base*2, output_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6); e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        out = self.d8(torch.cat([d7, e1], 1))
        return out
