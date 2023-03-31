import torch.nn as nn
from torchvision import models
import torch

import torch
import torch.nn as nn


class MFFAUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=32):
        super(MFFAUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.DAconv1 = self._DAconv(in_ch, base_ch * 2, repeat_time=2)
        self.DAconv2 = self._DAconv(base_ch * 2 + 3, base_ch * 4, repeat_time=2)
        self.DAconv3 = self._DAconv(base_ch * 6 + 3, base_ch * 8, repeat_time=3)
        self.DAconv4 = self._DAconv(base_ch * 14 + 3, base_ch * 16, repeat_time=3)
        self.DAconv5 = self._DAconv(base_ch * 30 + 3, base_ch * 32, repeat_time=3)

        self.DAconv2u = self._DAconv(base_ch * 2, base_ch * 4, repeat_time=2)
        self.upConv2u = self._UpConv(base_ch * 4, base_ch * 2)

        self.upConv5 = self._UpConv(base_ch * 32, base_ch * 16)
        self.attBlock1 = self._AttBlock(base_ch * 16, base_ch * 16, base_ch * 8)
        self.up5 = self._DAconv(base_ch * 32, base_ch * 16, repeat_time=3)

        self.upConv4 = self._UpConv(base_ch * 16, base_ch * 8)
        self.attBlock2 = self._AttBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up4 = self._DAconv(base_ch * 16, base_ch * 8, repeat_time=3)

        self.upConv3 = self._UpConv(base_ch * 8, base_ch * 4)
        self.attBlock3 = self._AttBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up3 = self._DAconv(base_ch * 8, base_ch * 4, repeat_time=3)

        self.upConv2 = self._UpConv(base_ch * 4, base_ch * 2)
        self.attBlock4 = self._AttBlock(base_ch * 2, base_ch * 2, base_ch * 1)
        self.up2 = self._DAconv(base_ch * 4, base_ch * 2, repeat_time=2)

        self.out = nn.Conv2d(base_ch * 2, out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # x0 = self.inc(x)
        x1 = self.DAconv1(x)

        x01 = self.Maxpool(x)
        x20 = self.Maxpool(x1)

        x2u = self.DAconv2u(x20)
        d2u = self.upConv2u(x2u)

        x22 = torch.cat((x01, x20), dim=1)
        x2 = self.DAconv2(x22)

        x02 = self.Maxpool(x22)
        x31 = self.Maxpool(x2)
        x32 = torch.cat((x02, x31), dim=1)
        x3 = self.DAconv3(x32)

        x03 = self.Maxpool(x32)
        x41 = self.Maxpool(x3)
        x42 = torch.cat((x03, x41), dim=1)
        x4 = self.DAconv4(x42)

        x04 = self.Maxpool(x42)
        x51 = self.Maxpool(x4)
        x52 = torch.cat((x04, x51), dim=1)
        x5 = self.DAconv5(x52)

        d51 = self.upConv5(x5)
        x4 = self.attBlock1(d51, x4)
        d5 = torch.cat((x4, d51), dim=1)
        d5 = self.up5(d5) + d51

        d41 = self.upConv4(d5)
        x3 = self.attBlock2(d41, x3)
        d4 = torch.cat((x3, d41), dim=1)
        d4 = self.up4(d4) + d41

        d31 = self.upConv3(d4)
        x2 = self.attBlock3(d31, x2)
        d3 = torch.cat((x2, d31), dim=1)
        d3 = self.up3(d3) + d31

        d21 = self.upConv2(d3)
        x1 = self.attBlock4(d21, x1)
        d2 = torch.cat((x1, d21), dim=1)
        d2 = self.up2(d2) + d21 + d2u

        x = self.out(d2)
        return x

    class _DAconv(nn.Module):
        def __init__(self, ch_in, ch_out, repeat_time):
            super(MFFAUNet._DAconv, self).__init__()

            self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
            self.dc = MFFAUNet._Conv(ch_out, ch_out, repeat_time)

        def forward(self, x):
            x = self.Conv_1x1(x)
            x1 = self.dc(x)
            return x + x1

    class _UpConv(nn.Module):
        def __init__(self, ch_in, ch_out):
            super(MFFAUNet._UpConv, self).__init__()
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.up(x)
            return x

    # class _Conv(nn.Module):
    #     def __init__(self, in_ch, out_ch, repeat_time):
    #         super(DAUNet._Conv, self).__init__()
    #
    #         self.conv = nn.Sequential(
    #             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
    #             nn.BatchNorm2d(out_ch),
    #             nn.ReLU()
    #         )
    #
    #         self.conv2 = nn.Sequential(
    #             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
    #             nn.BatchNorm2d(out_ch),
    #             nn.ReLU()
    #         )
    #
    #         self.conv3 = nn.Sequential(
    #             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
    #             nn.BatchNorm2d(out_ch),
    #             nn.ReLU()
    #         )
    #
    #         self.repeat_time = repeat_time
    #
    #     def forward(self, x):
    #         # x1 = self.inc(x)
    #         x1 = self.conv(x) + x
    #         x2 = self.conv2(x1) + x1 + x
    #         x3 = self.conv3(x2) + x2 + x1 + x
    #
    #         return x3

    class _Conv(nn.Module):
        def __init__(self, in_ch, out_ch, repeat_time):
            super(MFFAUNet._Conv, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

            self.repeat_time = repeat_time

        def forward(self, x):
            # x1 = self.inc(x)
            x1 = self.conv1(x)

            return x1

    class _AttBlock(nn.Module):
        def __init__(self, ing_ch, inl_ch, out_ch):
            super(MFFAUNet._AttBlock, self).__init__()

            self.Wg = nn.Sequential(
                nn.Conv2d(ing_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch), )

            self.Wl = nn.Sequential(
                nn.Conv2d(inl_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch), )

            self.psi = nn.Sequential(
                nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x1, x2):
            g1 = self.Wg(x1)

            x1 = self.Wl(x2)

            psi = self.relu(g1 + x1)
            psi = self.psi(psi)

            return x2 * psi


if __name__ == '__main__':
    from torchsummary import summary

    net = MFFAUNet(in_ch=3, out_ch=1).cuda()
    summary(net, (3, 256, 256))



