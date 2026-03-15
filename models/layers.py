import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False, groups=1):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class LWN(nn.Module):
    """Learnable Wavelet Node (LWN) 透過自適應群組卷積模擬 2D-DWT"""
    def __init__(self, channels):
        super(LWN, self).__init__()
        self.dwt_conv = BasicConv(channels, channels * 4, kernel_size=3, stride=2, relu=False, groups=channels)
        self.idwt_conv = BasicConv(channels * 4, channels, kernel_size=4, stride=2, relu=False, transpose=True, groups=channels)

    def forward(self, x):
        freqs = self.dwt_conv(x)
        reconstructed = self.idwt_conv(freqs)
        return reconstructed, freqs

class SEB(nn.Module):
    """Simple Enhancement Block (SEB)"""
    def __init__(self, channel):
        super(SEB, self).__init__()
        self.conv1 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class WFB(nn.Module):
    """Wavelet Feature Block (WFB) 結合小波變換與空間特徵增強
    
    forward 回傳兩個值：
      - down : dwt_conv 輸出的降採樣特徵 (channels*4, H/2, W/2)，
               作為下一個 encoder scale 的輸入，取代原本多餘的 MaxPool。
      - freqs: 同 down（即小波係數），供 Wavelet Loss 計算用。
    
    LWN 的 idwt_conv 仍保留，用於殘差增強 (res + x)，
    再由 SEB 進一步強化後存回 skip connection (self.skip)。
    skip 會在 MLWNet.forward 中被 decoder 的對應層加回。
    """
    def __init__(self, channel):
        super(WFB, self).__init__()
        self.lwn = LWN(channel)
        self.seb = SEB(channel)

    def forward(self, x):
        res, freqs = self.lwn(x)          # res: (C, H, W)  freqs: (C*4, H/2, W/2)
        skip = self.seb(res + x)          # skip: (C, H, W) — 給 decoder skip connection
        return freqs, skip                # freqs 作為下一層輸入（已降採樣），skip 供 decoder 用