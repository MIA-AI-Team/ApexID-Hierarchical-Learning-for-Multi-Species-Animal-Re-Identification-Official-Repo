import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.c = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, 
                           stride = stride, padding=padding, bias = bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels) 
    def forward(self, x):
        return self.bn(self.c(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cardinality=32):
        super().__init__()
        self.c = cardinality
        self.downsample = stride==2 or first
        res_channels = out_channels // 2
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)
        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, self.c)
        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)

        self.silu = nn.SiLU()

        if self.downsample:
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):

        f = self.silu(self.c1(x))
        f = self.silu(self.c2(f))
        f = self.c3(f)

        if self.downsample:
            x = self.p(x)
        
        h = self.silu(torch.add(f, x))

        return h