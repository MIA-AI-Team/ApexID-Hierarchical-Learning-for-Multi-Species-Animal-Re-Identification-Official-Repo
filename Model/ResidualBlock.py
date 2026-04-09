import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        """Initialize a convolution followed by batch normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Convolution kernel size.
            stride (int or tuple): Convolution stride.
            padding (int or tuple): Padding applied to the convolution.
            groups (int, optional): Number of blocked channel groups.
            bias (bool, optional): Whether the convolution uses a bias term.

        Returns:
            None
        """
        super().__init__()
        self.c = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, 
                           stride = stride, padding=padding, bias = bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels) 
    def forward(self, x):
        """Apply the convolution block to an input feature map.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Batch-normalized convolution output.
        """
        return self.bn(self.c(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cardinality=32):
        """Initialize a grouped residual bottleneck block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride used in the spatial convolution branch.
            first (bool, optional): Whether this is the first block in a stage and
                should project the residual path.
            cardinality (int, optional): Number of groups used in the middle
                convolution.

        Returns:
            None
        """
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
        """Run the residual block on an input tensor.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Output tensor after the residual addition and activation.
        """

        f = self.silu(self.c1(x))
        f = self.silu(self.c2(f))
        f = self.c3(f)

        if self.downsample:
            x = self.p(x)
        
        h = self.silu(torch.add(f, x))

        return h
