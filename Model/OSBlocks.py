import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Standard block:
        Conv -> BN -> Activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    """
    1x1 Conv -> BN -> Activation
    """

    def __init__(self, in_channels, out_channels, stride=1, act_layer=nn.ReLU):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )


class Conv1x1Linear(nn.Module):
    """
    1x1 Conv -> BN
    No activation.
    Useful for projection and expansion.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """
    OSNet Lite 3x3:
        1x1 pointwise conv
        3x3 depthwise conv
    """

    def __init__(self, in_channels, out_channels, stride=1, act_layer=nn.ReLU):
        super().__init__()

        self.pw = Conv1x1(in_channels, out_channels, act_layer=act_layer)

        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels,   # depthwise
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(out_channels)
        self.dw_act = nn.SiLU()

    def forward(self, x):
        x = self.pw(x)
        x = self.dw(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        return x


class ChannelGate(nn.Module):
    """
    Unified aggregation gate:
        GAP -> 1x1 -> SiLU -> 1x1 -> Sigmoid

    Produces channel-wise weights for each stream output.
    """

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU):
        super().__init__()

        hidden_channels = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return w


class OSBlock(nn.Module):
    """
    Omni-Scale block.

    Structure:
        1x1 reduction
        4 parallel streams:
            stream1: 1 LiteConv3x3
            stream2: 2 LiteConv3x3
            stream3: 3 LiteConv3x3
            stream4: 4 LiteConv3x3
        shared channel gate
        sum gated outputs
        1x1 expansion
        residual add
        activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_reduction=4,
        gate_reduction=16
    ):
        super().__init__()

        mid_channels = out_channels // bottleneck_reduction
        if mid_channels < 1:
            raise ValueError("mid_channels became < 1. Increase channel width.")

        self.conv1 = Conv1x1(in_channels, mid_channels)

        self.stream1 = LightConv3x3(mid_channels, mid_channels)

        self.stream2 = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        self.stream3 = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        self.stream4 = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        # Shared gate across all streams
        self.gate = ChannelGate(mid_channels, reduction=gate_reduction)

        self.conv3 = Conv1x1Linear(mid_channels, out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

        self.act = nn.SiLU()

    def _apply_gate(self, x):
        return self.gate(x) * x

    def forward(self, x):
        identity = x

        x = self.conv1(x)

        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)

        x = (
            self._apply_gate(s1) +
            self._apply_gate(s2) +
            self._apply_gate(s3) +
            self._apply_gate(s4)
        )

        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity
        x = self.act(x)
        return x


class TransitionBlock(nn.Module):
    """
    OSNet transition block:
        1x1 conv
        2x2 average pooling (stride 2)
    """

    def __init__(self, in_channels, out_channels, act_layer=nn.ReLU):
        super().__init__()

        self.conv = Conv1x1(in_channels, out_channels, act_layer=act_layer)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x