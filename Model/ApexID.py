import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from .OSBlocks import ConvLayer, Conv1x1, OSBlock, TransitionBlock
except ImportError:  
    from OSBlocks import ConvLayer, Conv1x1, OSBlock, TransitionBlock

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
class ApexID(nn.Module):
    """
    Args:
        config_name (str):
            OSNet width variant:
                - "osnet_x1_0"
                - "osnet_x0_75"
                - "osnet_x0_5"
                - "osnet_x0_25"

            If an old ResNet-like depth (50/101/152) is passed,
            it falls back to "osnet_x1_0" for compatibility.
        in_channels (int):
            Number of input channels. Can be 3 for RGB or 4 if using FFT.
        embedding_dim (int):
            Output embedding dimension.
    """

    def __init__(
        self,
        config_name="osnet_x1_0",
        in_channels=3,
        embedding_dim=512
    ):
        super().__init__()

        width_mult = self._resolve_width(config_name)

        # Standard OSNet widths scaled by width multiplier
        # Paper base layout roughly uses 64, 256, 384, 512 channels.
        c1 = self._make_channels(64, width_mult)
        c2 = self._make_channels(256, width_mult)
        c3 = self._make_channels(384, width_mult)
        c4 = self._make_channels(512, width_mult)

        self.conv1 = ConvLayer(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2
        self.conv2 = self._make_stage(c1, c2, num_blocks=2)
        self.transition1 = TransitionBlock(c2, c2)

        # conv3
        self.conv3 = self._make_stage(c2, c3, num_blocks=2)
        self.transition2 = TransitionBlock(c3, c3)

        # conv4
        self.conv4 = self._make_stage(c3, c4, num_blocks=2)

        # conv5
        self.conv5 = Conv1x1(c4, c4)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = c4
        self.neck = nn.BatchNorm1d(self.feature_dim)
        self.neck.bias.requires_grad_(False)

        self.embedding = nn.Linear(self.feature_dim, embedding_dim, bias=False)

        self.init_weight()

    @staticmethod
    def _resolve_width(config_name):
        """
        Resolve OSNet width multiplier.
        """
        configs = {
            "osnet_x1_0": 1.0,
            "osnet_x0_75": 0.75,
            "osnet_x0_5": 0.50,
            "osnet_x0_25": 0.25,
        }

        if isinstance(config_name, (int, float)):
            warnings.warn(
                "OSNet does not use ResNet-style depth configs like 50/101/152. "
                "Falling back to 'osnet_x1_0'."
            )
            return 1.0

        if config_name not in configs:
            raise ValueError(
                f"Unknown config_name={config_name}. "
                f"Expected one of {list(configs.keys())}."
            )

        return configs[config_name]

    @staticmethod
    def _make_channels(base_channels, width_mult):
        """
        Scale channels using width multiplier.
        """
        return max(8, int(round(base_channels * width_mult)))

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks):
        """
        Build one OSNet stage composed of repeated OSBlocks.
        """
        layers = [OSBlock(in_channels, out_channels)]
        for _ in range(num_blocks - 1):
            layers.append(OSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        """
        Forward only through backbone to produce pooled features.
        """
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.transition1(x)

        x = self.conv3(x)
        x = self.transition2(x)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        """
        Forward pass returning L2-normalized embeddings.
        """
        x = self.forward_features(x)
        x = self.neck(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def init_weight(self):
        """
        Initialize model weights.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)