import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ResidualBlock import ResidualBlock, ConvBlock
except ImportError:  
    from ResidualBlock import ResidualBlock, ConvBlock

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
    
class ApexID(nn.Module):
    def __init__(self, 
                 config_name, 
                 in_channels, # might be 4 new one for FFT
                 embedding_dim, 
                 C = 32):
        """Initialize the ApexID backbone and embedding projection head.

        Args:
            config_name (int): Backbone depth identifier selecting the residual
                stage configuration.
            in_channels (int): Number of channels in the input image tensor.
            embedding_dim (int): Size of the normalized embedding vector produced by
                the network.
            C (int, optional): Cardinality used in grouped residual convolutions.

        Returns:
            None
        """
        super().__init__()
        
        configurations = {
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }

        num_blocks = configurations[config_name]
        out_features = [256, 512, 1024, 2048]

        self.blocks = nn.ModuleList([ResidualBlock(64, 256, 1, True, cardinality=C)])

        for i in range(len(out_features)):
            if i > 0:
                self.blocks.append(ResidualBlock(out_features[i-1], out_features[i], 2, cardinality=C))
            for _ in range(num_blocks[i]-1):
                self.blocks.append(ResidualBlock(out_features[i], out_features[i], 1, cardinality=C))
        
        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.neck = nn.BatchNorm1d(2048)
        self.embedding = nn.Linear(2048, embedding_dim, bias=False)
        self.silu = nn.SiLU()

        self.init_weight()

    def forward(self, x):
        """Encode an input batch into L2-normalized embeddings.

        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            torch.Tensor: Normalized embedding tensor for the input batch.
        """
        x = self.silu(self.conv1(x))
        x = self.maxpool(x)

        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        x = self.embedding(x)
        x = F.normalize(x)  
        return x

    def init_weight(self):
        """Initialize convolution, linear, and batch normalization layers.

        Args:
            None

        Returns:
            None
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
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
