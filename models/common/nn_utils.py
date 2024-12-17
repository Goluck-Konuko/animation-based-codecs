'''
Code from First Order Motion Model for Image Animation (FOMM) with minor updates.
-> syncbnorm changes to nn.BatchNorm2d
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class OutputLayer(nn.Module):
    def __init__(self, in_features:int, out_features:int=3, kernel_size:tuple=(7,7), padding:tuple=(3,3), activation:str='sigmoid') -> None:
        super(OutputLayer, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

class Mask(nn.Module):
    def __init__(self, in_features:int, out_features:int=3, kernel_size:tuple=(7,7), padding:tuple=(3,3)) -> None:
        super(Mask, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.softmax(self.conv(x))
        return out

class KP_Output(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,kernel_size:tuple=(7, 7), padding:tuple=(3,3)) -> None:
        super(KP_Output, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features:int, kernel_size:tuple=(3,3), padding:tuple=(1,1)):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        self.norm2 = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features:int, out_features:int,scale_factor:int=2, kernel_size:int=3, padding:int=1, groups:int=1):
        super(UpBlock2d, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)

        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=self.scale_factor,mode='bilinear', align_corners=True)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features:int, out_features:int, kernel_size:int=3, padding:int=1, groups:int=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features:int, out_features:int, groups:int=1, kernel_size:int=3, padding:int=1):
        super(SameBlock2d, self).__init__()        
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion:int, in_features:int, num_blocks:int=3, max_features:int=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x:torch.Tensor)->List[torch.Tensor]:
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs
    
class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion:int, in_features:int, num_blocks:int=3, max_features:int=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion:int, in_features:int, num_blocks:int=3, max_features:int=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.decoder(self.encoder(x))
