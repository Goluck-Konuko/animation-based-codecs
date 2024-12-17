import torch

class RefinementNetwork(torch.nn.Module):
    def __init__(self, in_channel: int=3, out_channel: int=3, block_expansion: int=64,
                 pixel_shuffle: bool=False,**kwargs):
        super(RefinementNetwork, self).__init__()
        self.dconv_down1 = double_conv(in_channel, block_expansion)
        self.dconv_down2 = double_conv(block_expansion, block_expansion*2 )
        self.dconv_down3 = double_conv(block_expansion*2, block_expansion*2 )
        self.dconv_down4 = double_conv(block_expansion*2, block_expansion*4)        

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   

        self.dconv_up3 = double_conv(block_expansion*2 + block_expansion*4, block_expansion*2)
        self.dconv_up2 = double_conv(block_expansion*2 + block_expansion*2, block_expansion)
        self.dconv_up1 = double_conv(block_expansion + block_expansion, block_expansion)

        self.conv_last = torch.nn.Conv2d(block_expansion, out_channel, 1)

        self.pixel_shuffle = pixel_shuffle

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
 
        x = self.upsample(x)  
        
        x = torch.cat([x, conv2], dim=1)    
        x = self.dconv_up2(x)

        x = self.upsample(x)  
        x = torch.cat([x, conv1], dim=1)   
        x = self.dconv_up1(x)

            
        x = self.conv_last(x)
        return torch.sigmoid(x)
    
def double_conv(in_channels, out_channels, padding=1):
    return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, 3, padding=padding),
                torch.nn.ReLU(inplace=True)
                ) 
 

