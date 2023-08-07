"""`
    Base architecture from Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    Design changes:
        - Number of downsampling/upsampling layers.
        - Activation function for efficient bit allocation for -/+ values
        expected in the residual layer. The original architecture uses non-negative
        activations (GDN or ReLU)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

class DifferenceCoder(CompressionModel):
    def __init__(self,in_ft,out_ft, N, M, scale_factor=1,**kwargs):
        super(DifferenceCoder, self).__init__()
        self.g_a = nn.Sequential(
                    conv(in_ft, N),nn.ReLU(inplace=True),
                    conv(N, N),nn.ReLU(inplace=True),
                    conv(N,N),nn.ReLU(inplace=True),
                    conv(N, M))
        
        self.g_s = nn.Sequential(
                    deconv(M, N),nn.ReLU(inplace=True),
                    deconv(N, N),nn.ReLU(inplace=True),
                    deconv(N,N),nn.ReLU(inplace=True),
                    deconv(N, out_ft))
        
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),nn.LeakyReLU(inplace=True),
            conv(N, N),nn.LeakyReLU(inplace=True),conv(N, N))
        
        self.h_s = nn.Sequential(
            deconv(N, M),nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3))

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.N = N
        self.M = M

        #some inputs may be subsambled before compression
        self.scale_factor = scale_factor

    def downsample(self, frame, scale_factor=1):
        return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)

    def estimate_bitrate(self, likelihood):
        return torch.sum(-torch.log2(torch.clamp(likelihood, 2 ** (-16), 1.0))) 

    def forward(self, x):
        if self.scale_factor != 1:
            x  = self.downsample(x, self.scale_factor)
        B,H,W,_ = x.shape
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_bpp = self.estimate_bitrate(z_likelihoods)/(B*H*W)
       
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_bpp = self.estimate_bitrate(y_likelihoods)/(B*H*W)
        x_hat = self.g_s(y_hat)
        if self.scale_factor != 1:
            x_hat  = self.downsample(x_hat, 1//self.scale_factor)
        return x_hat, y_bpp+z_bpp
    
    def rans_compress(self, residual, scale_factor=1):
        enc_start = time.time()
        if scale_factor != 1:
            residual = self.downsample(residual,scale_factor)
        y = self.g_a(residual)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        bts = (len(y_strings[0])+len(z_strings[0])) * 8
        enc_time = time.time() - enc_start
        dec_start = time.time()
        res_hat = self.rans_decompress([y_strings, z_strings], z.size()[-2:], scale_factor=scale_factor)
        dec_time = time.time() - dec_start
        #update bitstream info
        out = {'time':{'enc_time': enc_time,'dec_time': dec_time},
                'bitstring_size':bts}
        out.update({'res_hat':res_hat})
        return out

    def rans_decompress(self, strings, shape, scale_factor=1):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat)
        if scale_factor != 1:
            x_hat = self.downsample(x_hat, 1//scale_factor)
        return x_hat