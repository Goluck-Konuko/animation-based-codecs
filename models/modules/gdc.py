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
                    conv(N,N),nn.ReLU(inplace=True),
                    conv(N, M))
        
        self.g_s = nn.Sequential(
                    deconv(M, N),nn.ReLU(inplace=True),
                    deconv(N, N),nn.ReLU(inplace=True),
                    deconv(N,N),nn.ReLU(inplace=True),
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
        self.variable_bitrate = kwargs['variable_bitrate']
        if self.variable_bitrate:
            self.levels = kwargs['levels']
            self.gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
            self.inverse_gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
            self.hyper_gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
            self.inverse_hyper_gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)


    def rescale(self, frame, scale_factor=1):
        return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)

    def estimate_bitrate(self, likelihood):
        return torch.sum(-torch.log2(torch.clamp(likelihood, 2 ** (-16), 1.0))) 
    

    def compute_gain(self, x: torch.Tensor, rate_idx: int,hyper=False)-> torch.Tensor:
        if hyper:
            x =  x * torch.abs(self.hyper_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            x =  x * torch.abs(self.gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x

    def compute_inverse_gain(self, x: torch.Tensor, rate_idx: int,hyper=False)->torch.Tensor:
        if hyper:
            x =  x * torch.abs(self.inverse_hyper_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            x =  x * torch.abs(self.inverse_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x
    
    def compute_interpolated_gain(self, x:torch.Tensor, rate_idx, q_value, hyper=False)->torch.Tensor:
        if hyper:
            gain = torch.abs(self.hyper_gain[rate_idx]) * (1 - q_value) + torch.abs(self.hyper_gain[rate_idx + 1]) * q_value
        else:
            gain = torch.abs(self.gain[rate_idx]) * (1 - q_value) + torch.abs(self.gain[rate_idx + 1]) * q_value
        x = x * gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x
    
    def compute_inverse_interpolated_gain(self, x:torch.Tensor, rate_idx:int, q_value:float, hyper=False)-> torch.Tensor:
        if hyper:
            inv_gain = torch.abs(self.inverse_hyper_gain[rate_idx]) * (1 - q_value) + torch.abs(self.inverse_hyper_gain[rate_idx + 1]) * (q_value)
        else:
            inv_gain = torch.abs(self.inverse_gain[rate_idx]) * (1 - q_value) + torch.abs(self.inverse_gain[rate_idx + 1]) * (q_value)
        x = x * inv_gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x

    def forward(self, x, rate_idx=0):
        if self.scale_factor != 1:
            x  = self.rescale(x, self.scale_factor)
        B,H,W,_ = x.shape
        y = self.g_a(x)
        #apply gain on latent (y)
        if self.variable_bitrate:
            y = self.compute_gain(y, rate_idx)
        z = self.h_a(y)
        #apply gain in hyperprior (z)
        if self.variable_bitrate:
            z = self.compute_gain(z, rate_idx, hyper=True)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_bpp = self.estimate_bitrate(z_likelihoods)/(B*H*W)
        #apply inverse gain on hyperprior (z_hat)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        y_bpp = self.estimate_bitrate(y_likelihoods)/(B*H*W)
        #apply inverse gain on latent (y_hat)
        if self.variable_bitrate:
            y_hat = self.compute_inverse_gain(y_hat, rate_idx)

        x_hat = self.g_s(y_hat)
        if self.scale_factor != 1:
            x_hat  = self.rescale(x_hat, 1//self.scale_factor)
        return x_hat, y_bpp+z_bpp
    
    def get_averages(self,scales, means, H,W):
        b,c,h,w = scales.shape
        sf = H//h 
        scales = torch.mean(self.rescale(scales, sf),dim=1, keepdim=True).repeat(1,3,1,1)
        means = torch.sum(self.rescale(means, sf), dim=1, keepdim=True).repeat(1,3,1,1)
        return scales, means

    def similarity(self, prev, cur)->torch.Tensor:
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(prev, cur)
        return output

    def rans_compress(self, residual,prev_latent, rate_idx=0,q_value=1.0,use_skip=False, skip_thresh=0.95):
        enc_start = time.time()
        B,C,H,W = residual.shape
        if self.scale_factor != 1:
            residual = self.rescale(residual,self.scale_factor)
        y = self.g_a(residual)
        if self.variable_bitrate:
            y = self.compute_interpolated_gain(y, rate_idx, q_value)
        if prev_latent != None and use_skip:
            sim = torch.mean(self.similarity(prev_latent, y)).item()
        else:
            sim = 0
        if sim > skip_thresh:
            #skip this residual
            return None, True
        else:
            z = self.h_a(y)
            if self.variable_bitrate:
                z = self.compute_interpolated_gain(z, rate_idx, q_value, hyper=True)
            
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            if self.variable_bitrate:
                z_hat = self.compute_inverse_interpolated_gain(z_hat, rate_idx, q_value, hyper=True)

            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            scale_h, mean_h = self.get_averages(scales_hat, means_hat, H,W)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
            bts = (len(y_strings[0])+len(z_strings[0])) * 8
            enc_time = time.time() - enc_start
            dec_start = time.time()
            res_hat = self.rans_decompress([y_strings, z_strings], z.size()[-2:],rate_idx=rate_idx, q_value=q_value)
            dec_time = time.time() - dec_start
            #update bitstream info
            out = {'time':{'enc_time': enc_time,'dec_time': dec_time},
                    'bitstring_size':bts}
            out.update({'res_hat':res_hat,'prev_latent':y, 
                        'scales': scale_h, 'means':mean_h})
            return out, False

    def rans_decompress(self, strings, shape, rate_idx=0, q_value=1.0):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_interpolated_gain(z_hat, rate_idx, q_value, hyper=True)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat)
        
        if self.variable_bitrate:
            y_hat = self.compute_inverse_interpolated_gain(y_hat, rate_idx, q_value)

        x_hat = self.g_s(y_hat)
        if self.scale_factor != 1:
            x_hat = self.rescale(x_hat, 1//self.scale_factor)
        return x_hat
