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


class ConvBlock(nn.Module):
    def __init__(self, in_ft,out_ft, kernel_size=5,stride=2, act=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ft,out_ft,kernel_size=kernel_size,stride=stride,padding=kernel_size // 2,)
        self.act = act
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.tensor)->torch.Tensor:
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x
        
class DeconvBlock(nn.Module):
    def __init__(self,in_ft, out_ft,kernel_size=5, stride=2, act=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
                                        in_ft,out_ft,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=stride - 1,
                                        padding=kernel_size // 2,
                                    )
        self.act = act
        if self.act:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.deconv(x)
        if self.act:
            x = self.relu(x)
        return x
    

class ResidualCoder(CompressionModel):
    '''Residual and Temporal residual Coding at low bitrates
    '''
    def __init__(self,in_ft:int,out_ft:int, N:int, M:int, scale_factor:int=1,**kwargs):
        super(ResidualCoder, self).__init__()
        num_int_layers = kwargs['num_intermediate_layers']
        self.g_a = nn.Sequential()
        self.g_a.add_module("inp", ConvBlock(in_ft, N))
        for idx in range(num_int_layers):
            self.g_a.add_module(f"conv_{idx}",ConvBlock(N,N))
        self.g_a.add_module("out",ConvBlock(N, M, act=False))
        
        self.g_s = nn.Sequential()
        self.g_s.add_module('inp', DeconvBlock(M,N))
        for idx in range(num_int_layers):
            self.g_s.add_module(f"deconv_{idx}", DeconvBlock(N,N))
        self.g_s.add_module('out',DeconvBlock(N, out_ft, act=False))
        
        
        self.h_a = nn.Sequential(
            ConvBlock(M, N, stride=1, kernel_size=3,act=False),nn.LeakyReLU(inplace=True),
            ConvBlock(N, N,act=False),nn.LeakyReLU(inplace=True),
            ConvBlock(N, N, act=False))
        
        self.h_s = nn.Sequential(
            DeconvBlock(N, M, act=False),nn.LeakyReLU(inplace=True),
            DeconvBlock(M, M * 3 // 2, act=False),nn.LeakyReLU(inplace=True),
            DeconvBlock(M * 3 // 2, M * 2, stride=1, kernel_size=3, act=False),)

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

    def resize(self, frame: torch.Tensor, scale_factor:float=1)->torch.Tensor:
        return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)

    def estimate_bitrate(self, likelihood: torch.Tensor)-> torch.Tensor:
        return torch.sum(-torch.log2(torch.clamp(likelihood, 2 ** (-16), 1.0))) 
    

    def compute_gain(self, x: torch.Tensor, rate_idx: int,hyper:bool=False)-> torch.Tensor:
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

    def forward(self, x: torch.Tensor,rate_idx:int=0, scale_factor: float=1)->tuple:
        if scale_factor != 1:
            x  = self.resize(x, scale_factor)
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
        if scale_factor != 1:
            x_hat  = self.resize(x_hat, 1//scale_factor)
        total_bpp = y_bpp+z_bpp
        return x_hat, total_bpp, y_likelihoods

    def similarity(self, prev, cur)->torch.Tensor:
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(prev, cur)
        return output

    def rans_compress(self, residual: torch.Tensor,  prev_latent: torch.Tensor, 
                      rate_idx: int=0,q_value: float=1.0,use_skip:bool=False, 
                      skip_thresh: float=0.95, scale_factor: float=1.0):
        enc_start = time.time()
        B,C,H,W = residual.shape
        if scale_factor != 1:
            residual = self.resize(residual,scale_factor)

        y = self.g_a(residual)
        if prev_latent != None and use_skip:
            sim = torch.mean(self.similarity(prev_latent, y)).item()
        else:
            sim = 0
        if sim > skip_thresh:
            #skip this residual
            return None, True
        else:
            if self.variable_bitrate:
                y = self.compute_gain(y, rate_idx)
        
            z = self.h_a(y)
            if self.variable_bitrate:
                z = self.compute_gain(z, rate_idx, hyper=True)
            
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            if self.variable_bitrate:
                z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)

            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            # scale_h, mean_h = self.get_averages(scales_hat, means_hat, H,W)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
            bts = (len(y_strings[0])+len(z_strings[0])) * 8
            enc_time = time.time() - enc_start
            dec_start = time.time()
            res_hat = self.rans_decompress([y_strings, z_strings], z.size()[-2:],scale_factor=scale_factor,rate_idx=rate_idx, q_value=q_value)
            dec_time = time.time() - dec_start
            #update bitstream info
            out = {'time':{'enc_time': enc_time,'dec_time': dec_time},
                    'bitstring_size':bts}
            out.update({'res_hat':res_hat,'prev_latent':y})
            return out, False

    def rans_decompress(self, strings, shape,scale_factor: float=1, rate_idx: int=0, q_value: float=1.0):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat)
        
        if self.variable_bitrate:
            y_hat = self.compute_inverse_gain(y_hat, rate_idx)

        x_hat = self.g_s(y_hat)
        if scale_factor != 1:
            x_hat = self.resize(x_hat, 1//scale_factor)
        return x_hat
   

class ConditionalResidualCoder(CompressionModel):
    '''A low bitrate conditional residual and temporal conditional residual coding'''
    def __init__(self,in_ft,out_ft, N, M,temporal=False, scale_factor=1,**kwargs):
        super(ConditionalResidualCoder, self).__init__()
        num_int_layers = kwargs['num_intermediate_layers']
        self.temporal = temporal
        self.g_a = nn.Sequential()
        in_dim = in_ft*2
        if self.temporal:
            in_dim += 3
        self.g_a.add_module("inp", ConvBlock(in_dim, N))
        for idx in range(num_int_layers):
            self.g_a.add_module(f"conv_{idx}",ConvBlock(N,N))
        self.g_a.add_module("out",ConvBlock(N, M, act=False))

        self.g_a_p = nn.Sequential()
        cn_dim = in_ft #Input dimension of the conditioning information
        if temporal:
            cn_dim +=3
        self.g_a_p.add_module("inp", ConvBlock(cn_dim, N))
        for idx in range(num_int_layers):
            self.g_a_p.add_module(f"conv_{idx}",ConvBlock(N,N))
        self.g_a_p.add_module("out",ConvBlock(N, M, act=False))
        
        self.g_s = nn.Sequential()
        self.g_s.add_module('inp', DeconvBlock(M*2,N))
        for idx in range(num_int_layers):
            self.g_s.add_module(f"deconv_{idx}", DeconvBlock(N,N))
        self.g_s.add_module('out',DeconvBlock(N, out_ft, act=False))
        
        
        self.h_a = nn.Sequential(
            ConvBlock(M, N, stride=1, kernel_size=3,act=False),nn.LeakyReLU(inplace=True),
            ConvBlock(N, N,act=False),nn.LeakyReLU(inplace=True),
            ConvBlock(N, N, act=False))
        
        self.h_s = nn.Sequential(
            DeconvBlock(N, M, act=False),nn.LeakyReLU(inplace=True),
            DeconvBlock(M, M * 3 // 2, act=False),nn.LeakyReLU(inplace=True),
            DeconvBlock(M * 3 // 2, M * 2, stride=1, kernel_size=3, act=False),)

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

    def resize(self, frame: torch.Tensor, scale_factor: float=1)-> torch.Tensor:
        return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)

    def estimate_bitrate(self, likelihood: torch.Tensor) -> torch.Tensor:
        return torch.sum(-torch.log2(torch.clamp(likelihood, 2 ** (-16), 1.0))) 
    
    def compute_gain(self, x: torch.Tensor, rate_idx: int,hyper: bool=False)-> torch.Tensor:
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

    def similarity(self, prev, cur)->torch.Tensor:
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(prev, cur)
        return output
    
    def forward(self, residual, animated_frame,prev_residual=None,scale_factor=1, rate_idx=0):
        if scale_factor != 1:
            residual  = self.resize(residual, scale_factor)
            animated_frame  = self.resize(animated_frame, scale_factor)
            if prev_residual is not None:
                prev_residual  = self.resize(prev_residual, scale_factor)

        B,H,W,_ = residual.shape
        if self.temporal:
            cr_inp = torch.cat([residual,prev_residual,animated_frame], dim=1)
        else:
            cr_inp = torch.cat([residual, animated_frame], dim=1)
        y = self.g_a(cr_inp)

        if self.temporal:
            cn_inp = torch.cat([prev_residual, animated_frame], dim=1)
            y_p =  self.g_a_p(cn_inp)
        else:
            y_p =  self.g_a_p(animated_frame)
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
        cr_bt = torch.cat((y_hat, y_p), dim=1)
        x_hat = self.g_s(cr_bt)

        if scale_factor != 1:
            x_hat  = self.resize(x_hat, 1//scale_factor)
        total_bpp = y_bpp+z_bpp
        return x_hat, total_bpp, y_likelihoods
    
    def get_averages(self,scales, means, H,W):
        b,c,h,w = scales.shape
        sf = H//h 
        scales = torch.mean(self.resize(scales, sf),dim=1, keepdim=True).repeat(1,3,1,1)
        means = torch.sum(self.resize(means, sf), dim=1, keepdim=True).repeat(1,3,1,1)
        return scales, means


    def rans_compress(self, residual, animated_frame,  prev_res_hat=None,prev_latent=None, rate_idx=0,q_value=1.0,scale_factor=1.0,use_skip=False, skip_thresh=0.9):
        enc_start = time.time()
        # self.scale_factor = scale_factor
        B,C,H,W = residual.shape
        if scale_factor != 1:
            residual = self.resize(residual,scale_factor)
            animated_frame = self.resize(animated_frame,scale_factor)
            if prev_res_hat is not None:
                prev_res_hat = self.resize(prev_res_hat, scale_factor)
        if self.temporal:
            c_in = torch.cat([residual,prev_res_hat, animated_frame], dim=1)
        else:
            c_in = torch.cat([residual, animated_frame], dim=1)
        y = self.g_a(c_in)

        if prev_latent != None and use_skip:
            sim = torch.mean(self.similarity(prev_latent, y)).item()
        else:
            sim = 0
        if sim > skip_thresh:
            #skip this residual
            return {'prev_latent': y}, True
        else:
            if self.variable_bitrate:
                y = self.compute_gain(y, rate_idx)
            

            z = self.h_a(y)
            if self.variable_bitrate:
                z = self.compute_gain(z, rate_idx, hyper=True)
            
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
            if self.variable_bitrate:
                z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)

            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            
            # scale_h, mean_h = self.get_averages(scales_hat, means_hat, H,W)
            indexes = self.gaussian_conditional.build_indexes(scales_hat)
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
            bts = (len(y_strings[0])+len(z_strings[0])) * 8
            enc_time = time.time() - enc_start
            dec_start = time.time()
            res_hat = self.rans_decompress([y_strings, z_strings], z.size()[-2:], animated_frame, prev_res_hat,scale_factor=scale_factor,rate_idx=rate_idx, q_value=q_value)
            dec_time = time.time() - dec_start
            #update bitstream info
            out = {'time':{'enc_time': enc_time,'dec_time': dec_time},
                    'bitstring_size':bts}
            out.update({'res_hat':res_hat,'prev_latent':y})
            return out, False

    def rans_decompress(self, strings, shape,animated_frame,prev_res_hat=None,scale_factor=1, rate_idx=0, q_value=1.0):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat)
        
        if self.variable_bitrate:
            y_hat = self.compute_inverse_gain(y_hat, rate_idx)
        if self.temporal:
            cn_in= torch.cat([prev_res_hat, animated_frame], dim=1)
            y_p = self.g_a_p(cn_in)
        else:
            y_p = self.g_a_p(animated_frame)
        c_out = torch.cat((y_hat, y_p), dim=1)
        x_hat = self.g_s(c_out)
        if scale_factor != 1:
            x_hat = self.resize(x_hat, 1//scale_factor)
        return x_hat
   

if __name__ == "__main__":

    from thop import profile
    
    img = torch.randn((1,3,256,256))
    rdc = ResidualCoder(3,3,48,int(48*1.5),variable_bitrate=True,levels=6,num_intermediate_layers=3)
    
    macs, params = profile(rdc, inputs=(img,))
    print("RDC|| GMacs: ",macs/1e9, " | #Params: ", params/1e6, " M")

    crdc = ConditionalResidualCoder(3,3,48,int(48*1.5),variable_bitrate=True,levels=6,num_intermediate_layers=3)
    macs, params = profile(crdc, inputs=(img,img))
    print("CRDC|| GMacs: ",macs/1e9, " | #Params: ", params/1e6, " M")