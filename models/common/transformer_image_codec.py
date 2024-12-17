'''
Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    Added bitrate interpolation at training and inference time with learned
    gain parameters.
}
'''

import math
import time
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from timm.models.layers import trunc_normal_
from .tic_utils import conv, deconv
from typing import List, Dict, Any
from .rtsb import RSTB

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TIC(nn.Module):
    def __init__(self, N: int =128, M: int=192,in_channel: int=3, 
                 input_resolution: List[int]=[256,256], 
                 variable_bitrate: bool=False, levels:int=1,**kwargs):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer
        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer
        )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer
        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer
        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer
        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer
        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer)
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        ## Bitrate control
        self.variable_bitrate = variable_bitrate
        if self.variable_bitrate:
            self.levels = levels
            # gain/ inverse gain params for y latent
            self.gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
            self.inverse_gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
            #gain/inverse gain params for z latent
            self.hyper_gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
            self.inverse_hyper_gain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

        self.apply(self._init_weights)   
    
    def estimate_bitrate(self, likelihood: torch.Tensor):
        return torch.sum(-torch.log2(torch.clamp(likelihood, 2 ** (-16), 1.0))) 
    
    def g_a(self, x: torch.Tensor, x_size: Any=None)->tuple:
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        x = self.g_a2(x)
        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        x = self.g_a4(x)
        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        x = self.g_a6(x)
        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x: torch.Tensor, x_size:Any=None)->tuple:
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        x = self.g_s1(x)
        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        x = self.g_s3(x)
        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        x = self.g_s5(x)
        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        x = self.g_s7(x)
        return x, attns

    def h_a(self, x: torch.Tensor, x_size: Any=None)->torch.Tensor:
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x: torch.Tensor, x_size: Any=None)->torch.Tensor:
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self)->torch.Tensor:
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m: Any)->None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def compute_gain(self, x: torch.Tensor, rate_idx: int,hyper:bool=False)-> torch.Tensor:
        # Bitrate interpolation at training time
        if hyper:
            x =  x * torch.abs(self.hyper_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            x =  x * torch.abs(self.gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x

    def compute_inverse_gain(self, x: torch.Tensor, rate_idx: int,hyper:bool=False)->torch.Tensor:
        # Inverse bitrate interpolation at training time
        if hyper:
            x =  x * torch.abs(self.inverse_hyper_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            x =  x * torch.abs(self.inverse_gain[rate_idx]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x
    
    def compute_interpolated_gain(self, x:torch.Tensor, rate_idx:int, q_value: float, hyper:bool=False)->torch.Tensor:
        # Bitrate interpolation at inference time
        if hyper:
            gain = torch.abs(self.hyper_gain[rate_idx]) * (1 - q_value) + torch.abs(self.hyper_gain[rate_idx + 1]) * q_value
        else:
            gain = torch.abs(self.gain[rate_idx]) * (1 - q_value) + torch.abs(self.gain[rate_idx + 1]) * q_value
        x = x * gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x
    
    def compute_inverse_interpolated_gain(self, x:torch.Tensor, rate_idx:int, q_value:float, hyper:bool=False)-> torch.Tensor:
        # Inverse bitrate interpolation at inference time
        if hyper:
            inv_gain = torch.abs(self.inverse_hyper_gain[rate_idx]) * (1 - q_value) + torch.abs(self.inverse_hyper_gain[rate_idx + 1]) * (q_value)
        else:
            inv_gain = torch.abs(self.inverse_gain[rate_idx]) * (1 - q_value) + torch.abs(self.inverse_gain[rate_idx + 1]) * (q_value)
        x = x * inv_gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x


    def forward(self, x: torch.Tensor, rate_idx:int=0):
        B,_,H,W = x.shape
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        if self.variable_bitrate:
            y = self.compute_gain(y, rate_idx)
        z = self.h_a(y, x_size)
        
        if self.variable_bitrate:
            z = self.compute_gain(z, rate_idx, hyper=True)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_bpp = self.estimate_bitrate(z_likelihoods)/(B*H*W)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)

        gaussian_params = self.h_s(z_hat, x_size)
        
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_bpp = self.estimate_bitrate(y_likelihoods)/(B*H*W)
        if self.variable_bitrate:
            y_hat = self.compute_inverse_gain(y_hat, rate_idx)

        x_hat, attns_s = self.g_s(y_hat, x_size)
        total_bpp = y_bpp+z_bpp
        return x_hat.clamp_(0,1),total_bpp,y_hat


    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def compress(self, x: torch.Tensor, rate_idx: int=0, interpol_value:float=0.0)->Dict[str, Any]:
        enc_start = time.time()
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        if self.variable_bitrate:
            y = self.compute_gain(y, rate_idx)
        z = self.h_a(y, x_size)
        if self.variable_bitrate:
            z = self.compute_gain(z, rate_idx, hyper=True)
            
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx,  hyper=True)
        
        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        bts = (len(y_strings[0]) + len(z_strings[0])) * 8
        enc_time = time.time() - enc_start

        #decode
        dec_start = time.time()
        rec_frame = self.decompress([y_strings, z_strings],  z.size()[-2:], rate_idx, interpol_value)
        dec_time = time.time() - dec_start

        return {"decoded": rec_frame, "bitstring_size":bts,'time':{'enc_time': enc_time,'dec_time': dec_time}}

    def decompress(self, strings:Any, shape:Any, rate_idx:int=0, interpol_value: float=0.0)->torch.Tensor:
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if self.variable_bitrate:
            z_hat = self.compute_inverse_gain(z_hat, rate_idx, hyper=True)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        if self.variable_bitrate:
            y_hat = self.compute_inverse_gain(y_hat, rate_idx)
            
        x_hat, attns_s = self.g_s(y_hat)
        return x_hat.clamp_(0, 1)

