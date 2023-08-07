import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F
from .gdc import DifferenceCoder
from .dmg import DenseMotionGenerator
from .nn_utils import ResBlock2d, SameBlock2d,UpBlock2d, DownBlock2d, OutputLayer 


class GeneratorDAC(nn.Module):
    """
    Similar Architecture to GeneratorFOM. Trained without jacobians i.e. Zero-order motion representation
    --added bitrate estimation method.
    """
    def __init__(self, num_channels=3, num_kp=10, block_expansion=64, max_features=1024, num_down_blocks=2,
                 num_bottleneck_blocks=3, estimate_occlusion_map=False, dense_motion_params=None, upsampler_params = None,**kwargs):
        super(GeneratorDAC, self).__init__()
        self.dense_motion_network = DenseMotionGenerator(num_kp=num_kp, num_channels=num_channels,
                                                        estimate_occlusion_map=estimate_occlusion_map,**dense_motion_params)

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3),)         

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = block_expansion * (2 ** num_down_blocks)
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
    
        self.final = OutputLayer(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
    
    def rescale(self, frame, scale_factor=1):
        return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)
        
    def deform_input(self, inp, deformation):
        '''Motion compensation using bilinear interpolation'''
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def reference_ft_encoder(self, reference_frame):
        out = self.first(reference_frame)        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out

    def animated_frame_decoder(self, out):
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return out

    def motion_prediction_and_compensation(self,reference_frame: torch.tensor=None, 
                                           reference_frame_features: torch.tensor=None,
                                            **kwargs):
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        occlusion_map = dense_motion['occlusion_map']

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        
        if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
        reference_frame_features = reference_frame_features * occlusion_map
        
        return reference_frame_features

    def animate(self, params) -> torch.Tensor:     
        # Encoding (downsampling) part      
        ref_fts = self.reference_ft_encoder(params['reference_frame'])
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': params['reference_frame'],
                    'reference_frame_features': ref_fts,
                    'kp_reference':params['kp_reference'],
                    'kp_target':params['kp_target']
                }
        def_ref_fts = self.motion_prediction_and_compensation(**motion_pred_params)
        # Decoding bottleneck layer
        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        #reconstruct the animated frame
        return self.animated_frame_decoder(out_ft_maps)
    
    def forward(self, reference_frame: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:       
        # Encoding (downsampling) part      
        ref_fts = self.reference_ft_encoder(reference_frame)
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        motion_pred_params = { 
                    'reference_frame': reference_frame,
                    'reference_frame_features': ref_fts,
                    'kp_reference':kwargs['kp_reference'],
                    'kp_target':kwargs['kp_target_0']
                }

        def_ref_fts  = self.motion_prediction_and_compensation(**motion_pred_params)
        # Decoding part
        out = self.bottleneck(def_ref_fts) #input the weighted average 
        output_dict["prediction_0"] = self.animated_frame_decoder(out)
        return output_dict

class GeneratorRDAC(GeneratorDAC):
    """
    Generator architecture using spatial attention layers and ConvNext Modules
    """
    def __init__(self, num_channels=3, num_kp=10, block_expansion=64, max_features=1024, num_down_blocks=2,
                 num_bottleneck_blocks=3, estimate_occlusion_map=False, 
                 dense_motion_params=None,
                 sdc_params=None,rec_network_params=None,
                 upsampler_params=None, 
                 **kwargs):
        super(GeneratorRDAC, self).__init__(num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map, dense_motion_params,upsampler_params, **kwargs)
        # spatial_difference_coder
        self.sdc = DifferenceCoder(num_channels,num_channels, kwargs['residual_features'], int(kwargs['residual_features']*1.5),**sdc_params)

        if rec_network_params['gen_rec']:
            self.rec_network = nn.Sequential()
            self.rec_network.add_module('up_conv_0',SameBlock2d(num_channels*2,block_expansion))
            for idx in range(1, rec_network_params['num_blocks']):
                self.rec_network.add_module(f'up_conv_{idx}',ResBlock2d(block_expansion,  kernel_size=(3, 3), padding=(1, 1)))
            self.rec_network.add_module('up_conv_out',OutputLayer(block_expansion, num_channels))
        else:
            self.rec_network = None

    def deform_residual(self, params: Dict[str, Any]) -> torch.tensor:
        #generate a dense motion estimate
        dense_motion = self.dense_motion_network(reference_image=params['prev_rec'], kp_target=params['kp_cur'],
                                                    kp_reference=params['kp_prev'])
        #apply it to the residual map
        warped_residual = self.deform_input(params['res_hat_prev'], dense_motion['deformation'])
        return warped_residual
    
    def train_spatial_residual(self, params: Dict[str, Any]):
        #perform animation of previous frame
        output_dict = {}
        animated_frame = self.animate(params)
        residual = params['target_frame'] - animated_frame
        output_dict['res'] = residual

        #downsample the residual map
        res_hat, bpp = self.sdc(residual)
        # res_hat = torch.clamp(res_hat,-1,1)
        output_dict['res_hat'] = res_hat
        output_dict['rate'] = bpp
        if self.rec_network:
            output_dict['enhanced_prediction'] = self.rec_network(torch.cat((animated_frame, res_hat), dim=1))
        else:
            output_dict['enhanced_prediction'] = (animated_frame+res_hat).clamp(0,1)

        return output_dict

    def update(self, output_dict, output,idx):
        output_dict.update({f'prediction_{idx}': output['prediction']})
        output_dict.update({f'res_{idx}': output['res']})
        output_dict.update({f'res_hat_{idx}': output['res_hat']})
        output_dict.update({f'rate_{idx}': output['rate']}) 
        output_dict.update({f'enhanced_pred_{idx}': output['enhanced_prediction']})
        return output_dict
    
    def forward(self, reference_frame, **kwargs):     
        output_dict = {}
        for idx in range(kwargs['n_target']):
            params = {'reference_frame': reference_frame,
                    'target_frame':kwargs[f'target_{idx}'],
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}']}
            
            output = self.train_spatial_residual(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict

    def compress_spatial_residual(self,residual_frame:torch.Tensor,scale_factor:float=1.0)->Dict[str, Any]:
        res_info = self.sdc.rans_compress(residual_frame, scale_factor)
        return res_info


class GeneratorRDAC_T(GeneratorRDAC):
    """
    Residual coding and temporal learning:
        - the temporal difference is computed directly between the current
        residual and the previously decoded residual
    """
    def __init__(self, num_channels=3, num_kp=10, block_expansion=64, max_features=1024, num_down_blocks=2,
                 num_bottleneck_blocks=3, estimate_occlusion_map=False, dense_motion_params=None,
                 sdc_params=None, rec_network_params=None,upsampler_params=None,  **kwargs):
        super(GeneratorRDAC_T, self).__init__(num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map, dense_motion_params,sdc_params,rec_network_params,upsampler_params, **kwargs)
        #Temporal_difference_coder 
        self.tdc= DifferenceCoder(num_channels,num_channels, kwargs['residual_features'], int(kwargs['residual_features']*1.5),**sdc_params)
    
    def train_temporal_residual(self, params: Dict[str, Any]):
        ## Current frame animation
        output_dict = {}
        motion_pred_params = { 
                                'reference_frame': params['reference_frame'],
                                'reference_frame_features': params['ref_fts'],
                                'kp_reference': params['kp_reference'],
                                'kp_target': params['kp_target']
                            }
        
        #animate current frame
        def_out = self.motion_prediction_and_compensation(**motion_pred_params)
        
        def_ref_fts  = def_out['def_ft_maps']
        # Decoding bottleneck layer
        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        
        out = self.animated_frame_decoder(out_ft_maps)
        output_dict["prediction"] = out

        #Compute and encode a generalized difference between the animated and target images
        residual = params['target_frame'] - out
        
        #if we have information about the previous frame residual
        #then we can use it as additional information to minimize the entropy of current frame
        
        residual_temp = residual-params[f"res_hat_prev"]
        output_dict['res'] = residual
        output_dict['res_temp'] = residual_temp

        #compress the temporal residual
        res_hat_temp, bpp = self.tdc(residual_temp)
        # res_hat_temp = torch.clamp(res_hat_temp,-1,1)
        output_dict['res_temp_hat'] = res_hat_temp
        # distortion = torch.mean((res_hat_temp - residual_temp).pow(2))
        
        output_dict['rd_info'] = {'rate':bpp}

        #reconstitute actual synthesis residual from previous frame and current
        res_hat =  params["res_hat_prev"]+res_hat_temp*2.0
        output_dict["res_hat"] = res_hat        
        if self.rec_network:
            output_dict['enhanced_prediction'] = self.rec_network(torch.cat((out, res_hat), dim=1))
        else:
            output_dict['enhanced_prediction'] = (out+res_hat).clamp(0,1)
        return output_dict

    def compress_spatial_residual(self,cur_residual_frame:torch.Tensor,prev_residual_frame:torch.Tensor,scale_factor:float=1.0)->Dict[str, Any]:
        temporal_residual = cur_residual_frame - prev_residual_frame
        res_info = self.sdc.rans_compress(temporal_residual, scale_factor)
        res_info.update({'res_hat':res_info['res_hat']+prev_residual_frame})
        return res_info
    
    def forward(self, reference_frame: torch.Tensor, **kwargs):     
        # Encoding (downsampling) part      
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        output_dict = {}
        for idx in range(kwargs['n_target']):
            params = {'reference_frame': reference_frame,
                    'target_frame':kwargs[f'target_{idx}'],
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}']}
            if idx == 0:
                output = self.train_spatial_residual(params)
            else:
                params.update({'res_hat_prev': output_dict[f'res_hat_0']})
                output = self.train_temporal_residual(params)
            
            if 'res_temp' in output:
                output_dict.update({f'res_temp_{idx}': output['res_temp']})
                output_dict.update({f'res_temp_hat_{idx}': output['res_temp_hat']})

            output_dict = self.update(output_dict, output, idx)
        return output_dict