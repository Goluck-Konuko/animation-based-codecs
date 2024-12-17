import torch
from torch import nn
from typing import Dict, Any
import torch.nn.functional as F

from .refinement_net import RefinementNetwork
from .residual_coders import ResidualCoder, ConditionalResidualCoder

from ..common.dense_motion_generator import DenseMotionGenerator
from ..common.nn_utils import ResBlock2d, SameBlock2d,UpBlock2d, DownBlock2d, OutputLayer
from ..common.transformer_image_codec import TIC


class RDAC_Generator(nn.Module):
    """
    Animation-Based Generator with residual/ conditional residual coding.

    "Konuko et al., “Predictive coding for animation-based video compression”.ICIP 2023"
    "Konuko et al., “Improved predictive coding for animation-based video compression”.EUVIP 2024"


    """
    def __init__(self, num_channels=3, num_kp=10, block_expansion=64, max_features=1024, num_down_blocks=2,
                  num_bottleneck_blocks=3, estimate_occlusion_map=False,
                  dense_motion_params: Dict[str, Any]=None,residual_coder_params: Dict[str, Any]=None,iframe_params: Dict[str, Any]=None, **kwargs):
        super(RDAC_Generator, self).__init__()
        if dense_motion_params:
            self.dense_motion_network = DenseMotionGenerator(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,**dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

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
        
        # spatial residual coding 
        residual_features = residual_coder_params['residual_features']
        N, M = residual_features, int(residual_features*1.5)
        if residual_coder_params['residual_coding']:
            if residual_coder_params['residual_type'] =='conditional_residual':
                self.sdc = ConditionalResidualCoder(num_channels,num_channels, N, M,**residual_coder_params)
            else:
                self.sdc = ResidualCoder(num_channels,num_channels, N, M,**residual_coder_params)
        
        # temporal residual coding
        if residual_coder_params['temporal_residual_coding']:
            if residual_coder_params['residual_type'] =='conditional_residual':
                self.tdc = ConditionalResidualCoder(num_channels,num_channels, N, M,**residual_coder_params, temporal=True)
            else:
                self.tdc = ResidualCoder(num_channels,num_channels, N, M,**residual_coder_params)
        
        if kwargs['refinement_network_params']['gen_rec']:
            self.refinement_network = RefinementNetwork(**kwargs['refinement_network_params'])
        else:
            self.refinement_network = None

        if kwargs['ref_coder']:
            self.ref_coder = TIC(**iframe_params)
        else:
            self.ref_coder = None

        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.residual_coder_params = residual_coder_params
        self.motion_compensation = kwargs['motion_compensation']
        #reconstruction network
        self.ref_network_params = kwargs['refinement_network_params']
    
    def deform_input(self, inp: torch.Tensor, deformation: torch.Tensor)-> torch.Tensor:
        '''Motion compensation using bilinear interpolation'''
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def reference_ft_encoder(self, reference_frame: torch.Tensor)->torch.Tensor:
        out = self.first(reference_frame)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out

    def animated_frame_decoder(self, out: torch.Tensor) ->torch.Tensor:
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return out

    def motion_prediction_and_compensation(self,reference_frame: torch.Tensor=None, 
                                           reference_frame_features: torch.Tensor=None,
                                            **kwargs)-> tuple:
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        occlusion_map = dense_motion['occlusion_map']

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        
        if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
        reference_frame_features = reference_frame_features * occlusion_map
        
        return reference_frame_features, dense_motion

    

    def warp_residual(self, params: Dict[str, Any]) -> torch.Tensor:
        #generate a dense motion estimate
        dense_motion = self.dense_motion_network(reference_frame=params['prev_pred_frame'], kp_target=params['cur_frame_kp'],
                                                    kp_reference=params['prev_frame_kp'])
        #apply it to the residual map
        warped_residual = self.deform_input(params['prev_res_hat'], dense_motion['deformation'])
        return warped_residual
    

    def animate_training(self, params: Dict[str, Any])-> Dict[str, torch.Tensor]:
        #perform animation of previous frame
        output_dict = {}

        motion_pred_params = { 
                            'reference_frame': params['reference_frame'],
                            'reference_frame_features': params['ref_fts'],
                            'kp_reference': params['kp_reference'],
                            'kp_target': params['kp_target']}

        #then animate current frame with residual info from the current frame
        def_ref_fts, dense_motion_params  = self.motion_prediction_and_compensation(**motion_pred_params)

        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        
        animated_frame = self.animated_frame_decoder(out_ft_maps)
        output_dict["prediction"] = animated_frame
        output_dict['occlusion_map'] = dense_motion_params['occlusion_map']
        #Compute and encode a generalized difference between the animated and target images
        residual = params['target_frame'] - animated_frame
        output_dict['res'] = residual
        #if we have information about the previous frame residual
        #then we can use it as additional information to minimize the entropy of current frame
        if self.residual_coder_params['residual_coding']:
            if self.residual_coder_params['residual_type'] == 'conditional_residual':
                if self.residual_coder_params['temporal_residual_coding'] and 'prev_res_hat' in params:
                    residual_temp = (residual-params[f"prev_res_hat"])/2.0
                    output_dict['res_temp'] = residual_temp
                    res_hat_temp, bpp, prob = self.tdc(residual_temp,animated_frame,params[f"prev_res_hat"],rate_idx = params['rate_idx'])
                    output_dict['res_temp_hat'] = res_hat_temp
                    res_hat =  params["prev_res_hat"]+res_hat_temp*2.0                
                else:
                    res_hat, bpp, prob = self.sdc(residual,animated_frame,rate_idx = params['rate_idx'])
            
            else:
                if self.residual_coder_params['temporal_residual_coding'] and 'prev_res_hat' in params:
                    residual_temp = (residual-params[f"prev_res_hat"])/2.0
                    res_hat_temp, bpp, prob = self.tdc(residual_temp,rate_idx = params['rate_idx'])
                    output_dict['res_temp_hat'] = res_hat_temp
                    res_hat =  params["prev_res_hat"]+res_hat_temp*2.0                
                else:
                    res_hat, bpp, prob = self.sdc(residual,rate_idx = params['rate_idx'])
            
            output_dict['rate'] = bpp
            output_dict["res_hat"] = res_hat 
            # output_dict['res_latent'] = res_latent     
            output_dict['enhanced_prediction'] = (animated_frame+res_hat).clamp(0,1)
        
        if self.refinement_network:
            ref_pred = self.refinement_network(animated_frame+res_hat)
            # print(ref_pred.shape)
            output_dict['sr_prediction'] = ref_pred
            output_dict['res_noise'] = ref_pred - output_dict['enhanced_prediction']
        output_dict.update(**dense_motion_params)
        return output_dict
    
     
    def forward(self, **kwargs)-> Dict[str, Any]:     
        # Encoding (downsampling) part
        reference_frame = kwargs['reference']      
        ref_fts = self.reference_ft_encoder(reference_frame)
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        output_dict = {}
        for idx in range(kwargs['num_targets']):
            params = {'reference_frame': reference_frame,
                    'ref_fts': ref_fts,
                    'target_frame':kwargs[f'target_{idx}'],
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}'],
                    'rate_idx': kwargs[f'rate_idx']}
        
            if self.residual_coder_params['temporal_residual_coding'] and idx>0:
                # prev_animation = output_dict[f'enhanced_prediction_{idx-1}'] #.detach().clone()
                if f'sr_prediction_{idx-1}' in output_dict:
                    prev_animation = output_dict[f'sr_prediction_{idx-1}'].detach().clone()
                else:
                    prev_animation = output_dict[f'enhanced_prediction_{idx-1}'].detach().clone()

                params.update({'prev_pred_frame': prev_animation})
                if self.motion_compensation:
                    #create motion compensation params for residual
                    m_comp_params = {'prev_pred_frame': prev_animation,
                                     'cur_frame_kp':kwargs[f'kp_target_{idx}'],
                                     'prev_frame_kp':kwargs[f'kp_target_{idx-1}'],
                                     'prev_res_hat': output_dict[f'res_hat_{idx-1}']
                                     }
                    warped_prev_residual = self.warp_residual(m_comp_params)
                    params.update({'prev_res_hat': warped_prev_residual})
                else:
                    params.update({'prev_res_hat': output_dict[f'res_hat_{idx-1}']})
                    
            output = self.animate_training(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict

    def generate_animation(self, params:Dict[str, Any])-> torch.Tensor:     
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation   
        motion_pred_params = { 
                    'reference_frame': params['reference_frame'],
                    'reference_frame_features': params['ref_fts'],
                    'kp_reference':params['kp_reference'],
                    'kp_target':params['kp_target']
                }
        def_ref_fts, _ = self.motion_prediction_and_compensation(**motion_pred_params)
        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        #reconstruct the animated frame
        return self.animated_frame_decoder(out_ft_maps ,params['reference_frame'])
    

    def compress_spatial_residual(self,residual_frame:torch.Tensor, prev_latent:torch.Tensor=None,
                                  rate_idx=0, q_value=1.0,use_skip=False, skip_thresh=0.9, scale_factor=1.0)->Dict[str, Any]:
        res_info = self.sdc.rans_compress(residual_frame,prev_latent, rate_idx, q_value, use_skip, skip_thresh, scale_factor)
        return res_info

    def compress_temporal_residual(self,residual_frame:torch.Tensor, prev_latent:torch.Tensor,
                                  rate_idx=0, q_value=1.0,use_skip=False, skip_thresh=0.9, scale_factor=1.0)->Dict[str, Any]:
        res_info = self.tdc.rans_compress(residual_frame,prev_latent, rate_idx, q_value, use_skip, skip_thresh, scale_factor)
        return res_info
    
    def update(self, output_dict:Dict[str,torch.Tensor], output:Dict[str,torch.Tensor],idx:int)->Dict[str, torch.Tensor]:
        for item in output:
            output_dict.update({f"{item}_{idx}": output[item]})
        return output_dict

# if __name__ == "__main__":
#     from thop import profile
#     reference = torch.randn((1,3,256,256))
#     kp_reference = {'value': torch.randn((1,10,2), dtype=torch.float32)}
#     kp_target = {'value': torch.randn((1,10,2), dtype=torch.float32)}
#     res_coding_info = {'residual_coding':True, 
#                     'temporal_residual_coding':False,
#                     'residual_features': 48,
#                     'residual_type': 'residual',
#                     'num_intermediate_layers':3,
#                     'variable_bitrate': True,
#                     'levels':7}
#     rec_network_params = {'gen_rec': True, 'sample_noise': False}
#     generator = RDAC_Generator(motion_compensation=False,multiRes=True,dense_motion_params=None,temporal_animation=False,
#                               residual_coder_params=res_coding_info,
#                               rec_network_params=rec_network_params, ref_coder=False)
#     # out = generator(reference)
#     # print(out.shape)
#     macs, params = profile(generator, inputs=(reference,))
#     print("Macs: ",macs/1e9, " GMACs | #Params: ", params/1e6, " M")

