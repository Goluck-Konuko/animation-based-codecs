import torch
from torch import nn
from typing import Dict, Any, List
import torch.nn.functional as F
from ..common.dense_motion_generator import DenseMotionGenerator
from ..common.nn_utils import ResBlock2d, SameBlock2d,UpBlock2d, DownBlock2d, OutputLayer
from ..common.transformer_image_codec import TIC
from .utils import laplacian_feature_filter

class HDAC_Generator(nn.Module):
    """
    Motion Transfer Generator with a scalable base layer encoder and a conditional feature fusion

    "Konuko et al ,“A hybrid deep animation codec for low-bitrate video conferencing,” in ICIP, 2022"
    """
    def __init__(self, num_channels: int=3, num_kp:int =10, block_expansion:int=64, max_features:int=1024, num_down_blocks:int=2,
                 num_bottleneck_blocks:int=3, estimate_occlusion_map:bool=False, dense_motion_params:Dict[str, Any]=None,iframe_params:Dict[str, Any]=None,**kwargs):
        super(HDAC_Generator, self).__init__()
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

        self.base = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        base_down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            base_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.base_down_blocks = nn.ModuleList(base_down_blocks)
        
        main_bottleneck = []
        in_features = block_expansion * (2 ** num_down_blocks)*2
        #Regular residual block architecture
        for i in range(num_bottleneck_blocks):
            main_bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.main_bottleneck = nn.ModuleList(main_bottleneck)
        self.bt_output_layer = SameBlock2d(in_features, in_features//2, kernel_size=(3, 3), padding=(1, 1))
        #Upsampling layers
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = OutputLayer(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))


        if kwargs['ref_coder']:
            self.ref_coder = TIC(**iframe_params)
        else:
            self.ref_coder = None

        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
    
    
    
    def deform_input(self, inp, deformation):
        '''Motion compensation using bilinear interpolation'''
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def reference_ft_encoder(self, reference_frame: torch.Tensor)-> torch.Tensor:
        '''Embedding network -> extracts a latent representation from the reference frame'''
        out = self.first(reference_frame)        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out
    
    def base_layer_ft_encoder(self, base_layer_frame : torch.Tensor) -> torch.Tensor:
        '''Encodes the base layer frame into a latent code with dimensions
          matching the latent obtained from the reference frame'''
        out = self.base(base_layer_frame)    
        for i in range(len(self.base_down_blocks)):
            out = self.base_down_blocks[i](out)
        return out

    def animated_frame_decoder(self, out: torch.Tensor)->torch.Tensor:
        '''Frame generation from the transformed latent representation'''
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return out

    def deform_base_layer(self, params: Dict[str, Any]) -> torch.Tensor:
        #generate a dense motion estimate between the base layer idx and the target idx
        dense_motion = self.dense_motion_network(reference_frame=params['reference'], kp_target=params['kp_cur'],
                                                    kp_reference=params['kp_prev'])
        #apply the predicted optical flow to warp the base layer frame towards the target idx
        warped = self.deform_input(params['deform_target'], dense_motion['deformation'])
        return warped

    def motion_prediction_and_compensation(self,reference_frame: torch.tensor=None, 
                                           reference_frame_features: torch.tensor=None,
                                            **kwargs)->tuple:
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        occlusion_map = dense_motion['occlusion_map']

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        
        if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
        reference_frame_features = reference_frame_features * occlusion_map
        
        return reference_frame_features, dense_motion

    def train_forward(self, params:Dict[str, Any])->Dict[str, torch.Tensor]:       
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': params['reference_frame'],
                    'reference_frame_features': params['ref_fts'],
                    'kp_reference':params['kp_reference'],
                    'kp_target':params['kp_target']
                }

        def_ref_fts, motion_info  = self.motion_prediction_and_compensation(**motion_pred_params)
        
        bl_fts = self.base_layer_ft_encoder(params['base_layer'])
        #Investigate the effect of base layer masking at training time

        bt_input = torch.cat((def_ref_fts,bl_fts), dim=1)
        #block-wise fusion at the generator bottleneck
        for layer in self.main_bottleneck:
            bt_input = layer(bt_input)
        bt_output = self.bt_output_layer(bt_input)

        output = {}
        output['context'] = bt_output.detach().clone()
        output['prediction'] = self.animated_frame_decoder(bt_output)
        output['occlusion_map'] = motion_info['occlusion_map']
        return output
    
    def forward(self, **kwargs)->Dict[str, Any]:     
        # Encoding (downsampling) part      
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        output_dict = {} 
        reference_frame = kwargs['reference']

        if self.ref_coder is not None:
            #Compress the reference frame first if a pretrained image codec is available
            with torch.no_grad():
                dec_reference, ref_bpp, _ = self.ref_coder(reference_frame, rate_idx=kwargs['rate_idx'])
        else:
            dec_reference = reference_frame
        
        # Encoding (downsampling) part      
        ref_fts = self.reference_ft_encoder(dec_reference)          
        
        
        output_dict.update({'reference': dec_reference})
        
        for idx in range(kwargs['num_targets']):
            params = {'reference_frame': dec_reference,
                      'ref_fts': ref_fts,
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}'],
                    'base_layer':kwargs[f'base_layer_{idx}']}
            output = self.train_forward(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict
    
    def update(self, output_dict:Dict[str,torch.Tensor], output:Dict[str,torch.Tensor],idx:int)->Dict[str, torch.Tensor]:
        for item in output:
            output_dict.update({f"{item}_{idx}": output[item]})
        return output_dict
        
    def generate_animation(self, reference_frame: torch.Tensor, 
                    kp_reference:Dict[str, torch.Tensor],
                    kp_target:Dict[str, torch.Tensor],
                    base_layer: torch.Tensor)->torch.Tensor: 
        '''Forward inference process'''      
        # Encoding (downsampling) part          
        ref_fts = self.reference_ft_encoder(reference_frame)
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': reference_frame,
                    'reference_frame_features':ref_fts,
                    'kp_reference': kp_reference,
                    'kp_target': kp_target
                }

        def_ref_fts, _  = self.motion_prediction_and_compensation(**motion_pred_params)      

        bl_fts = self.base_layer_ft_encoder(base_layer)
        bt_input = torch.cat((def_ref_fts,bl_fts), dim=1)
        #block-wise fusion at the generator bottleneck
        for layer in self.main_bottleneck:
            bt_input = layer(bt_input)
        bt_output = self.bt_output_layer(bt_input)
        return self.animated_frame_decoder(bt_output)


class HDAC_HF_Generator(nn.Module):
    """
    HDAC framework with High frequency shuttling mechanism.

    "Konuko et al., “Improving Reconstruction Fidelity in Generative Face Video Coding using High-Frequency Shuttling”.VCIP 2024"
    
    """
    def __init__(self, num_channels: int=3, num_kp:int =10, block_expansion:int=64, max_features:int=1024, num_down_blocks:int=2,
                 num_bottleneck_blocks:int=3, estimate_occlusion_map:bool=False, dense_motion_params:Dict[str, Any]=None,iframe_params:Dict[str, Any]=None,**kwargs):
        super(HDAC_HF_Generator, self).__init__()
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

        self.base = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        base_down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            base_down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.base_down_blocks = nn.ModuleList(base_down_blocks)
        
        main_bottleneck = []
        in_features = block_expansion * (2 ** num_down_blocks)*2
        #Regular residual block architecture
        for i in range(num_bottleneck_blocks):
            main_bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.main_bottleneck = nn.ModuleList(main_bottleneck)
        self.bt_output_layer = SameBlock2d(in_features, in_features//2, kernel_size=(3, 3), padding=(1, 1))
        #Upsampling layers
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = OutputLayer(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

        
        if kwargs['ref_coder']:
            self.ref_coder = TIC(**iframe_params)
        else:
            self.ref_coder = None

        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.deform_hf_components = kwargs['deform_hf_components']
        self.hf_component_occlusion = kwargs['hf_component_occlusion']
        self.hf_filter  = kwargs['hf_filter']

    def apply_occlusion(self, inp, occlusion_map):
        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear', align_corners=True)
        return inp*occlusion_map
    
    def deform_input(self, inp, deformation):
        '''Motion compensation using bilinear interpolation'''
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def reference_ft_encoder(self, reference_frame: torch.Tensor)->tuple:
        '''Skip features extraction and latent feature embedding'''
        out = self.first(reference_frame) 
        hf_details = [] 
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            hf_details.append(out)
        return out, hf_details
    
    def base_layer_ft_encoder(self, base_layer_frame : torch.Tensor) -> torch.Tensor:
        '''Encodes the base layer frame into a latent code with dimensions
          matching the latent obtained from the reference frame'''
        out = self.base(base_layer_frame)    
        for i in range(len(self.base_down_blocks)):
            out = self.base_down_blocks[i](out)
        return out
    

    def animated_frame_decoder(self, out:torch.Tensor, hf_details:List[torch.Tensor]):
        '''Upsampling network with high frequency detail addition'''
        for i in range(len(self.up_blocks)):
            out = F.normalize(out+hf_details[i])
            out = self.up_blocks[i](out)
        out = self.final(out)
        return out

    def deform_base_layer(self, params: Dict[str, Any]) -> torch.Tensor:
        #generate a dense motion estimate between the base layer idx and the target idx
        dense_motion = self.dense_motion_network(reference_frame=params['reference'], kp_target=params['kp_cur'],
                                                    kp_reference=params['kp_prev'])
        #apply the predicted optical flow to warp the base layer frame towards the target idx
        warped = self.deform_input(params['deform_target'], dense_motion['deformation'])
        return warped

    def motion_prediction_and_compensation(self,reference_frame: torch.Tensor=None, 
                                           reference_frame_features: torch.Tensor=None,
                                            **kwargs)->tuple:
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        occlusion_map = dense_motion['occlusion_map']

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        
        if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
        reference_frame_features = reference_frame_features * occlusion_map
        
        return reference_frame_features, dense_motion

    def train_forward(self, params: Dict[str, Any])->Dict[str,torch.Tensor]:       
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': params['reference_frame'],
                    'reference_frame_features': params['ref_fts'],
                    'kp_reference':params['kp_reference'],
                    'kp_target':params['kp_target']
                }

        def_ref_fts, motion_info  = self.motion_prediction_and_compensation(**motion_pred_params)
        
        #Motion and occlusion aware detail shuttling/ Laplacian feature boosting
        hf_ref_fts_pyramid = []
        ref_fts_pyramid = params['hf_details']
        for id,ft in enumerate(ref_fts_pyramid):
            hf_fts = ft #pass the features directly
            if self.deform_hf_components:
                hf_fts = self.deform_input(hf_fts,motion_info['deformation'])
            if self.hf_component_occlusion:
                hf_fts = self.apply_occlusion(hf_fts, motion_info['occlusion_map'])
            
            if self.hf_filter  in ['1','2','3','4']:
                #use fixed filter kernels
                with torch.no_grad():
                    hf_fts = laplacian_feature_filter(ft,kernel_type=self.hf_filter)
            hf_ref_fts_pyramid.append(hf_fts)

        hf_ref_fts_pyramid = hf_ref_fts_pyramid[::-1]

        bl_fts = self.base_layer_ft_encoder(params['base_layer'])
        #Investigate the effect of base layer masking at training time

        bt_input = torch.cat((def_ref_fts,bl_fts), dim=1)
        #block-wise fusion at the generator bottleneck
        for layer in self.main_bottleneck:
            bt_input = layer(bt_input)
        bt_output = self.bt_output_layer(bt_input)

        output = {}
        output['context'] = bt_output.detach().clone()
        output['prediction'] = self.animated_frame_decoder(bt_output,hf_ref_fts_pyramid)
        output['occlusion_map'] = motion_info['occlusion_map']
        output['hf_details'] = hf_ref_fts_pyramid
        return output
    
    def forward(self, **kwargs)->Dict[str, Any]:     
        # Encoding (downsampling) part      
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        output_dict = {} 
        reference_frame = kwargs['reference']

        if self.ref_coder is not None:
            with torch.no_grad():
                dec_reference, ref_bpp, _ = self.ref_coder(reference_frame, rate_idx=kwargs['rate_idx'])
        else:
            dec_reference = reference_frame
        
        # Encoding (downsampling) part      
        ref_fts, hf_details = self.reference_ft_encoder(dec_reference)  

        
        
        output_dict.update({'reference': dec_reference})
        
        for idx in range(kwargs['num_targets']):
            params = {'reference_frame': dec_reference,
                      'ref_fts': ref_fts,
                      'hf_details': hf_details,
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}'],
                    'base_layer':kwargs[f'base_layer_{idx}']}
            

            output = self.train_forward(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict
    
    def update(self, output_dict:Dict[str,torch.Tensor], output:Dict[str,torch.Tensor],idx: int)->Dict[str,torch.Tensor]:
        for item in output:
            output_dict.update({f"{item}_{idx}": output[item]})
        return output_dict
        
    def generate_animation(self,reference_frame: torch.Tensor, 
                    kp_reference:Dict[str, torch.Tensor],
                    kp_target:Dict[str, torch.Tensor],
                    base_layer: torch.Tensor)->torch.Tensor: 
        '''Forward animation process for inference'''      
        # Encoding (downsampling) part      
        ref_fts, hf_details = self.reference_ft_encoder(reference_frame)          
        
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': reference_frame,
                    'reference_frame_features': ref_fts,
                    'kp_reference':kp_reference,
                    'kp_target':kp_target
                }

        def_ref_fts, motion_info  = self.motion_prediction_and_compensation(**motion_pred_params)
        hf_ref_fts_pyramid = []
        ref_fts_pyramid = hf_details
        for id,ft in enumerate(ref_fts_pyramid):
            hf_fts = ft #pass the features directly
            if self.deform_hf_components:
                hf_fts = self.deform_input(hf_fts,motion_info['deformation'])
            if self.hf_component_occlusion:
                hf_fts = self.apply_occlusion(hf_fts, motion_info['occlusion_map'])
            
            if self.hf_filter  in ['1','2','3','4']:
                #use fixed filter kernels
                with torch.no_grad():
                    hf_fts = laplacian_feature_filter(ft,kernel_type=self.hf_filter)
            hf_ref_fts_pyramid.append(hf_fts)

        hf_ref_fts_pyramid = hf_ref_fts_pyramid[::-1]

        bl_fts = self.base_layer_ft_encoder(base_layer)
        #Investigate the effect of base layer masking at training time

        bt_input = torch.cat((def_ref_fts,bl_fts), dim=1)
        #block-wise fusion at the generator bottleneck
        for layer in self.main_bottleneck:
            bt_input = layer(bt_input)
            
        bt_output = self.bt_output_layer(bt_input)
        return self.animated_frame_decoder(bt_output, hf_ref_fts_pyramid)
    
