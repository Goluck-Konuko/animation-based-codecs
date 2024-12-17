import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any
from ..common.nn_utils import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, OutputLayer
from ..common.dense_motion_generator import DenseMotionGenerator
from ..common.transformer_image_codec import TIC
  
class DAC_Generator(nn.Module):
    """
    The DAC motion transfer network: Adapted from FOMM to work with quantization-aware zero order motion representation
    (Keypoints only and no Jacobians).
    Minor updates to the network layers such as BatchNorm and activation layers

    "Konuko et al., “Ultra-low bitrate video conferencing using deep image animation”.ICASSP 2021"
    
    """
    def __init__(self, num_channels:int=3, num_kp:int=10, block_expansion:int=64, max_features:int=1024, num_down_blocks:int=2,
                 num_bottleneck_blocks:int=3, estimate_occlusion_map:bool=False, dense_motion_params:Dict[str, Any]=None,iframe_params:Dict[str, Any]=None,  **kwargs):
        super(DAC_Generator, self).__init__()
        if dense_motion_params:
            self.dense_motion_network = DenseMotionGenerator(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,**dense_motion_params)
        else:
            self.dense_motion_network = None
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

    def animated_frame_decoder(self, out: torch.Tensor)->torch.Tensor:
        '''Frame generation from the transformed latent representation'''
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return out

    def motion_prediction_and_compensation(self,reference_frame: torch.Tensor=None, 
                                           reference_frame_features: torch.Tensor=None,
                                            **kwargs)->tuple:
        '''Dense motion prediction from the input reference frame and the sparse motion keypoints.
        ->Latent space navigation through warping and application of the occlusion mask.
        '''
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        occlusion_map = dense_motion['occlusion_map']

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        
        if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
        reference_frame_features = reference_frame_features * occlusion_map
        
        return reference_frame_features, dense_motion

    
    def animate_training(self, params) -> Dict[str, torch.Tensor]:       
        '''The actual forward animation method at training time'''
        # Transforming feature representation according to deformation and occlusion
        output = {}
        motion_pred_params = { 
                    'reference_frame': params['reference_frame'],
                    'reference_frame_features': params['ref_fts'],
                    'kp_reference':params['kp_reference'],
                    'kp_target':params['kp_target']
                }

        def_ref_fts, dense_motion_params  = self.motion_prediction_and_compensation(**motion_pred_params)

        # Decoding part
        out = self.bottleneck(def_ref_fts)
        output["prediction"] = self.animated_frame_decoder(out)
        return output
    
    def forward(self, **kwargs)->Dict[str, Any]:     
        # Encoding (downsampling) part 
        output_dict = {} 
        reference_frame = kwargs['reference']
        if self.ref_coder is not None:
            with torch.no_grad():
                dec_reference, ref_bpp, _ = self.ref_coder(reference_frame, rate_idx=kwargs['rate_idx'])
        else:
            dec_reference = reference_frame
        
        output_dict.update({'reference': dec_reference})
        ref_fts = self.reference_ft_encoder(dec_reference)
        # Transforming feature representation according to deformation and occlusion
        # Previous frame animation
        
        for idx in range(kwargs['num_targets']):
            params = {'reference_frame': reference_frame,
                    'ref_fts': ref_fts,
                    'target_frame':kwargs[f'target_{idx}'],
                    'kp_reference': kwargs['kp_reference'],
                    'kp_target': kwargs[f'kp_target_{idx}'],
                    'rate_idx': kwargs[f'rate_idx'],
                    }
            
            output = self.animate_training(params)
            output_dict = self.update(output_dict, output, idx)
        return output_dict

    
    def update(self, output_dict:Dict[str, torch.Tensor], output:Dict[str, torch.Tensor],idx:int):
        for item in output:
            output_dict.update({f"{item}_{idx}": output[item]})
        return output_dict

    def generate_animation(self, reference_frame: torch.Tensor, 
                    kp_reference:Dict[str, torch.Tensor],
                    kp_target:Dict[str, torch.Tensor]) -> torch.Tensor: 
        '''The forward animation process at inference time'''    
        # Encoding (downsampling) part      
        ref_fts = self.reference_ft_encoder(reference_frame)
        # Transforming feature representation according to deformation and occlusion
        motion_pred_params = { 
                    'reference_frame': reference_frame,
                    'reference_frame_features': ref_fts,
                    'kp_reference':kp_reference,
                    'kp_target':kp_target
                }
        def_ref_fts, _ = self.motion_prediction_and_compensation(**motion_pred_params)
        # Decoding bottleneck layer
        out_ft_maps = self.bottleneck(def_ref_fts) #input the weighted average 
        #reconstruct the animated frame
        return self.animated_frame_decoder(out_ft_maps)



if __name__ == "__main__":
    from thop import profile
    img = torch.randn((1,3,256,256))

    kp = {'value': torch.randn((1,10,2))}
    mtn = DAC_Generator()

    # out = kp_detector(img)
    # print(out['value'].shape)
    macs, params = profile(mtn, inputs=(img,kp,kp))
    print("Macs: ",macs/1e9, " GMACs | #Params: ", params/1e6, " M")
