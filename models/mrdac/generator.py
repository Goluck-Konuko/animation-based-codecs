import torch
import numpy as np
from torch import nn
from typing import Dict, Any
import torch.nn.functional as F
from ..common.dense_motion_generator import DenseMotionGenerator
from ..common.nn_utils import ResBlock2d, SameBlock2d,UpBlock2d, DownBlock2d
from ..common.train_utils import MRContrastiveLoss
from ..common.transformer_image_codec import TIC


class MRDAC(nn.Module):
    """
    Multi-reference Animation framework trained with a contrastive loss function.
    
    "Konuko et al., “Multi-Reference Generative Face Video Compression with Contrastive Learning”.MMSP 2024"
    """
    def __init__(self, num_channels:int =3, num_kp:int=10, block_expansion:int=64, max_features:int=1024, num_down_blocks:int=2,
                 num_bottleneck_blocks:int=3, estimate_occlusion_map:int=False,dense_motion_params:Dict[str,Any]=None,iframe_params:Dict[str,Any]=None, **kwargs):
        super(MRDAC, self).__init__()
        if dense_motion_params:
            self.dense_motion_network = DenseMotionGenerator(num_kp=num_kp, num_channels=num_channels,
                                                            estimate_occlusion_map=estimate_occlusion_map,
                                                            **dense_motion_params)
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
        self.bottleneck.add_module('out',SameBlock2d(in_features, in_features , kernel_size=(3, 3), padding=(1, 1),))
         
        self.aggregation = torch.nn.Sequential()
        self.aggregation.add_module('r_1', ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        self.aggregation.add_module('r_2', ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
            
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))

        if kwargs['ref_coder']:
            self.ref_coder = TIC(**iframe_params)  
        else:
            self.ref_coder = None

        if kwargs['use_contrastive_loss']:
            self.cl_criterion = MRContrastiveLoss()

        self.contrastive_loss = kwargs['use_contrastive_loss']
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.use_ref_weights = kwargs['use_ref_weights']

    def deform_input(self, inp: torch.Tensor, deformation: torch.Tensor)->torch.Tensor:
        '''Motion compensation using bilinear interpolation'''
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        def_fts = F.grid_sample(inp, deformation, align_corners=True)
        return def_fts

    def reference_ft_encoder(self, reference_frame: torch.Tensor)-> torch.Tensor:
        out = self.first(reference_frame)        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        return out

    def animated_frame_decoder(self, out: torch.Tensor)-> torch.Tensor:
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        return torch.sigmoid(out)

    def motion_prediction_and_compensation(self,reference_frame: torch.Tensor=None, 
                                           reference_frame_features: torch.Tensor=None,
                                            **kwargs)->tuple:
        dense_motion = self.dense_motion_network(reference_frame=reference_frame, kp_target=kwargs['kp_target'],
                                                    kp_reference=kwargs['kp_reference'])
        

        reference_frame_features = self.deform_input(reference_frame_features, dense_motion['deformation'])
        if 'occlusion_map' in dense_motion:
            occlusion_map = dense_motion['occlusion_map']
            if reference_frame_features.shape[2] != occlusion_map.shape[2] or reference_frame_features.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=reference_frame_features.shape[2:], mode='bilinear', align_corners=True)
            reference_frame_features = reference_frame_features * occlusion_map

        return reference_frame_features, dense_motion

    def compute_ref_weights(self, frame_indices: torch.Tensor)->torch.Tensor:
        N, _ = frame_indices.shape
        ref_weights = []
        for it in range(N):
            w_vector = frame_indices[it,:]
            w_vector = torch.abs(w_vector[:-1] - w_vector[-1])
            w_vector = (1 - w_vector/torch.max(w_vector))+0.5
            w_vector = w_vector/torch.max(w_vector)
            ref_weights.append(w_vector)
        
        ref_weights = torch.stack(ref_weights)
        return ref_weights
    
    def forward(self, **kwargs)->Dict[str, Any]:     
        # Encoding (downsampling) part  
        num_references = kwargs['num_references']
        N,_,_,_ = kwargs['target_0'].shape
        output = {}
        ref_fts_list = []
        rate = 0.0
        contrastive_loss = 0.0

        ref_weights = self.compute_ref_weights(kwargs['rf_weights'])

        for idx in range(num_references):
            #compress the reference frame using the pretrained image codec
            if self.ref_coder is not None:
                with torch.no_grad():
                    dec_reference, ref_bpp, _ = self.ref_coder(kwargs[f'reference_{idx}'], rate_idx=kwargs['rate_idx'])
            else:
                dec_reference = kwargs[f'reference_{idx}']
            ref_ft = self.reference_ft_encoder(dec_reference)
            ref_fts_list.append(ref_ft)
                    
            output[f'reference_{idx}'] = dec_reference
            rate += ref_bpp
            

        reference_features = []
        for idx in range(num_references):
            ref_fts = ref_fts_list[idx]
            motion_pred_params = { 
                        'reference_frame': output[f'reference_{idx}'], #make sure these are the decoded reference features
                        'reference_frame_features': ref_fts,
                        'kp_reference':kwargs[f'kp_reference_{idx}'],
                        'kp_target':kwargs[f'kp_target_0']}

            out_ref_fts, dense_motion_params  = self.motion_prediction_and_compensation(**motion_pred_params)
            #pass deformed reference features through 2 Res blocks before aggregation
            out_ref_fts = self.aggregation(out_ref_fts)
                    
            if idx>0 and self.contrastive_loss:
                #compute contrastive loss between the reference features
                contrastive_loss += self.cl_criterion(reference_features[idx-1],out_ref_fts)

            if self.use_ref_weights:
                r_weights = ref_weights[:,idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                out_ref_fts = out_ref_fts*r_weights

            reference_features.append(out_ref_fts)
            output[f'occlusion_map_{idx}'] = dense_motion_params['occlusion_map']
        
        ### Feature aggregation strategies
        #Permutation invariant maxpooling
        ref_features = reference_features[0]
        for idx in range(1,num_references):
            next_fts = reference_features[idx]
            ref_features = torch.max(ref_features,next_fts)

        # Decoding part
        out = self.bottleneck(ref_features)
        animated_frame = self.animated_frame_decoder(out)
        output[f"prediction_0"] = animated_frame
        output[f'target_0'] = kwargs[f'target_0']

        if self.contrastive_loss:
            output['contrastive_loss'] = contrastive_loss
        return output
    
    def update(self, output_dict, output,idx=0):
        for item in output:
            output_dict.update({f"{item}_{idx}": output[item]})
        return output_dict


    def generate_animation(self, params: Dict[str, Any]) -> torch.Tensor: 
        '''The forward animation process at inference time'''    
        # Encoding (downsampling) part    
        ref_weights = None
        if 'ref_indices' in params:
            ref_weights = self.compute_ref_weights(params['ref_indices'])
            # print(ref_weights)
        
        ref_fts = []
        for i, r_idx in enumerate(params['reference_info']):
            ref_frame,ref_ft, kp_ref = params['reference_info'][r_idx]
            # r_fts = self.reference_ft_encoder(ref_frame)
            # Transforming feature representation according to deformation and occlusion
            motion_pred_params = { 
                        'reference_frame': ref_frame,
                        'reference_frame_features': ref_ft,
                        'kp_reference':kp_ref,
                        'kp_target':params['kp_target']
                    }
            def_ref_fts, _ = self.motion_prediction_and_compensation(**motion_pred_params)
            def_ref_fts = self.aggregation(def_ref_fts)
            if ref_weights is not None:
                r_weights = ref_weights[:,i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                def_ref_fts = def_ref_fts*r_weights
            ref_fts.append(def_ref_fts)
        
        reference_features = ref_fts[0]
        for fts in ref_fts[1:]:
            reference_features = torch.max(reference_features,fts).float()

        out = self.bottleneck(reference_features)
        #reconstruct the animated frame
        return self.animated_frame_decoder(out)

# if __name__ == "__main__":
#     generator = MRDAC(block_expansion=16,dense_motion_params=None)
#     img = torch.rand(1,3,256,256)
#     num_params = sum(p.numel() for p in generator.parameters())/1e6
#     print(num_params)


