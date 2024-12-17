'''
Code from First Order Motion Model for Image Animation (FOMM) with minor updates.
'''

import torch
from torch import nn
import torch.nn.functional as F
from .nn_utils import Hourglass, Mask, OutputLayer
from .train_utils import AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionGenerator(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_reference and kp_target
    """

    def __init__(self, block_expansion=64, num_blocks=2, max_features=1024, num_kp=10, num_channels=3, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01,norm='batch',qp=False, **kwargs):
        super(DenseMotionGenerator, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = Mask(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))
        if estimate_occlusion_map:
            self.occlusion = OutputLayer(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:                               
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        self.num_levels = 5
        self.sigma = 1.5
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, self.scale_factor)


    def downsample(self, frame):
        return F.interpolate(frame, scale_factor=(self.scale_factor, self.scale_factor),mode='bilinear', align_corners=True)

    def create_heatmap_representations(self, reference_frame, kp_target, kp_reference):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = reference_frame.shape[2:]
        gaussian_target = kp2gaussian(kp_target, spatial_size=spatial_size, kp_variance=self.kp_variance)
        gaussian_reference = kp2gaussian(kp_reference, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = gaussian_target - gaussian_reference

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1]).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)
        return heatmap

    def create_sparse_motions(self, reference_frame, kp_target, kp_reference):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        bs, _, h, w = reference_frame.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_reference['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - kp_target['value'].view(bs, self.num_kp, 1, 1, 2)
        
        target_to_reference = coordinate_grid + kp_reference['value'].view(bs, self.num_kp, 1, 1, 2)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, target_to_reference], dim=1)
        return sparse_motions

    def create_deformed_reference_frame(self, reference_frame, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = reference_frame.shape
        reference_repeat = reference_frame.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        reference_repeat = reference_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(reference_repeat, sparse_motions, align_corners=True)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def forward(self, reference_frame, kp_target, kp_reference):
        if self.scale_factor != 1:
           reference_frame = self.down(reference_frame)

        bs, _, h, w = reference_frame.shape
        out_dict = {}
        heatmap_representation = self.create_heatmap_representations(reference_frame, kp_target, kp_reference)
        out_dict['heatmap'] = torch.sum(heatmap_representation, dim=1)
        sparse_motion = self.create_sparse_motions(reference_frame, kp_target, kp_reference)
        out_dict['sparse_motion'] = torch.sum(sparse_motion, dim=1).permute(0,3,1,2)
        deformed_reference = self.create_deformed_reference_frame(reference_frame, sparse_motion)
        out_dict['sparse_deformed'] = deformed_reference

        inp = torch.cat([heatmap_representation, deformed_reference], dim=2)
        inp = inp.view(bs, -1, h, w)

        prediction = self.hourglass(inp)

        mask = self.mask(prediction)
        out_dict['mask'] = mask.sum(dim=1)
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['deformation'] = deformation
        if self.occlusion:
            occlusion_map = self.occlusion(prediction)
            out_dict['occlusion_map'] = occlusion_map
        return out_dict

if __name__ == "__main__":
    from thop import profile
    img = torch.randn((1,3,64,64))

    kp = {'value': torch.randn((1,10,2))}
    dmg = DenseMotionGenerator()

    # out = kp_detector(img)
    # print(out['value'].shape)
    macs, params = profile(dmg, inputs=(img,kp,kp))
    print("Macs: ",macs/1e9, " GMACs | #Params: ", params/1e6, " M")