import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_utils import Hourglass,KPOut
from .train_utils import make_coordinate_grid


class KPD(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """
    def __init__(self, block_expansion=64, num_kp=10, num_channels=3, max_features=512,
                 num_blocks=3, temperature=0.1, estimate_jacobian=False, scale_factor=1,
                 pad=3,quantize=True):
        super(KPD, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = KPOut(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        self.jacobian = estimate_jacobian

        self.temperature = temperature
        self.scale_factor = scale_factor

    def downsample(self, frame):
        return F.interpolate(frame, scale_factor=(self.scale_factor, self.scale_factor),mode='bilinear', align_corners=True)

    def region2affine(self, region):
        shape = region.shape
        region = region.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], region.type()).unsqueeze_(0).unsqueeze_(0)
        kp_coord = (region * grid).sum(dim=(2, 3))
        region_params = {'value': kp_coord}

        if self.jacobian:
            mean_sub = grid - kp_coord.unsqueeze(-2).unsqueeze(-2)
            covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
            covar = covar * region.unsqueeze(-1)
            covar = covar.sum(dim=(2, 3))
            region_params['jacobian'] = covar
        return region_params

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.downsample(x)
            
        feature_map = self.predictor(x)

        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.region2affine(heatmap)
        out.update({'heatmap': heatmap})        
        return out

if __name__ == "__main__":
    img = torch.randn((1,3,256,256))
    kp_detector = KPD(estimate_jacobian=True)
    kps = kp_detector(img)
    print(kps['value'].shape, kps['kp_bits'],kps['jacobian'].shape,kps['j_bits'])