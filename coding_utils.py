import torch
import numpy as np
from typing import Union, Protocol
from models.modules.train_utils import make_coordinate_grid

class Generator(Protocol):
    def forward(self):
        ...

class KPD(Protocol):
    def forward(self):
        ...


def load_pretrained_model(model: Union[Generator, KPD], path:str ,name: str='generator', device:str='cpu'):
    cpk = torch.load(path, map_location=device)
    model.load_state_dict(cpk[name], strict=True)
    return model

def frame2tensor(frame: Union[np.ndarray, torch.Tensor], cuda: bool=True) ->torch.Tensor:
    '''0-255 [1,H,W,C] -> 0-1 [1,C,H,W]'''
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).permute(0,3,1,2).float()
    frame = frame.permute(0,3,1,2)/255.0
    if torch.cuda.is_available() and cuda:
        frame = frame.cuda()
    return frame

def tensor2frame(tensor: torch.Tensor) -> np.ndarray:
    '''0-1 (1,C,H,W) -> 0-255 (H,W,C) -> '''
    return (tensor.detach().cpu().squeeze().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)

def kp2gaussian(mean, spatial_size, kp_variance=0.01):
    """
    Transform a keypoint into gaussian like representation
    """
    # mean = kp['value']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def get_rd_point(path):
    pth = path.split('/')[-1].split('.')[0]
    if 'rd' in pth:
        rd_pt = int(pth.split('_')[-1])
    else:
        rd_pt = 1
    return rd_pt