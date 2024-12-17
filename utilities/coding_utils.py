import torch
import numpy as np
from typing import Union, Protocol

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


def frame2tensor(frame, cuda=True):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float()
    else:
        frame = frame.permute(0,3,1,2)
    frame = frame/255.0
    return frame

def tensor2frame(tensor):
    return (tensor.detach().cpu().squeeze().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

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