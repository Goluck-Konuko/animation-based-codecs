import torch
import numpy as np
from typing import Dict


data_range = [0, 1]

def load_image_array(rgb_data,color_conv='709',def_bits=8, device='cpu'):
    rgb_data = torch.tensor(rgb_data,dtype=torch.float,device=device).permute(2, 0, 1)
    rgb_data = convert_and_round_plane(rgb_data, [0, 255], data_range,def_bits).unsqueeze(0)
    yuv_t = rgb_to_yuv(rgb_data, color_conv).clamp(min(data_range),
                                                max(data_range))
    yuv_t = round_plane(yuv_t, def_bits)
    yuv_data = {
                'Y': yuv_t[0, 0],
                'U': yuv_t[0, 1],
                'V': yuv_t[0, 2]
                }
    return yuv_data

def load_image(filename,color_conv='709',def_bits=8, device='cpu'):
    from PIL import Image
    with Image.open(filename) as im:
        mode = im.mode
        rgb_data = np.array(im.convert('RGB'))
    if def_bits == -1:
        if ';' in mode:
            # TODO: Check support of this feature
            s_tmp = mode.split(';')
            def_bits = int(s_tmp[1])
        else:
            def_bits = 8
    rgb_data = torch.tensor(rgb_data,dtype=torch.float,device=device).permute(2, 0, 1)
    rgb_data = convert_and_round_plane(rgb_data, [0, 255], data_range,def_bits).unsqueeze(0)
    yuv_t = rgb_to_yuv(rgb_data, color_conv).clamp(min(data_range),
                                                max(data_range))
    yuv_t = round_plane(yuv_t, def_bits)
    shape = yuv_t.shape[-2:]
    bitdepth_ans = def_bits
    yuv_data = {
                'Y': yuv_t[0, 0],
                'U': yuv_t[0, 1],
                'V': yuv_t[0, 2]
                }
    return yuv_data, rgb_data


def round_plane(plane, bits):
        return plane.mul((1 << bits) - 1).round().div((1 << bits) - 1)

def convertup_and_round_plane(plane, cur_range, new_range, bits):
        return convert_range(plane, cur_range, new_range).mul((1 << bits) - 1).round()

def convert_and_round_plane(plane, cur_range, new_range, bits):
        return round_plane(convert_range(plane, cur_range, new_range), bits)

def convert_range(plane, cur_range, new_range=[0, 1]):
    if cur_range[0] == new_range[0] and cur_range[1] == new_range[1]:
        return plane
    return (plane + cur_range[0]) * (new_range[1] - new_range[0]) / (
        cur_range[1] - cur_range[0]) - new_range[0]

def convert_yuvdict_to_tensor(yuv, device='cpu'):
        size = yuv['Y'].shape
        c = len(yuv)
        ans = torch.zeros((1, c, size[-2], size[-1]),
                          dtype=torch.float,
                          device=torch.device(device))
        ans[:, 0, :, :] = yuv['Y']
        ans[:, 1, :, :] = yuv['U'] if 'U' in yuv else yuv['Y']
        ans[:, 2, :, :] = yuv['V'] if 'V' in yuv else yuv['Y']
        return ans

def color_conv_matrix(color_conv='709'):
    if color_conv == '601':
        # BT.601
        a = 0.299
        b = 0.587
        c = 0.114
        d = 1.772
        e = 1.402
    elif color_conv == '709':
        # BT.709
        a = 0.2126
        b = 0.7152
        c = 0.0722
        d = 1.8556
        e = 1.5748
    elif color_conv == '2020':
        # BT.2020
        a = 0.2627
        b = 0.6780
        c = 0.0593
        d = 1.8814
        e = 1.4747
    else:
        raise NotImplementedError

    return a, b, c, d, e


def rgb_to_yuv(image: torch.Tensor, color_conv='709') -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            'Input size must have a shape of (*, 3, H, W). Got {}'.format(
                image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    a1, b1, c1, d1, e1 = color_conv_matrix(color_conv)

    y: torch.Tensor = a1 * r + b1 * g + c1 * b
    u: torch.Tensor = (b - y) / d1 + 0.5
    v: torch.Tensor = (r - y) / e1 + 0.5

    out: torch.Tensor = torch.stack([y, u, v], -3)

    return out

def yuv_to_rgb(image: torch.Tensor, color_conv='709') -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YUV Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = yuv_to_rgb(input)  # 2x3x4x5

    Took from https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html#rgb_to_yuv
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            'Input size must have a shape of (*, 3, H, W). Got {}'.format(
                image.shape))

    y: torch.Tensor = image[..., 0, :, :]
    u: torch.Tensor = image[..., 1, :, :] - 0.5
    v: torch.Tensor = image[..., 2, :, :] - 0.5

    a, b, c, d, e = color_conv_matrix(color_conv)

    r: torch.Tensor = y + e * v  # coefficient for g is 0
    g: torch.Tensor = y - (c * d / b) * u - (a * e / b) * v
    b: torch.Tensor = y + d * u  # coefficient for b is 0

    out: torch.Tensor = torch.stack([r, g, b], -3)

    return out

def write_yuv(yuv: Dict[str, torch.Tensor], f: str, bits: int=8):
    """
    dump a yuv file to the provided path
    @path: path to dump yuv to (file must exist)
    @bits: bitdepth
    @frame_idx: at which idx to write the frame (replace), -1 to append
    """
    nr_bytes = np.ceil(bits / 8)
    if nr_bytes == 1:
        data_type = np.uint8
    elif nr_bytes == 2:
        data_type = np.uint16
    elif nr_bytes <= 4:
        data_type = np.uint32
    else:
        raise NotImplementedError(
            'Writing more than 16-bits is currently not supported!')

    # rescale to range of bits
    for plane in yuv:
        yuv[plane] = convertup_and_round_plane(yuv[plane], data_range, data_range,bits).cpu().numpy()

    # dump to file
    lst = []
    for plane in ['Y', 'U', 'V']:
        if plane in yuv.keys():
            lst = lst + yuv[plane].ravel().tolist()

    raw = np.array(lst)

    raw.astype(data_type).tofile(f)

