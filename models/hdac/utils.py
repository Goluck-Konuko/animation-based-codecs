import torch
import torch.nn.functional as F

#Laplacian kernels

kernel_1 = [[-1.0, -1.0, -1.0],
            [-1.0,  8.0, -1.0],
            [-1.0, -1.0, -1.0]]

kernel_2 = [[1.0, 1.0, 1.0],
            [1.0, -8.0, 1.0],
            [1.0, 1.0, 1.0]] 

kernel_3 = [[0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]]

kernel_4 =  [[0.0,  1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0,  -1.0, 0.0]]


LAPLACIAN = {'1': kernel_1,'2': kernel_2,
             '3': kernel_3,'4': kernel_4}


def laplacian_feature_filter(feature_maps, kernel_type='1'):
    """
    Apply a Laplacian high pass filter to the input feature maps using convolution.

    Args:
    feature_maps (torch.Tensor): The input tensor of feature maps with shape (N, C, H, W),
                                 where N is the batch size, C is the number of channels,
                                 H is the height, and W is the width.

    Returns:
    torch.Tensor: The high frequency components of the input feature maps.
    """
    device = feature_maps.get_device() if torch.cuda.is_available() else 'cpu'
    # Define the Laplacian kernel
    kernel = torch.tensor(LAPLACIAN[kernel_type], dtype=torch.float32).reshape(1, 1, 3, 3).to(device)

    # Ensure the kernel is compatible with the number of channels in feature_maps
    C = feature_maps.size(1)
    kernel = kernel.repeat(C, 1, 1, 1)

    # Padding to keep dimensions the same
    padding = 1

    # Apply the Laplacian high pass filter using a 2D convolution
    high_freq_components = F.conv2d(feature_maps, kernel, padding=padding, groups=C)

    return high_freq_components



