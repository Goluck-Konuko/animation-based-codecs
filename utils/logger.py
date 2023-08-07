
import os
import torch
import imageio
import collections
import numpy as np
import torch.nn.functional as F
from skimage.draw import disk
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Union

def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        color_part = part * color
        parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    return result


class Logger:
    def __init__(self, log_dir: str, 
                 checkpoint_freq: int=100, 
                 zfill_num:int=8, log_file_name:str='log.txt',
                 mode:str='test'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer()
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.mode = mode
        self.epoch_losses = []

    def log_scores(self, loss_names:List[str]):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        self.epoch_losses = {y[0].replace(' ',''): float(y[1]) for y in [x.split('-') for x in loss_string.split(';')]}

        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp:Dict[str, torch.Tensor], out:Dict[str, torch.Tensor], 
                      name:str=None):
        if name:
            visualizations_dir =self.visualizations_dir+f'_{name}' 
        else:
            visualizations_dir = self.visualizations_dir
        
        if not os.path.exists(visualizations_dir):
            os.makedirs(visualizations_dir)

        viz_params = {
                **inp,
                **out
                }
        image = self.visualizer.visualize(**viz_params)

        imageio.imsave(os.path.join(visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)
        return image
    
    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-new-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None,
                 optimizer=None, optimizer_discriminator=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'], strict=False)

        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'], strict=False)

        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'], strict=False)
            
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'], strict=False)
                   
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'], strict=False)
        if 'epoch' in checkpoint:
            return checkpoint['epoch']
        else:
            return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models=None, inp=None, out=None, name=None):
        self.epoch = epoch
        if models is not None:
            self.models = models
            if (self.epoch + 1) % self.checkpoint_freq == 0:
                self.save_cpk()
        self.log_scores(self.names)
        image= self.visualize_rec(inp, out, name=name)
        return image, self.epoch_losses

from torchvision.utils import flow_to_image

class Visualizer:
    def __init__(self, kp_size: int=5, draw_border:bool=False, 
                 colormap: str='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image:np.ndarray, kp_array:np.ndarray) -> np.ndarray:
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]),self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images: np.ndarray, kp:np.ndarray)->np.ndarray:
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images:np.ndarray)->np.ndarray:
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def detach_frame(self, frame:torch.tensor, size: list=[256, 256])->np.ndarray:
        frame = F.interpolate(frame.data.cpu(), size=size).numpy()
        return np.transpose(frame, [0, 2, 3, 1])

    def visualize(self, **out)->np.ndarray:
        images = []
        # Source image with keypoints
        if 'kp_target_0' in out:
            del out['kp_target_0']
        if 'kp_target' in out:
            del out['kp_target']

        if 'reference' in out:
            reference = out['reference'].data.cpu()
            kp_reference = out['kp_reference']['value'].data.cpu().numpy()
            reference = np.transpose(reference, [0, 2, 3, 1])
            if 'kp_src' in out:
                images.append((reference, -1*out['kp_src'].data.cpu().numpy()))
            images.append((reference, kp_reference))
            B, H,W,C = reference.shape

        target_frames = sorted([name for name in out.keys() if 'target' in name])
        for idx in range(len(target_frames)):
            org_frame = self.detach_frame(out[target_frames[idx]],[H,W])
            images.append(org_frame)

            if f'base_layer_{idx}' in out:
                bl_frame = self.detach_frame(out[f'base_layer_{idx}'],[H,W])
                images.append(bl_frame)

            if f'prediction_{idx}' in out:
                anim_frame = self.detach_frame(out[f'prediction_{idx}'],[H,W])
                images.append(anim_frame)

            if f'res_{idx}' in out:
                res_frame = (self.detach_frame(out[f'res_{idx}'],[H,W])+1.0)/2.0
                images.append(res_frame)

            if f'res' in out:
                res_frame = (self.detach_frame(out[f'res'],[H,W])+1.0)/2.0
                images.append(res_frame)

            if f'res_temp_{idx}' in out:
                res_temp_frame = (self.detach_frame(out[f'res_temp_{idx}'],[H,W])+1.0)/2.0
                images.append(res_temp_frame)

            if f'res_temp_hat_{idx}' in out:
                res_temp_hat_frame = (self.detach_frame(out[f'res_temp_hat_{idx}'],[H,W])+1.0)/2.0
                images.append(res_temp_hat_frame)

            if f'res_hat_{idx}' in out:
                res_hat_frame = (self.detach_frame(out[f'res_hat_{idx}'],[H,W])+1.0)/2.0
                images.append(res_hat_frame)

            if f'res_hat' in out:
                res_hat_frame = (self.detach_frame(out['res_hat'],[H,W])+1.0)/2.0
                images.append(res_hat_frame)

            if f'enhanced_pred_{idx}' in out:
                enh_frame = self.detach_frame(out[f'enhanced_pred_{idx}'],[H,W])
                images.append(enh_frame)           

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
