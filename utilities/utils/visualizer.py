import numpy as np
import torch.nn.functional as F

from skimage.draw import disk
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image

class Visualizer:
    def __init__(self, kp_size=5,frame_dim=(256,256,3), draw_border=False, colormap='gist_rainbow',region_bg_color=(0, 0, 0)):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.region_bg_color = region_bg_color
        self.h, self.w, self.c = frame_dim

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]),self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
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

    def detach_frame(self, frame):
        frame = F.interpolate(frame.data.cpu(), size=[self.h, self.w]).numpy()
        return np.transpose(frame, [0, 2, 3, 1])

    def visualize(self, **out):
        images = []
        if 'reference_frame' in out:
            source = out['reference_frame'].data.cpu()
            kp_source = out['kp_reference']['value'].data.cpu().numpy()
            source = np.transpose(source, [0, 2, 3, 1])
            if 'kp_src' in out:
                images.append((source, -1*out['kp_src'].data.cpu().numpy()))
            images.append((source, kp_source))
  
        if 'target_frame' in  out:
            driving = out['target_frame'].data.cpu()
            kp_driving = out['kp_target']['value'].data.cpu().numpy()
            driving = np.transpose(driving, [0, 2, 3, 1])
            images.append(driving)
            images.append((driving, kp_driving))

        if 'base_layer' in  out:
            driving = out['base_layer'].data.cpu()
            base_layer = np.transpose(driving, [0, 2, 3, 1])
            images.append(base_layer)

        if 'prediction' in out:
            anim_frame = self.detach_frame(out['prediction'])
            images.append(anim_frame)

        if f'res' in out:
            res_frame = (self.detach_frame(out[f'res'])+1.0)/2.0
            images.append(res_frame)

        if f'res_temp' in out:
            res_temp_frame = (self.detach_frame(out['res_temp'])+1.0)/2.0
            images.append(res_temp_frame)

        if f'res_temp_hat' in out:
            res_temp_hat_frame = (self.detach_frame(out['res_temp_hat'])+1.0)/2.0
            images.append(res_temp_hat_frame)

        if f'res_hat' in out:
            res_hat_frame = (self.detach_frame(out['res_hat'])+1.0)/2.0
            images.append(res_hat_frame)

        if f'enhanced_prediction' in out:
            enh_frame = self.detach_frame(out['enhanced_prediction'])
            images.append(enh_frame)
        

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
