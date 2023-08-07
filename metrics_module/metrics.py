import os
import torch
import imageio
import contextlib
import numpy as np
from typing import Dict, Any
from skimage import  img_as_float32
from facenet_pytorch import InceptionResnetV1
from .utils import convert_range, convert_yuvdict_to_tensor, load_image_array, write_yuv
import warnings
warnings.filterwarnings("ignore")
import math
import torch.nn.functional as F

data_range = [0, 1]

class MetricParent:
    def __init__(self, bits=8, max_val=255, mvn=1, name=''):
        self.__name = name
        self.bits = bits
        self.max_val = max_val
        self.__metric_val_number = mvn
        self.metric_name = ''

    def set_bd_n_maxval(self, bitdepth=None, max_val=None):
        if bitdepth is not None:
            self.bits = bitdepth
        if max_val is not None:
            self.max_val = max_val

    def name(self):
        return self.__name

    def metric_val_number(self):
        return self.__metric_val_number

    def calc(self, orig, rec):
        raise NotImplementedError


class PSNRMetric(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args,
                         **kwards,
                         mvn=3,
                         name=['PSNR_Y', 'PSNR_U', 'PSNR_V'])

    def calc(self, org, dec, weight=None, _lambda=1.0):
        ans = []
        for plane in org:
            a = org[plane].mul((1 << self.bits) - 1)
            b = dec[plane].mul((1 << self.bits) - 1)
            sq_diff = (a-b)**2
            if weight is not None:
                sq_diff = sq_diff*weight[:,:,0]
            mse = torch.mean(sq_diff)
            if mse == 0.0:
                ans.append(100)
            else:
                ans.append(20 * np.log10(self.max_val) - 10 * np.log10(mse))
        return float(ans[0])


class MSSSIMTorch(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (PyTorch)')

    def calc(self, org, dec):
        ans = 0.0
        from pytorch_msssim import ms_ssim
        if 'Y' not in org or 'Y' not in dec:
            return -100.0
        plane = 'Y'
        
        a = org[plane].mul((1 << self.bits) - 1)
        b = dec[plane].mul((1 << self.bits) - 1)
        a.unsqueeze_(0).unsqueeze_(0)
        b.unsqueeze_(0).unsqueeze_(0)
        ans = ms_ssim(a, b, data_range=self.max_val).item()

        return ans


class MSSSIM_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='MS-SSIM (IQA)')
        from IQA_pytorch.MS_SSIM import MS_SSIM
        self.ms_ssim = MS_SSIM(channels=1)

    def calc(self, org, dec):
        ans = 0.0
        if 'Y' not in org or 'Y' not in dec:
            return -100.0
        plane = 'Y'
        a = org[plane].unsqueeze(0).unsqueeze(0)
        b = dec[plane].unsqueeze(0).unsqueeze(0)
        ans = self.ms_ssim(a, b, as_loss=False).item()

        return ans

from psnr_hvsm import psnr_hvs_hvsm
class PSNR_HVS(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='PSNR_HVS')
        
    def pad_img(self, img, mult):
        h, w = img.shape[-2:]
        w_diff = int(math.ceil(w / mult) * mult) - w
        h_diff = int(math.ceil(h / mult) * mult) - h
        return F.pad(img, (0, w_diff, 0, h_diff), mode='replicate')

    def calc(self, orig, rec):      
        a = orig['Y']
        b = rec['Y']
        a = convert_range(a, data_range, [0, 1])
        b = convert_range(b,data_range, [0, 1])
        a_img = self.pad_img(a.unsqueeze(0).unsqueeze(0), 8).squeeze()
        b_img = self.pad_img(b.unsqueeze(0).unsqueeze(0), 8).squeeze()
        a_img = a_img.cpu().numpy().astype(np.float64)
        b_img = b_img.cpu().numpy().astype(np.float64)
        p_hvs, p_hvs_m = psnr_hvs_hvsm(a_img, b_img)
        return p_hvs


class VIF_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='VIF')
        from IQA_pytorch import VIFs
        self.vif = VIFs(channels=1)

    def calc(self, org, dec):
        ans = 0.0
        if 'Y' not in org or 'Y' not in dec:
            return -100.0
        plane = 'Y'
        b = convert_range(dec[plane].unsqueeze(0).unsqueeze(0), data_range,[0, 1])
        a = convert_range(org[plane].unsqueeze(0).unsqueeze(0), data_range,[0, 1])
        self.vif = self.vif.to(a.device)
        ans = self.vif(a, b, as_loss=False).item()
        return ans

from piq import fsim
class FSIM_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='FSIM')

    def calc(self, org: np.array, dec: np.array):  
        ans = 0.0
        # print(org.shape, dec.shape)
        org = torch.tensor(org).unsqueeze(0).permute(0,3, 1, 2)
        dec = torch.tensor(dec).unsqueeze(0).permute(0,3, 1, 2)
        ans = fsim(org, dec).item()
        return ans


class NLPD_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='NLPD')
        from IQA_pytorch import NLPD
        self.chan = 1
        self.nlpd = NLPD(channels=self.chan)

    def calc(self, org, dec):
        ans = 0.0
        if 'Y' not in org or 'Y' not in dec:
            return -100.0
        if self.chan == 1:
            plane = 'Y'
            b = org[plane].unsqueeze(0).unsqueeze(0)
            a = dec[plane].unsqueeze(0).unsqueeze(0)
        elif self.chan == 3:
            b = convert_yuvdict_to_tensor(org, org['Y'].device)
            a = convert_yuvdict_to_tensor(dec, dec['Y'].device)
        self.nlpd = self.nlpd.to(a.device)
        ans = self.nlpd(a, b, as_loss=False).item()
        return ans

class IWSSIM(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='IW-SSIM')
        from .iw_ssim import IW_SSIM
        self.iwssim = IW_SSIM()

    def calc(self, org, dec):
        ans = 0.0
        if 'Y' not in org or 'Y' not in dec:
            return -100.0
        plane = 'Y'
        # IW-SSIM takes input in a range 0-250
        a = convert_range(org[plane], data_range,[0, 255])
        b = convert_range(dec[plane],data_range,[0, 255])
        ans = self.iwssim.test(a.detach().cpu().numpy(),
                               b.detach().cpu().numpy())
        return ans.item()


class VMAF(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='VMAF')
        import platform
        if platform.system() == 'Linux':
            self.URL = 'https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf'
            self.OUTPUT_NAME = os.path.join(os.path.dirname(__file__),
                                            'vmaf.linux')
        else:
            # TODO: check that
            self.URL = 'https://github.com/Netflix/vmaf/releases/download/v2.2.1/vmaf.exe'
            self.OUTPUT_NAME = os.path.join(os.path.dirname(__file__),
                                            'vmaf.exe')

    def download(self, url, output_path):
        import requests
        r = requests.get(url, stream=True)  # , verify=False)
        if r.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)

    def check(self):
        if not os.path.exists(self.OUTPUT_NAME):
            import stat
            self.download(self.URL, self.OUTPUT_NAME)
            os.chmod(self.OUTPUT_NAME, stat.S_IEXEC)

    def calc(self, org: Dict[str, torch.Tensor], dec: Dict[str, torch.Tensor]) -> float:

        import subprocess
        import tempfile
        fp_o = tempfile.NamedTemporaryFile(delete=False)
        fp_r = tempfile.NamedTemporaryFile(delete=False)

        write_yuv(org, fp_o, self.bits)
        write_yuv(dec, fp_r, self.bits)

        out_f = tempfile.NamedTemporaryFile(delete=False)
        out_f.close()

        self.check()

        args = [
            self.OUTPUT_NAME, '-r', fp_o.name, '-d', fp_r.name, '-w',
            str(org['Y'].shape[1]), '-h',
            str(org['Y'].shape[0]), '-p', '420', '-b',
            str(self.bits), '-o', out_f.name, '--json'
        ]
        subprocess.run(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        import json
        with open(out_f.name, 'r') as f:
            tmp = json.load(f)
        ans = tmp['frames'][0]['metrics']['vmaf']

        os.unlink(fp_o.name)
        os.unlink(fp_r.name)
        os.unlink(out_f.name)

        return ans

from lpips import LPIPS
class LPIPS_IQA(MetricParent):
    def __init__(self,net='alex', *args, **kwargs):
        super().__init__(*args, **kwargs, name=net)
        if net == 'alex':
            self.lpips = LPIPS(net='alex')
        else:
            self.lpips = LPIPS(net='vgg')

        self.lpips = to_cuda(self.lpips)

    def calc(self, org: np.array, dec: np.array, weight=None):  
        ans = 0.0
        compute_weighted = weight is not None
        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        
        org = to_cuda(org)
        dec = to_cuda(dec)
        if compute_weighted:
            weight = to_cuda(weight)

        if compute_weighted:
            ans = self.lpips(org, dec, weight).item()
        else:
            ans = self.lpips(org, dec).item()  
        return ans

def to_cuda(frame):
    if torch.cuda.is_available():
        frame = frame.cuda()
    return frame

class DISTS_IQA(MetricParent):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards, name='DISTS')
        from DISTS_pytorch import DISTS
        self.dist = DISTS()
        self.dist = to_cuda(self.dist)

    def calc(self, org: torch.Tensor, dec: torch.Tensor):  
        ans = 0.0

        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        if torch.cuda.is_available():
            org = org.cuda()
            dec = dec.cuda()
        ans = self.dist(org, dec).item()
        return ans

from typing import List
import torch.nn as nn
from torchvision import models


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = F.interpolate(out, scale_factor=(self.scale, self.scale),mode='bilinear', align_corners=True)

        return out
        
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict

class MSVGG(MetricParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, name='msVGG')
        self.loss_weights = [10, 10, 10, 10, 10]
        self.scales  = [1, 0.5, 0.25,0.125]
        self.wm_scales = [1, 0.5, 0.25, 0.125, 0.0625]
        self.vgg = Vgg19()
        self.pyramid = ImagePyramide(self.scales, 3)        
        self.pyramid_f  = to_cuda(self.pyramid_f)        
        self.vgg = to_cuda(self.vgg)
        self.pyramid = to_cuda(self.pyramid)
        self.wm_pyramide = to_cuda(self.wm_pyramide)
 
    def calc(self, org: np.array, dec: np.array): 	
        org = torch.tensor(org[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        dec = torch.tensor(dec[np.newaxis].astype(np.float32)).permute(0,3, 1, 2)
        
        org = to_cuda(org)
        dec = to_cuda(dec)
        
        pyramide_real = self.pyramid(org)
        pyramide_generated = self.pyramid(dec)
        value_total = 0.0
        for scale in self.scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

            for i, _ in enumerate(self.loss_weights):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                value_total += value.item()*self.loss_weights[i]
        return value_total
 
class Metrics:
    def __init__(self,metrics: List[str] = ['psnr','psnr_hvs','ms_ssim','vif','nlpd','iw_ssim','vmaf'], temporal=False) -> None:
        
        self.metrics = metrics
        self.temporal = temporal
        if 'psnr' in self.metrics:
            self.psnr = PSNRMetric()

        if 'psnr_hvs' in self.metrics:
            self.psnr_hvs = PSNR_HVS()

        if 'lpips' in self.metrics:
            with contextlib.redirect_stdout(None):
                self.lpips = LPIPS_IQA(net='alex')

        if 'lpips_vgg' in self.metrics:
            with contextlib.redirect_stdout(None):
                self.lpips_vgg = LPIPS_IQA(net='vgg')

        if 'msVGG' in self.metrics:
            self.msVGG = MSVGG()

        if 'fsim' in self.metrics:
            self._fsim = FSIM_IQA()

        if 'nlpd' in self.metrics:
            self.nlpd = NLPD_IQA()

        if 'iw_ssim' in self.metrics:
            self.iw_ssim = IWSSIM()

        if 'ms_ssim' in self.metrics:
            self.ms_ssim = MSSSIM_IQA()

        if 'vif' in self.metrics:
            self.vif = VIF_IQA()

        if 'vmaf' in self.metrics:
            self.vmaf = VMAF()

        if 'dists' in self.metrics:
            self.dists = DISTS_IQA()
            
    def compute_metrics(self,org: List[np.array], dec: List[np.array])-> Dict[str, List[float]]:
        org_rgb = np.array(org).astype(np.uint8)
        dec_rgb = np.array(dec).astype(np.uint8)
        
        org_yuv, dec_yuv = {},{}
        
        for idx, frame in enumerate(org):
            frame_yuv = load_image_array(frame)
            dec_frame = load_image_array(dec[idx])

            org_yuv[idx] = frame_yuv
            dec_yuv[idx] = dec_frame
        all_metrics = {}
        
        org_rgb = img_as_float32(org_rgb)
        dec_rgb = img_as_float32(dec_rgb)
        
        #compute specified metrics
        if 'psnr' in self.metrics:
            s_psnr = []
            for idx in dec_yuv:
                val = self.psnr.calc(org_yuv[idx], dec_yuv[idx])
                s_psnr.append(val)
            if self.temporal:
                all_metrics['psnr'] = s_psnr
            else:
                all_metrics['psnr'] = np.mean(s_psnr)

        if 'psnr_hvs' in self.metrics:
            s_psnr_hvs = []
            
            for idx in dec_yuv:
                val = self.psnr_hvs.calc(org_yuv[idx], dec_yuv[idx])
                s_psnr_hvs.append(val)
            if self.temporal:
                all_metrics['psnr_hvs'] = s_psnr_hvs
            else:
                all_metrics['psnr_hvs'] = np.mean(s_psnr_hvs)
            

        if 'lpips' in self.metrics:
            s_lpips = []
            
            for idx, dec_frame in enumerate(dec_rgb):
                val = self.lpips.calc(org_rgb[idx], dec_frame)
                s_lpips.append(val)
            if self.temporal:
                all_metrics['lpips'] = s_lpips
            else:
                all_metrics['lpips'] =  np.mean(s_lpips)

        if 'lpips_vgg' in self.metrics:
            s_lpips = []
            
            for idx, dec_frame in enumerate(dec_rgb):
                val = self.lpips_vgg.calc(org_rgb[idx], dec_frame)
                s_lpips.append(val)
            if self.temporal:
                all_metrics['lpips_vgg'] = s_lpips
            else:
                all_metrics['lpips_vgg'] =  np.mean(s_lpips)

        if 'dists' in self.metrics:
            s_dists = []
            
            for idx, dec_frame in enumerate(dec_rgb):
                val = self.dists.calc(org_rgb[idx], dec_frame)
                s_dists.append(val)
            if self.temporal:
                all_metrics['dists'] = s_dists
            else:
                all_metrics['dists'] =  np.mean(s_dists)


        if 'msVGG' in self.metrics:
            s_msVGG = []
            for idx, dec_frame in enumerate(dec_rgb):
                val = self.msVGG.calc(org_rgb[idx], dec_frame)
                s_msVGG.append(val)
            if self.temporal:
                all_metrics['msVGG'] = s_msVGG
            else:
                all_metrics['msVGG'] =  np.mean(s_msVGG)

        if 'fsim' in self.metrics:
            s_fsim = []
            for idx, dec_frame in enumerate(dec_rgb):
                val = self._fsim.calc(org_rgb[idx], dec_frame)
                s_fsim.append(val)
            if self.temporal:
                all_metrics['fsim'] = s_fsim
            else:
                all_metrics['fsim'] =  np.mean(s_fsim)

        if 'nlpd' in self.metrics:
            s_nlpd   = []
            for idx in dec_yuv:
                val =   self.nlpd.calc(org_yuv[idx], dec_yuv[idx])
                s_nlpd.append(val)
            if self.temporal:
                all_metrics['nlpd'] = s_nlpd
            else:
                all_metrics['nlpd'] =  np.mean(s_nlpd)
        
        if 'iw_ssim' in self.metrics:
            s_iw_ssim   = []
            for idx in dec_yuv:
                val =   self.iw_ssim.calc(org_yuv[idx], dec_yuv[idx])
                s_iw_ssim.append(val)
            if self.temporal:
                all_metrics['iw_ssim'] = s_iw_ssim
            else:
                all_metrics['iw_ssim'] =  np.mean(s_iw_ssim)

        if 'ms_ssim' in self.metrics:
            s_ms_ssim   = []
            for idx in dec_yuv:
                val =   self.ms_ssim.calc(org_yuv[idx], dec_yuv[idx])
                s_ms_ssim.append(val)
            if self.temporal:
                all_metrics['ms_ssim'] = s_ms_ssim
            else:
                all_metrics['ms_ssim'] =  np.mean(s_ms_ssim)

        if 'ms_ssim_pytorch' in self.metrics:
            s_ms_ssim   = []
            for idx in dec_yuv:
                val =   self.ms_ssim_pytorch.calc(org_yuv[idx], dec_yuv[idx])
                s_ms_ssim.append(val)
            if self.temporal:
                all_metrics['ms_ssim_pytorch'] =s_ms_ssim
            else:
                all_metrics['ms_ssim_pytorch'] =  np.mean(s_ms_ssim)

        if 'vif' in self.metrics:
            s_vif   = []
            for idx in dec_yuv:
                val =   self.vif.calc(org_yuv[idx], dec_yuv[idx])
                s_vif.append(val)
            if self.temporal:
                all_metrics['vif'] = s_vif
            else:
                all_metrics['vif'] =  np.mean(s_vif)

        if 'vmaf' in self.metrics: 
            s_vmaf   = []
            for idx in dec_yuv:   
                val =   self.vmaf.calc(org_yuv[idx], dec_yuv[idx])
                s_vmaf.append(val)
            if self.temporal:
                all_metrics['vmaf'] = s_vmaf
            else:
                all_metrics['vmaf'] = np.mean(s_vmaf)
        for item in all_metrics:
            all_metrics[item]=all_metrics[item]
            if isinstance(all_metrics[item],np.ndarray):
                all_metrics[item]=all_metrics[item].tolist()
        return all_metrics

if __name__ == "__main__":
    tgt = 1
    org = imageio.mimread('videos/org.mp4', memtest=False)[:8]
    dec = imageio.mimread('videos/dec.mp4', memtest=False)

    metrics = Metrics()
    all_metrics = metrics.compute_metrics(org, dec, metrics=['psnr_hvs','ms_ssim','nlpd','vmaf'])
    print(all_metrics)
    