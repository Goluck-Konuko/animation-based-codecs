import os
import json
import torch
import imageio
import numpy as np
from tqdm import trange
from metrics import Metrics
from utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from image_coders import ImageCoder
from coding_utils import *
from typing import Protocol
from entropy_coders import KpEntropyCoder

class Generator(Protocol):
    def forward(self):
        ...

class KPD(Protocol):
    def forward(self):
        ...

class Dataset(Protocol):
    def __getitem__(self):
        ...


import os
import json
import torch
import imageio
import numpy as np
from tqdm import trange
from metrics import Metrics
from utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from image_coders import ImageCoder
from coding_utils import *
from PIL import Image
from typing import Protocol, Dict, Any
from entropy_coders import KpEntropyCoder, ResEntropyCoder
from conventional_codecs import HEVC, VVC_VTM, VvenC


class Generator(Protocol):
    def forward(self):
        ...

class KPD(Protocol):
    def forward(self):
        ...

class AnimationCodec:
    def __init__(self,generator: Generator, kp_detector: KPD, 
                 image_coder: ImageCoder, kp_coder: KpEntropyCoder, eval_params=None) -> None:
        self.generator = generator 
        self.kp_detector = kp_detector
        self.image_coder = image_coder
        self.kp_coder = kp_coder
        self.num_frames = eval_params['num_frames']
        self.gop_size = eval_params['gop_size']
        self.eval_params = eval_params
        self.fps = eval_params['fps']
        self.video = None
        if 'bl_qp' in eval_params:
            self.base_layer_qp = eval_params['bl_qp']
        else:
            self.base_layer_qp = 50

        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.gops = []
        self.original_video = []
        self.decoded_video = []
        self.visualization = []
        self.animated_video = []
        self.scales, self.means = [], []

    def create_gops(self):
        if self.num_frames >= self.gop_size:
            num_gops = self.num_frames//self.gop_size
        else:
            num_gops = 1
        for idx in range(num_gops):
            self.gops.append(self.video[idx*self.gop_size: idx*self.gop_size+self.gop_size])


    def reset(self)-> None:
        self.num_frames= -1
        self.video = None
        
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.gops = []
        self.original_video = []
        self.decoded_video = []
        self.visualization = []
        self.animated_video = []
        self.scales, self.means = [], []

    def get_bitrate(self, fps=None):
        '''Returns the bitrate of the compressed video'''
        if fps is None:
            fps = self.fps
        return ((self.total_bits*fps)/(1000*self.num_frames))

class ConventionalCodec:
    def __init__(self, eval_params: Dict[str, Any]) -> None:
        self.codec_name =  eval_params['ref_codec']
        self.num_frames = eval_params['num_frames']
        self.gop_size = eval_params['gop_size']
        
        self.eval_params = eval_params
        self.fps = eval_params['fps']
        self.video = None
        self.qp = eval_params['qp']  

        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.gops = []
        self.original_video = []
        self.decoded_video = []
        self.visualization = []

    def create_gops(self):
        if self.num_frames >= self.gop_size:
            #subdivide the input video into gops
            num_gops = self.num_frames//self.gop_size
        else:
            num_gops = 1
        for idx in range(num_gops):
            self.gops.append(self.video[idx*self.gop_size: idx*self.gop_size+self.gop_size])

    def reset(self):
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.original_video = []
        self.decoded_video = []
        self.visualization = []
    
    def get_bitrate(self, fps=None):
        '''Returns the bitrate of the compressed video'''
        if fps is None:
            fps = self.fps
        return ((self.total_bits*fps)/(1000*self.num_frames))
    
    def run(self):
        for gop in self.gops:
            if self.codec_name =='hevc':
                info = run_hevc(gop, self.qp)
            elif self.codec_name == 'vvc':
                info = run_vvc(gop, self.qp)
            elif self.codec_name == 'vvenc':
                info = run_vvenc(gop, self.qp)
            else:
                raise NotImplementedError(f"Unimplemented codec of name: [{self.codec_name}]")
        
            self.decoded_video.extend(info['dec_frames'])
            self.original_video.extend(list(gop))
            self.total_bits += info['bitstring_size']
            self.enc_time += info['time']['enc_time']
            self.dec_time += info['time']['dec_time']

def resize_frames(frames: np.ndarray, scale_factor=1)->np.ndarray:
    N, H, W, C = frames.shape
    
    if scale_factor != 1:
        out = []
        for idx in range(N):
            img = Image.fromarray(frames[idx])
            img = img.resize((int(H*scale_factor), int(W*scale_factor)),resample=Image.Resampling.LANCZOS)
            out.append(np.asarray(img))
        return np.array(out)
    else:
        return frames

def run_hevc(gop: np.ndarray,h_qp: int, fps: float=10):
    print("Running HEVC..")
    N, h,w,_ = gop.shape
    hevc_params = {
            'qp':h_qp,
            'sequence':gop,
            'gop_size':N,
            'fps':fps,
            'frame_dim': (h,w)
        }
    encoder = HEVC(**hevc_params,)
    info_out = encoder.run()
    return info_out

def run_vvenc(gop: np.ndarray,h_qp: int, fps: float=10) -> dict:
    print("Running VvenC..")
    N,H,W,C = gop.shape
    params = {'qp': h_qp,
            'fps':fps,
            'frame_dim': f"{H}x{W}" ,
            'gop_size': N, 
            'sequence': gop,
            'out_path': 'vvc_logs/'}
    encoder = VvenC(**params)
    info_out = encoder.run()
    return info_out

def run_vvc(gop: np.ndarray,h_qp: int, fps: float=10)-> dict:
    print("Running VVC_VTM..")
    N,H,W,C = np.array(gop).shape
    # print(H, W)
    params = {'qp': h_qp,
            'fps':fps,
            'frame_dim': [H,W] ,
            'gop_size': N, 
            'n_frames':N,
            'sequence': gop}
    encoder = VVC_VTM(**params)
    info_out = encoder.run()
    return info_out
        
def update_bits_and_time(codec,info):
    codec.total_bits += info['bitstring_size']
    codec.enc_time += info['time']['enc_time']
    codec.dec_time += info['time']['dec_time']
    return codec

def to_tensor(x: np.array)->torch.Tensor:
    x = torch.from_numpy(x).unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

import torch.nn.functional as F
def rescale(frame, scale_factor=0.5):
    return F.interpolate(frame, scale_factor=(scale_factor, scale_factor),mode='bilinear', align_corners=True)


def animation_coder(codec, visualizer: Visualizer):
    '''Animation-only CODEC'''
    for gop in codec.gops:
        org_reference = frame2tensor(gop[0], cuda=False)
        codec.original_video.append(tensor2frame(org_reference))

        dec_reference_info = codec.image_coder(org_reference)
        codec = update_bits_and_time(codec, dec_reference_info)

        reference_frame = dec_reference_info['decoded']
        kp_reference = codec.kp_detector(reference_frame)
        codec.kp_coder.kp_reference = kp_reference
        
        codec.decoded_video.append(tensor2frame(reference_frame))
        codec.animated_video.append(tensor2frame(reference_frame))

        for idx in trange(1,codec.gop_size):
            target_frame = frame2tensor(gop[idx])
            codec.original_video.append(tensor2frame(target_frame))
            kp_target = codec.kp_detector(target_frame)

            kp_coding_info = codec.kp_coder.encode_kp(kp_target = kp_target)
            codec = update_bits_and_time(codec, kp_coding_info)
            
            kp_target_hat = kp_coding_info['kp_hat']

            #animation and residual coding
            anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
            animated_frame = codec.generator.animate(reference_frame, **anim_params)
            residual_frame = target_frame-animated_frame
            codec.decoded_video.append(tensor2frame(animated_frame))

            viz_params = {'reference_frame':reference_frame,
                            'target_frame':target_frame,
                            'res': residual_frame,
                            **anim_params,
                            'prediction': animated_frame}

            viz_img = visualizer.visualize(**viz_params)
            codec.visualization.append(viz_img)
    return codec

def hybrid_coder(codec, visualizer: Visualizer):
    '''
    Hybrid coding framework using deep image animation and HEVC
    '''
    for gop in codec.gops:
        org_reference = frame2tensor(gop[0], cuda=False)
        codec.original_video.extend(gop)
        
        dec_reference_info = codec.image_coder(org_reference)
        codec = update_bits_and_time(codec, dec_reference_info)

        #create base layer_stream
        #downsample the base layer video
        bl_video = gop[1:codec.num_frames]
        base_layer_info = run_hevc(bl_video, codec.base_layer_qp)
        
        #upsample the decoded base layer
        base_layer = base_layer_info['dec_frames']
        codec = update_bits_and_time(codec, base_layer_info)

        reference_frame = dec_reference_info['decoded']
        kp_reference = codec.kp_detector(reference_frame)
        codec.kp_coder.kp_reference = kp_reference
        
        codec.decoded_video.append(tensor2frame(reference_frame))
        codec.animated_video.append(tensor2frame(reference_frame))

        for idx in trange(1,codec.gop_size):
            target_frame = frame2tensor(gop[idx])
            base_layer_frame = frame2tensor(base_layer[idx-1])
            
            kp_target = codec.kp_detector(target_frame)

            kp_coding_info = codec.kp_coder.encode_kp(kp_target = kp_target)
            codec = update_bits_and_time(codec, kp_coding_info)

            kp_target_hat = kp_coding_info['kp_hat']
            #animation and residual coding
            anim_params = {'reference_frame':reference_frame,
                           'base_layer':base_layer_frame,
                           'kp_reference':kp_reference,
                           'kp_target':kp_target_hat}
            animated_frame = codec.generator.animate(anim_params)

            residual_frame = target_frame - animated_frame
            codec.decoded_video.append(tensor2frame(animated_frame))

            viz_params = {'reference_frame':reference_frame,
                            'target_frame':target_frame,
                            'res': residual_frame,
                            'prediction': animated_frame,
                            **anim_params}
            
            viz_img = visualizer.visualize(**viz_params)
            codec.visualization.append(viz_img)
    return codec

def predictive_coder(codec, visualizer: Visualizer, temporal_prediction=False):
    '''Predictive coding with deep image animation
    - Uses image animation and learned residual coding
    '''
    for gop in codec.gops:
        org_reference = frame2tensor(gop[0], cuda=False)
        codec.original_video.extend(gop)

        dec_reference_info = codec.image_coder(org_reference)
        codec = update_bits_and_time(codec, dec_reference_info)

        reference_frame = dec_reference_info['decoded']
        if torch.cuda.is_available():
            reference_frame = reference_frame.cuda()
        with torch.no_grad():
            kp_reference = codec.kp_detector(reference_frame)
            codec.kp_coder.kp_reference = kp_reference
            
            codec.decoded_video.append(tensor2frame(reference_frame))
            codec.animated_video.append(tensor2frame(reference_frame))

            prev_latent, prev_rec, kp_prev_target, prev_residual_frame = None, None, None, None
            
            for idx in trange(1,codec.num_frames):
                target_frame = frame2tensor(gop[idx])
                kp_target = codec.kp_detector(target_frame)

                kp_coding_info = codec.kp_coder.encode_kp(kp_target = kp_target)
                codec = update_bits_and_time(codec, kp_coding_info)

                kp_target_hat = kp_coding_info['kp_hat']
                #animation and residual coding
                anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
                animated_frame = codec.generator.animate(reference_frame, **anim_params)
                
                #compute frame residual
                residual_frame = target_frame-animated_frame
                eval_params = {'rate_idx': codec.eval_params['rd_point'],
                               'q_value':codec.eval_params['q_value'],
                               'use_skip': codec.eval_params['use_skip'],
                               'skip_thresh':codec.eval_params['skip_thresh']}
                if not temporal_prediction:
                    res_coding_info, skip = codec.generator.compress_spatial_residual(residual_frame,prev_latent, **eval_params)
                else:
                    if idx == 1:
                        res_coding_info, skip = codec.generator.compress_temporal_residual(residual_frame,prev_latent, **eval_params)     
                    else:
                        temporal_residual = residual_frame - prev_res_hat
                        res_coding_info, skip = codec.generator.compress_spatial_residual(temporal_residual,prev_latent, **eval_params)
                        if not skip:
                            res_coding_info['res_hat'] = res_coding_info['res_hat']+prev_res_hat
                                
                if not skip:
                    prev_latent = res_coding_info['prev_latent']
                    prev_res_hat = res_coding_info['res_hat']
                    codec = update_bits_and_time(codec, res_coding_info)
                    enh_prediction = (animated_frame + prev_res_hat).clamp(0,1)
                else:
                    enh_prediction = (animated_frame + prev_res_hat).clamp(0,1)

                codec.animated_video.append(tensor2frame(animated_frame))
                codec.decoded_video.append(tensor2frame(enh_prediction))

                viz_params = {'reference_frame':reference_frame,
                              'target_frame':target_frame,
                              'res': residual_frame,
                              'res_hat': prev_res_hat,
                              'prediction': animated_frame,
                              'enhanced_prediction':enh_prediction,
                              **anim_params}

                viz_img = visualizer.visualize(**viz_params)
                codec.visualization.append(viz_img)
    return codec

def test(config,dataset,generator, kp_detector,**kwargs ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']

    visualizer = Visualizer(**config['visualizer_params'])
    monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['temporal'])
    #get a pretrained dac_model based on current rdac config
    if model_id != 'baselines':
        pretrained_cpk_path = kwargs['checkpoint']
        rd_point = config['eval_params']['rd_point']


        if pretrained_cpk_path is not None:
            generator = load_pretrained_model(generator, path=pretrained_cpk_path, device=device)
            kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)

        generator.eval()
        temporal_prediction = False
        if 'rdac' in model_id:
            generator.sdc.update()
            if config['model_params']['generator_params']['temporal_residual_learning']:
                generator.tdc.update()
                temporal_prediction = True

        kp_detector.eval()
        if torch.cuda.is_available():
            generator = generator.cuda()
            kp_detector = kp_detector.cuda()

        reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
        motion_kp_coder = KpEntropyCoder()

        codec = AnimationCodec(generator, kp_detector, reference_image_coder, motion_kp_coder,config['eval_params'])

        all_metrics = {}
        with torch.no_grad():
            for x in dataset:
                video = x['video']
                N,h,w,c = video.shape
                n_frames = min(num_frames, N)
                #update codec params for this sequence
                codec.num_frames = n_frames
                codec.video = video
                codec.create_gops()
                name = x['name']
                out_path = os.path.join(kwargs['log_dir'],name[0].split('.')[0])
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                if model_id == 'dac':
                    codec = animation_coder(codec, visualizer)
                elif model_id == 'hdac':
                    codec = hybrid_coder(codec, visualizer)
                elif model_id == 'rdac':
                    codec = predictive_coder(codec, visualizer, temporal_prediction)
                else:
                    raise NotImplementedError(f"Codec of type <{model_id}> is not Available!")
                
                imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4",codec.decoded_video, fps=10)
                imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4",codec.visualization, fps=10)
                
                if len(codec.animated_video)== len(codec.decoded_video):
                    comp_vid = np.concatenate((np.array(codec.original_video),np.array(codec.animated_video),np.array(codec.decoded_video)), axis=2)
                else:
                    comp_vid = np.concatenate((np.array(codec.original_video),np.array(codec.decoded_video)), axis=2)
                
                imageio.mimsave(f"{out_path}/{rd_point}_anim_enh.mp4",comp_vid, fps=10)

                metrics = monitor.compute_metrics(codec.original_video,codec.decoded_video)
                metrics.update({'bitrate':codec.get_bitrate()})
                all_metrics[name[0]] = metrics
                print(metrics)
                codec.reset()
        with open(f"{kwargs['log_dir']}/metrics_{rd_point}.json", 'w') as f:
            json.dump(all_metrics, f, indent=4)
    else:
        all_metrics = {}
        for x in dataset:
            codec = ConventionalCodec(config['eval_params'])
            video = x['video']
            N,_,_,_ = video.shape
            n_frames = min(num_frames, N)
            #update codec params for this sequence
            codec.num_frames = n_frames
            codec.video = video
            name = x['name']
            out_path = os.path.join(kwargs['log_dir'],name[0].split('.')[0])
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            #run codec to compress one GOP
            codec.create_gops()
            codec.run()
            metrics = monitor.compute_metrics(codec.video[:codec.num_frames],codec.decoded_video)
            metrics.update({'bitrate':codec.get_bitrate()})
            all_metrics[name[0]] = metrics
            # break
        with open(f"{kwargs['log_dir']}/metrics_{codec.qp}.json", 'w') as f:
            json.dump(all_metrics, f, indent=4)

            
    
  

            



# class Codec:
#     def __init__(self,generator: Generator, kp_detector: KPD, 
#                  image_coder: ImageCoder, kp_coder: KpEntropyCoder) -> None:
#         self.generator = generator 
#         self.kp_detector = kp_detector
#         self.image_coder = image_coder
#         self.kp_coder = kp_coder
#         self.num_frames = -1
#         self.fps = 10
#         self.video = None
#         self.total_bits = 0
#         self.enc_time = 0
#         self.dec_time = 0

#         self.original_video = []
#         self.decoded_video = []
#         self.visualization = []
#         self.animated_video = []

#     def reset(self)-> None:
#         self.num_frames= -1
#         self.video = None
#         self.total_bits = 0
#         self.enc_time = 0
#         self.dec_time = 0

#         self.original_video = []
#         self.decoded_video = []
#         self.visualization = []
#         self.animated_video = []

#     def get_bitrate(self, fps=None):
#         '''Returns the bitrate of the compressed video'''
#         if fps is None:
#             fps = self.fps
#         return ((self.total_bits*fps)/(1000*self.num_frames))
    

# def update_bits_and_time(codec,info):
#     codec.total_bits += info['bitstring_size']
#     codec.enc_time += info['time']['enc_time']
#     codec.dec_time += info['time']['dec_time']
#     return codec

# def animation_coder(codec, visualizer: Visualizer):
#     org_reference = frame2tensor(codec.video[:,0,:,:], cuda=False)
#     codec.original_video.append(tensor2frame(org_reference))

#     dec_reference_info = codec.image_coder(org_reference)
#     codec = update_bits_and_time(codec, dec_reference_info)

#     reference_frame = dec_reference_info['decoded']
#     kp_reference = codec.kp_detector(reference_frame)
#     codec.kp_coder.kp_reference = kp_reference
    
#     codec.decoded_video.append(tensor2frame(reference_frame))
#     codec.animated_video.append(tensor2frame(reference_frame))

#     for idx in trange(1,codec.num_frames):
#         target_frame = frame2tensor(codec.video[:,idx,:,:])
#         codec.original_video.append(tensor2frame(target_frame))
#         kp_target = codec.kp_detector(target_frame)

#         kp_coding_info = codec.kp_coder.encode_kp(kp_target = kp_target)
#         codec = update_bits_and_time(codec, kp_coding_info)

#         kp_target_hat = kp_coding_info['kp_hat']
#         #animation and residual coding
#         anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
#         animated_frame = codec.generator.animate(reference_frame, **anim_params)
#         residual_frame = target_frame-animated_frame
#         codec.decoded_video.append(tensor2frame(animated_frame))

#         viz_params = {'reference_frame':reference_frame,
#                         'target_frame':target_frame,
#                         'res': residual_frame,
#                         **anim_params,
#                         'prediction': animated_frame}
        
#         viz_img = visualizer.visualize(**viz_params)
#         codec.visualization.append(viz_img)
#     return codec

# def hybrid_coder(codec, visualizer: Visualizer,method='rdac'):
#     org_reference = frame2tensor(codec.video[:,0,:,:], cuda=False)
#     codec.original_video.append(tensor2frame(org_reference))

#     dec_reference_info = codec.image_coder(org_reference)
#     codec = update_bits_and_time(codec, dec_reference_info)

#     reference_frame = dec_reference_info['decoded']
#     if torch.cuda.is_available():
#         reference_frame = reference_frame.cuda()

#     kp_reference = codec.kp_detector(reference_frame)
#     codec.kp_coder.kp_reference = kp_reference
    
#     codec.decoded_video.append(tensor2frame(reference_frame))
#     codec.animated_video.append(tensor2frame(reference_frame))

#     for idx in trange(1,codec.num_frames):
#         target_frame = frame2tensor(codec.video[:,idx,:,:])
#         codec.original_video.append(tensor2frame(target_frame))
#         kp_target = codec.kp_detector(target_frame)

#         kp_coding_info = codec.kp_coder.encode_kp(kp_target = kp_target)
#         codec = update_bits_and_time(codec, kp_coding_info)

#         kp_target_hat = kp_coding_info['kp_hat']
#         # weight_map = generate_weight_map(kp_target_hat['value'],(256,256))
        
#         #animation and residual coding
#         anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
#         animated_frame = codec.generator.animate(reference_frame, **anim_params)
#         residual_frame = target_frame-animated_frame
#         res_coding_info = codec.generator.compress_residual(residual_frame)
#         codec = update_bits_and_time(codec, res_coding_info)
#         enh_prediction = (animated_frame + res_coding_info['res_hat']).clamp(0,1)

#         codec.animated_video.append(tensor2frame(animated_frame))
#         codec.decoded_video.append(tensor2frame(enh_prediction))

#         viz_params = {'reference_frame':reference_frame,
#                         'target_frame':target_frame,
#                         'res': residual_frame,
#                         'res_hat': res_coding_info['res_hat'],
#                         **anim_params,
#                         'prediction': animated_frame,
#                         'enhanced_prediction':enh_prediction}
#         viz_img = visualizer.visualize(**viz_params)
#         codec.visualization.append(viz_img)
#     return codec

# def test(config,dataset:Dataset,generator:Generator, kp_detector:KPD,**kwargs ):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model_id = kwargs['model_id']
#     num_frames = config['eval_params']['num_frames']

#     #get a pretrained dac_model based on current rdac config
#     pretrained_cpk_path = kwargs['checkpoint']
#     rd_point = 1
#     if pretrained_cpk_path is not None:
#         generator = load_pretrained_model(generator, path=pretrained_cpk_path, device=device)
#         kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
#         rd_point = get_rd_point(pretrained_cpk_path)
#     generator.eval()
    
#     if model_id == 'rdac':
#         generator.sdc.update()
    
#     if 'rdac_t' in model_id:
#         generator.tdc.update()


#     kp_detector.eval()
#     if torch.cuda.is_available():
#         generator = generator.cuda()
#         kp_detector = kp_detector.cuda()

#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
#     visualizer = Visualizer(**config['visualizer_params'])
#     monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['temporal'])
#     reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
#     motion_kp_coder = KpEntropyCoder()

#     codec = Codec(generator, kp_detector, reference_image_coder, motion_kp_coder)

#     all_metrics = {}
#     with torch.no_grad():
#         for x in dataloader:
#             video = x['video']
#             _, N, _, _, _ = video.shape
#             n_frames = min(num_frames, N)
#             #update codec params for this sequence
#             codec.num_frames = n_frames
#             codec.video = video

#             name = x['name']
#             out_path = os.path.join(kwargs['log_dir'],name[0].split('.')[0])
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)

#             if model_id == 'dac':
#                 codec = animation_coder(codec, visualizer)
#             elif model_id == 'rdac':
#                 codec = hybrid_coder(codec, visualizer, method=model_id)

#             imageio.mimsave(f"{out_path}/enh_video.mp4",codec.decoded_video, fps=10)
#             imageio.mimsave(f"{out_path}/viz.mp4",codec.visualization, fps=10)
            
#             if len(codec.animated_video)== len(codec.decoded_video):
#                 comp_vid = np.concatenate((np.array(codec.original_video),np.array(codec.animated_video),np.array(codec.decoded_video)), axis=1)
#             else:
#                 comp_vid = np.concatenate((np.array(codec.original_video),np.array(codec.decoded_video)), axis=1)
            
#             imageio.mimsave(f"{out_path}/anim_enh.mp4",comp_vid, fps=10)
            
#             metrics = monitor.compute_metrics(codec.original_video,codec.decoded_video)
#             metrics.update({'bitrate':codec.get_bitrate()})
#             all_metrics[name[0]] = metrics
#             codec.reset()
            
#     with open(f"{kwargs['log_dir']}/metrics.json", 'w') as f:
#         json.dump(all_metrics, f, indent=4)
  

            
