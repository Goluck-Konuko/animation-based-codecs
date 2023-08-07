import os
import json
import torch
import imageio
import numpy as np
from tqdm import trange
from metrics_module import Metrics
from utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from image_coders import ImageCoder
from coding_utils import *
from typing import Protocol
from entropy_coders import KpEntropyCoder, ResEntropyCoder

class Generator(Protocol):
    def forward(self):
        ...

class KPD(Protocol):
    def forward(self):
        ...

class Dataset(Protocol):
    def __getitem__(self):
        ...


class Codec:
    def __init__(self,generator: Generator, kp_detector: KPD, 
                 image_coder: ImageCoder, kp_coder: KpEntropyCoder) -> None:
        self.generator = generator 
        self.kp_detector = kp_detector
        self.image_coder = image_coder
        self.kp_coder = kp_coder
        self.num_frames = -1
        self.fps = 10
        self.video = None
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.original_video = []
        self.decoded_video = []
        self.visualization = []
        self.animated_video = []

    def reset(self)-> None:
        self.num_frames= -1
        self.video = None
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.original_video = []
        self.decoded_video = []
        self.visualization = []
        self.animated_video = []

    def get_bitrate(self, fps=None):
        '''Returns the bitrate of the compressed video'''
        if fps is None:
            fps = self.fps
        return ((self.total_bits*fps)/(1000*self.num_frames))
    

def update_bits_and_time(codec,info):
    codec.total_bits += info['bitstring_size']
    codec.enc_time += info['time']['enc_time']
    codec.dec_time += info['time']['dec_time']
    return codec

def animation_coder(codec, visualizer: Visualizer):
    org_reference = frame2tensor(codec.video[:,0,:,:], cuda=False)
    codec.original_video.append(tensor2frame(org_reference))

    dec_reference_info = codec.image_coder(org_reference)
    codec = update_bits_and_time(codec, dec_reference_info)

    reference_frame = dec_reference_info['decoded']
    kp_reference = codec.kp_detector(reference_frame)
    codec.kp_coder.kp_reference = kp_reference
    
    codec.decoded_video.append(tensor2frame(reference_frame))
    codec.animated_video.append(tensor2frame(reference_frame))

    for idx in trange(1,codec.num_frames):
        target_frame = frame2tensor(codec.video[:,idx,:,:])
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

def hybrid_coder(codec, visualizer: Visualizer,method='rdac'):
    org_reference = frame2tensor(codec.video[:,0,:,:], cuda=False)
    codec.original_video.append(tensor2frame(org_reference))

    dec_reference_info = codec.image_coder(org_reference)
    codec = update_bits_and_time(codec, dec_reference_info)

    reference_frame = dec_reference_info['decoded']
    if torch.cuda.is_available():
        reference_frame = reference_frame.cuda()

    kp_reference = codec.kp_detector(reference_frame)
    codec.kp_coder.kp_reference = kp_reference
    
    codec.decoded_video.append(tensor2frame(reference_frame))
    codec.animated_video.append(tensor2frame(reference_frame))

    for idx in trange(1,codec.num_frames):
        target_frame = frame2tensor(codec.video[:,idx,:,:])
        codec.original_video.append(tensor2frame(target_frame))
        kp_target = codec.kp_detector(target_frame)

        kp_coding_info = codec.kp_coder.encode_kp(kp_target = kp_target)
        codec = update_bits_and_time(codec, kp_coding_info)

        kp_target_hat = kp_coding_info['kp_hat']
        # weight_map = generate_weight_map(kp_target_hat['value'],(256,256))
        
        #animation and residual coding
        anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
        animated_frame = codec.generator.animate(reference_frame, **anim_params)
        residual_frame = target_frame-animated_frame
        res_coding_info = codec.generator.compress_residual(residual_frame)
        codec = update_bits_and_time(codec, res_coding_info)
        enh_prediction = (animated_frame + res_coding_info['res_hat']).clamp(0,1)

        codec.animated_video.append(tensor2frame(animated_frame))
        codec.decoded_video.append(tensor2frame(enh_prediction))

        viz_params = {'reference_frame':reference_frame,
                        'target_frame':target_frame,
                        'res': residual_frame,
                        'res_hat': res_coding_info['res_hat'],
                        **anim_params,
                        'prediction': animated_frame,
                        'enhanced_prediction':enh_prediction}
        viz_img = visualizer.visualize(**viz_params)
        codec.visualization.append(viz_img)
    return codec

def test(config,dataset:Dataset,generator:Generator, kp_detector:KPD,**kwargs ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']

    #get a pretrained dac_model based on current rdac config
    pretrained_cpk_path = kwargs['checkpoint']
    rd_point = 1
    if pretrained_cpk_path is not None:
        generator = load_pretrained_model(generator, path=pretrained_cpk_path, device=device)
        kp_detector = load_pretrained_model(kp_detector, path=pretrained_cpk_path,name='kp_detector',device=device)
        rd_point = get_rd_point(pretrained_cpk_path)
    generator.eval()
    
    if model_id == 'rdac':
        generator.sdc.update()
    
    if 'rdac_t' in model_id:
        generator.tdc.update()


    kp_detector.eval()
    if torch.cuda.is_available():
        generator = generator.cuda()
        kp_detector = kp_detector.cuda()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=True)
    visualizer = Visualizer(**config['visualizer_params'])
    monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['temporal'])
    reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
    motion_kp_coder = KpEntropyCoder()

    codec = Codec(generator, kp_detector, reference_image_coder, motion_kp_coder)

    all_metrics = {}
    with torch.no_grad():
        for x in dataloader:
            video = x['video']
            _, N, _, _, _ = video.shape
            n_frames = min(num_frames, N)
            #update codec params for this sequence
            codec.num_frames = n_frames
            codec.video = video

            name = x['name']
            out_path = os.path.join(kwargs['log_dir'],name[0].split('.')[0])
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            if model_id == 'dac':
                codec = animation_coder(codec, visualizer)
            elif model_id == 'rdac':
                codec = hybrid_coder(codec, visualizer, method=model_id)

            imageio.mimsave(f"{out_path}/enh_video.mp4",codec.decoded_video, fps=10)
            imageio.mimsave(f"{out_path}/viz.mp4",codec.visualization, fps=10)
            
            if len(codec.animated_video)== len(codec.decoded_video):
                comp_vid = np.concatenate((np.array(codec.original_video),np.array(codec.animated_video),np.array(codec.decoded_video)), axis=1)
            else:
                comp_vid = np.concatenate((np.array(codec.original_video),np.array(codec.decoded_video)), axis=1)
            
            imageio.mimsave(f"{out_path}/anim_enh.mp4",comp_vid, fps=10)
            
            metrics = monitor.compute_metrics(codec.original_video,codec.decoded_video)
            metrics.update({'bitrate':codec.get_bitrate()})
            all_metrics[name[0]] = metrics
            codec.reset()
            
    with open(f"{kwargs['log_dir']}/metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=4)
  

            
