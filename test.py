import os
import json
import torch
import imageio
import numpy as np
from tqdm import trange
# from metrics import Metrics
from utilities.utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from utilities.image_coders import ImageCoder
from utilities.entropy_coders import KpEntropyCoder
from utilities.coding_utils import *
from typing import Protocol


class animation_model(Protocol):
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
from utilities.metrics import Metrics
from utilities.utils.visualizer import Visualizer
from torch.utils.data import DataLoader
from utilities.image_coders import ImageCoder
from utilities.coding_utils import *
from PIL import Image
from typing import Protocol, Dict, Any
from utilities.entropy_coders import KpEntropyCoder
from utilities.anchors import HEVC #, VVC_VTM, VvenC


class animation_model(Protocol):
    def forward(self):
        ...

class KPD(Protocol):
    def forward(self):
        ...

class KPD(Protocol):
    def forward(self):
        ...

class Models:
    def __init__(self,animation_model: Generator, kp_detector_model: KPD, 
                 kp_coder: KpEntropyCoder) -> None:
        self.animation_model = animation_model 
        self.kp_detector_model = kp_detector_model

        #Entropy coders
        self.kp_coder = kp_coder
        self.out_path = None

class Inputs:
    def __init__(self,eval_params=None) -> None:
        self.num_frames = eval_params['num_frames']
        self.gop_size = eval_params['gop_size']
        self.eval_params = eval_params
        self.fps = eval_params['fps']
        self.device = 'cpu'
        self.video = None
        if 'bl_qp' in eval_params:
            self.base_layer_qp = eval_params['bl_qp']
        else:
            self.base_layer_qp = 50

        self.gops = []
        self.original_video = []

    def create_gops(self):
        if self.num_frames >= self.gop_size:
            num_gops = self.num_frames//self.gop_size
        else:
            num_gops = 1
        for idx in range(num_gops):
            self.gops.append(self.video[idx*self.gop_size: idx*self.gop_size+self.gop_size])

def tensor2rgb(tensor):
    '''1x3xHxW ->1x3xHxW (unit8)'''
    return (tensor.detach().cpu().squeeze().numpy() * 255.0).astype(np.uint8)


class Outputs:
    def __init__(self,out_path = 'results') -> None:
        self.total_bits = 0
        self.enc_time = 0
        self.dec_time = 0

        self.decoded_video = []
        self.visualization = []

        self.f_dec=open(f"{out_path}/decoded.rgb",'w') 

    def update_decoded(self, dec_frame):
        chw = tensor2rgb(dec_frame)
        chw.tofile(self.f_dec)
        hwc = np.transpose(chw, [1,2,0])
        self.decoded_video.append(hwc)

    def update_bits_and_time(self, info):
        self.total_bits += info['bitstring_size']
        self.enc_time += info['time']['enc_time']
        self.dec_time += info['time']['dec_time']

    def get_bitrate(self, fps:float, num_frames: int):
        '''Returns the bitrate of the compressed video'''
        return ((self.total_bits*fps)/(1000*num_frames))


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

def animation_coder(models, input_data,output_data,  visualizer: Visualizer):
    for gop in input_data.gops:
        org_reference = frame2tensor(gop[0]).to(input_data.device)
        input_data.original_video.extend(gop)

        dec_reference_info = models.animation_model.ref_coder.compress(org_reference,rate_idx=input_data.eval_params['qp'])
        output_data.update_bits_and_time(dec_reference_info)

        reference_frame = dec_reference_info['decoded']
        kp_reference = models.kp_detector_model(reference_frame)
        models.kp_coder.kp_reference = kp_reference
        
        output_data.update_decoded(reference_frame)

        for idx in trange(1, input_data.gop_size):
            target_frame = frame2tensor(gop[idx])
            kp_target = models.kp_detector_model(target_frame)

            kp_coding_info = models.kp_coder.encode_kp(kp_target = kp_target)
            output_data.update_bits_and_time(kp_coding_info)
            
            kp_target_hat = kp_coding_info['kp_hat']

            #animation and residual coding
            anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
            animated_frame = models.animation_model.generate_animation(reference_frame, **anim_params)
            residual_frame = target_frame-animated_frame
            # output_data.decoded_video.append(tensor2frame(animated_frame))
            output_data.update_decoded(animated_frame)

            viz_params = {'reference_frame':reference_frame,
                            'target_frame':target_frame,
                            'res': residual_frame,
                            **anim_params,
                            'prediction': animated_frame}
            viz_img = visualizer.visualize(**viz_params)
            output_data.visualization.append(viz_img)
    return output_data

def hybrid_coder(models, input_data,output_data, visualizer: Visualizer, scale_factor: float= 1):
    for gop in input_data.gops:
        org_reference = frame2tensor(gop[0])
        input_data.original_video.extend(gop)
        
        dec_reference_info = models.animation_model.ref_coder.compress(org_reference,rate_idx=input_data.eval_params['qp'])
        output_data.update_bits_and_time(dec_reference_info)

        #create base layer_stream
        #downsample the base layer video
        bl_video = resize_frames(gop[1:input_data.gop_size],scale_factor)
        base_layer_info = run_hevc(bl_video, input_data.base_layer_qp)
        
        #upsample the decoded base layer
        base_layer = resize_frames(base_layer_info['dec_frames'], 1//scale_factor)
        output_data.update_bits_and_time(base_layer_info)

        reference_frame = dec_reference_info['decoded']
        kp_reference = models.kp_detector_model(reference_frame)
        models.kp_coder.kp_reference = kp_reference
        
        output_data.update_decoded(reference_frame)

        for idx in trange(1,input_data.gop_size):
            target_frame = frame2tensor(gop[idx])
            base_layer_frame = frame2tensor(base_layer[idx-1])
            
            kp_target = models.kp_detector_model(target_frame)

            kp_coding_info = models.kp_coder.encode_kp(kp_target = kp_target)
            output_data.update_bits_and_time(kp_coding_info)

            kp_target_hat = kp_coding_info['kp_hat']
            #animation and residual coding
            anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
            animated_frame = models.animation_model.generate_animation(reference_frame=reference_frame,
                        kp_reference=kp_reference,kp_target = kp_target_hat, base_layer = base_layer_frame)

            residual_frame = target_frame - animated_frame
            output_data.update_decoded(animated_frame)

            viz_params = {'reference_frame':reference_frame,
                            'target_frame':target_frame,
                            'res': residual_frame,
                            **anim_params,
                            'prediction': animated_frame}
            
            viz_img = visualizer.visualize(**viz_params)
            output_data.visualization.append(viz_img)
    return output_data

def predictive_coder(models, input_data, visualizer: Visualizer,method='rdac'):
    output_data  = Outputs()
    for gop in input_data.gops:
        input_data.original_video.extend(gop)
        org_reference = frame2tensor(gop[0], cuda=False)
        dec_reference_info = models.image_coder(org_reference)
        output_data.update_bits_and_time(dec_reference_info)

        reference_frame = dec_reference_info['decoded']
        with torch.no_grad():
            kp_reference = models.kp_detector_model(reference_frame)
            models.kp_coder.kp_reference = kp_reference
            
            output_data.decoded_video.append(tensor2frame(reference_frame))
            output_data.animated_video.append(tensor2frame(reference_frame))
            
            prev_latent = None
            for idx in trange(1,input_data.gop_size):
                target_frame = frame2tensor(gop[idx])
                kp_target = models.kp_detector_model(target_frame)

                kp_coding_info = models.kp_coder.encode_kp(kp_target = kp_target)
                output_data.update_bits_and_time(kp_coding_info)

                kp_target_hat = kp_coding_info['kp_hat']
                # saliency_map = generate_saliency_map(kp_target_hat['value'],(256,256))
                saliency_map = None
                #animation and residual coding
                anim_params = {'kp_reference':kp_reference,'kp_target':kp_target_hat}
                animated_frame = models.animation_model.animate(reference_frame, **anim_params)
                residual_frame = target_frame-animated_frame
                eval_params = {'rate_idx': input_data.eval_params['rd_point'],
                            'q_value':input_data.eval_params['q_value'],
                            'use_skip': input_data.eval_params['use_skip'],
                            'skip_thresh':input_data.eval_params['skip_thresh']}
                
                if idx == 1:
                    res_coding_info, skip = models.animation_model.compress_spatial_residual(residual_frame,prev_latent, **eval_params)     
                    prev_res_hat = res_coding_info['res_hat']
                else:
                    temporal_residual_frame = residual_frame - prev_res_hat
                    res_coding_info, skip = models.animation_model.compress_temporal_residual(temporal_residual_frame,prev_latent, **eval_params)
                    if not skip:
                        prev_res_hat = (res_coding_info['res_hat']+prev_res_hat)   

                if not skip:
                    prev_latent = res_coding_info['prev_latent']
                    output_data.update_bits_and_time(res_coding_info)
                    enh_prediction = (animated_frame + prev_res_hat).clamp(0,1)
                else:
                    enh_prediction = (animated_frame + prev_res_hat).clamp(0,1)

                output_data.animated_video.append(tensor2frame(animated_frame))
                output_data.decoded_video.append(tensor2frame(enh_prediction))
                
                viz_params = {'reference_frame':reference_frame,
                                'target_frame':target_frame,
                                'res': residual_frame,'res_hat': prev_res_hat,
                                'prediction': animated_frame,'enhanced_prediction':enh_prediction,
                                **anim_params}

                viz_img = visualizer.visualize(**viz_params)
                output_data.visualization.append(viz_img)
    return output_data


def mr_animation_coder(models, input_data,output_data,  visualizer: Visualizer):
    #Bidirectional prediction with uniform reference sampling
    total_frames = min(input_data.num_frames, len(input_data.video))
    coarse_gop_indices = list(range(0, total_frames,input_data.gop_size))
    coarse_gop_indices.append(total_frames-1)

    reference_info = {}
    last_ref_idx, next_ref_idx,last_coarse_idx = 0, 0, 0
    reference_frames_list  = []
    encoded_ref_count = 0
    used_ref_ids = []
    for idx in trange(total_frames):  
        if idx==0: #and len(list(reference_info.keys()))<input_data.eval_params['active_references']:
            reference_frames_list.append(input_data.video[idx])
            used_ref_ids.append(idx)

            org_reference = frame2tensor(input_data.video[idx])
            org_reference = org_reference.to(input_data.device)
            
            dec_reference_info = models.animation_model.ref_coder.compress(org_reference,rate_idx=input_data.eval_params['qp'])
            output_data.update_bits_and_time(dec_reference_info)
            encoded_ref_count +=1

            reference_frame = dec_reference_info['decoded']
            
            kp_reference = models.kp_detector_model(reference_frame)
            #compute the spatial complexity as a gradient of pixels
            # Calculate complexity for the first frame
            # Calculate the average complexity of all previous frames
            ref_fts = models.animation_model.reference_ft_encoder(reference_frame)
            models.kp_coder.kp_reference = kp_reference
            
            reference_info.update({idx: (reference_frame,ref_fts, kp_reference)})
            output_data.update_decoded(reference_info[idx][0])

            #check the reference info buffer and remove the earliest reference if buffer exceeds
            #number of active references
            
            #Look ahead and encode the future reference frame
            next_idx = min(len(input_data.video)-1,idx+input_data.gop_size)
            last_coarse_idx = next_idx
            

            next_org_reference = frame2tensor(input_data.video[next_idx])
            next_org_reference = next_org_reference.to(input_data.device)
            
            next_kp_reference = models.kp_detector_model(next_org_reference)        
            next_dec_reference_info = models.animation_model.ref_coder.compress(next_org_reference,rate_idx=input_data.eval_params['qp'])
            
            output_data.update_bits_and_time(next_dec_reference_info)
            
            next_reference_frame = next_dec_reference_info['decoded']
            next_kp_reference = models.kp_detector_model(next_org_reference)
            next_ref_fts = models.animation_model.reference_ft_encoder(next_reference_frame)
            reference_info.update({next_idx: (next_reference_frame,next_ref_fts, next_kp_reference)})
            #check the reference info buffer and remove the earliest reference if buffer exceeds
            #number of active references
            #Buffered params
            last_ref_idx = idx
            next_ref_idx = next_idx
            used_ref_ids.append(next_idx)
            encoded_ref_count +=1

        elif idx == next_ref_idx:
            last_ref_idx = next_ref_idx
            models.kp_coder.kp_reference = reference_info[last_ref_idx][2]
            output_data.update_decoded(reference_info[last_ref_idx][0])
            #Look ahead and encode the future reference frame
            if not last_coarse_idx>=len(input_data.video)-input_data.eval_params['search_window']-1:
                next_idx = min(last_coarse_idx+input_data.gop_size,len(input_data.video)-1)
                last_coarse_idx = next_idx
                if next_idx not in used_ref_ids:
                    next_org_reference = frame2tensor(input_data.video[next_idx])
                    next_org_reference = next_org_reference.to(input_data.device)
                    next_kp_reference = models.kp_detector_model(next_reference_frame)

                    next_dec_reference_info = models.animation_model.ref_coder.compress(next_org_reference,rate_idx=input_data.eval_params['qp'])
                    output_data.update_bits_and_time(next_dec_reference_info)
                    

                    next_reference_frame = next_dec_reference_info['decoded']
                    next_kp_reference = models.kp_detector_model(next_reference_frame)

                    next_ref_fts = models.animation_model.reference_ft_encoder(next_reference_frame)
                    
                    reference_info.update({next_idx: (next_reference_frame,next_ref_fts, next_kp_reference)})

                    encoded_ref_count +=1
                    next_ref_idx = next_idx
                    used_ref_ids.append(next_idx)
        else:
            #compute weight values for the references
            target_frame = frame2tensor(input_data.video[idx]).to(input_data.device)
            kp_target = models.kp_detector_model(target_frame)
            
            kp_coding_info = models.kp_coder.encode_kp(kp_target = kp_target)
            output_data.update_bits_and_time(kp_coding_info)
            kp_target_hat = kp_coding_info['kp_hat']

            active_references = reference_info
            anim_params = {'reference_info':active_references,'kp_target':kp_target_hat}
            
            if input_data.eval_params['use_ref_weights']:
                indices = [int(x) for x in list(active_references.keys())]
                indices.extend([idx])
                ref_indices = np.array(indices)
                ref_indices = torch.tensor(ref_indices).unsqueeze(0).to(input_data.device)
                anim_params['ref_indices'] = ref_indices

            animated_frame = models.animation_model.generate_animation(anim_params)
            output_data.update_decoded(animated_frame)

            viz_params = {'reference_frame':reference_info[last_ref_idx][0],
                        'target_frame':target_frame,
                        'kp_reference':reference_info[last_ref_idx][2],
                        'kp_target':kp_target_hat,
                        'prediction': animated_frame}
            viz_img = visualizer.visualize(**viz_params)
            output_data.visualization.append(viz_img) 
    return output_data


def test(config,dataset,animation_model_arch, kp_detector_arch,**kwargs ):
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
            animation_model = load_pretrained_model(animation_model_arch, path=pretrained_cpk_path, device=device)
            kp_detector_model = load_pretrained_model(kp_detector_arch, path=pretrained_cpk_path,name='kp_detector',device=device)

        animation_model.eval()
        temporal_prediction = False
        if 'rdac' in model_id:
            animation_model.sdc.update(force=True)
            animation_model.tdc.update(force=True)

        kp_detector_model.eval()
        if torch.cuda.is_available():
            animation_model = animation_model.cuda()
            kp_detector_model= kp_detector_model.cuda()

        reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
        motion_kp_coder = KpEntropyCoder()

        models = Models(animation_model, kp_detector_model, reference_image_coder, motion_kp_coder)    

        all_metrics = {}
        with torch.no_grad():
            for x in dataset:
                video = x['video']
                N,h,w,c = video.shape
                n_frames = min(num_frames, N)
                input_data = Inputs(config['eval_params'])
                
                #update codec params for this sequence
                input_data.num_frames = n_frames
                input_data.video = video
                input_data.create_gops()
                name = x['name']
                out_path = os.path.join(kwargs['log_dir'],name[0].split('.')[0])
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                if model_id == 'dac':
                    output_data = animation_coder(models,input_data, visualizer)
                elif model_id == 'hdac':
                    output_data = hybrid_coder(models,input_data, visualizer)
                elif model_id  == 'rdac':
                    output_data = predictive_coder(models,input_data, visualizer)
                else:
                    raise NotImplementedError(f"Codec of type <{model_id}> is not Available!")
                
                imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4",output_data.decoded_video, fps=10)
                imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4",output_data.visualization, fps=10)
                
                if len(output_data.animated_video)== len(output_data.decoded_video):
                    comp_vid = np.concatenate((np.array(input_data.original_video),np.array(output_data.animated_video),np.array(output_data.decoded_video)), axis=2)
                else:
                    comp_vid = np.concatenate((np.array(input_data.original_video),np.array(output_data.decoded_video)), axis=2)
                
                imageio.mimsave(f"{out_path}/{rd_point}_anim_enh.mp4",comp_vid, fps=10)

                metrics = monitor.compute_metrics(input_data.original_video,output_data.decoded_video)
                metrics.update({'bitrate':output_data.get_bitrate(input_data.fps, input_data.num_frames)})
                all_metrics[name[0]] = metrics
                print(metrics)
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


def test_dac(config,dataset,animation_model_arch, kp_detector_arch,**kwargs ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']

    visualizer = Visualizer(**config['visualizer_params'])
    monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['per_frame_metrics'])
    #get a pretrained dac_model based on current rdac config

    pretrained_cpk_path = config['dataset_params']['cpk_path']
    rd_point = config['eval_params']['rd_point']

    if pretrained_cpk_path is not None:
        animation_model = load_pretrained_model(animation_model_arch, path=pretrained_cpk_path, device=device)
        kp_detector_model = load_pretrained_model(kp_detector_arch, path=pretrained_cpk_path,name='kp_detector',device=device)

    animation_model.ref_coder.update(force=True)
    animation_model.eval()
    kp_detector_model.eval()

    kp_detector_model.eval()
    if torch.cuda.is_available():
        animation_model = animation_model.cuda()
        kp_detector_model= kp_detector_model.cuda()

    # reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
    motion_kp_coder = KpEntropyCoder()

    models = Models(animation_model, kp_detector_model, motion_kp_coder)    

    all_metrics = {}
    with torch.no_grad():
        for x in dataset:
            video = x['video']
            N,h,w,c = video.shape
            n_frames = min(num_frames, N)
            input_data = Inputs(config['eval_params'])
            
            #update codec params for this sequence
            input_data.num_frames = n_frames
            input_data.video = video
            input_data.device = device
            input_data.create_gops()
            name = x['name']
            out_path = os.path.join(kwargs['log_dir'],name.split('.')[0])
            os.makedirs(out_path, exist_ok=True)
            output_data  = Outputs(out_path)

            output_data = animation_coder(models,input_data,output_data, visualizer)
            output_data.f_dec.close()

            imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4",output_data.decoded_video, fps=10)
            imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4",output_data.visualization, fps=10)
            
    
            metrics = monitor.compute_metrics(input_data.video[:len(output_data.decoded_video)],output_data.decoded_video)
            metrics.update({'bitrate':output_data.get_bitrate(input_data.fps, input_data.num_frames)})
            all_metrics[name[0]] = metrics
            print(metrics)
            break
    with open(f"{kwargs['log_dir']}/metrics_{rd_point}.json", 'w') as f:
        json.dump(all_metrics, f, indent=4)
    

def test_rdac(config,dataset,animation_model_arch, kp_detector_arch,**kwargs ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']

    visualizer = Visualizer(**config['visualizer_params'])
    monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['per_frame_metrics'])
    #get a pretrained dac_model based on current rdac config
    pretrained_cpk_path = config['dataset_params']['cpk_path']
    rd_point = config['eval_params']['rd_point']


    if pretrained_cpk_path is not None:
        animation_model = load_pretrained_model(animation_model_arch, path=pretrained_cpk_path, device=device)
        kp_detector_model = load_pretrained_model(kp_detector_arch, path=pretrained_cpk_path,name='kp_detector',device=device)

    animation_model.eval()
    if 'rdac' in model_id:
        animation_model.sdc.update(force=True)
        animation_model.tdc.update(force=True)

    kp_detector_model.eval()
    if torch.cuda.is_available():
        animation_model = animation_model.cuda()
        kp_detector_model= kp_detector_model.cuda()

    reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
    motion_kp_coder = KpEntropyCoder()

    models = Models(animation_model, kp_detector_model, reference_image_coder, motion_kp_coder)    

    all_metrics = {}
    with torch.no_grad():
        for x in dataset:
            video = x['video']
            N,h,w,c = video.shape
            n_frames = min(num_frames, N)
            input_data = Inputs(config['eval_params'])
            
            #update codec params for this sequence
            input_data.num_frames = n_frames
            input_data.video = video
            input_data.create_gops()
            name = x['name']
            out_path = os.path.join(kwargs['log_dir'],name[0].split('.')[0])
            os.makedirs(out_path, exist_ok=True)
            output_data = predictive_coder(models,input_data, visualizer)

            imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4",output_data.decoded_video, fps=10)
            imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4",output_data.visualization, fps=10)
            
            if len(output_data.animated_video)== len(output_data.decoded_video):
                comp_vid = np.concatenate((np.array(input_data.original_video),np.array(output_data.animated_video),np.array(output_data.decoded_video)), axis=2)
            else:
                comp_vid = np.concatenate((np.array(input_data.original_video),np.array(output_data.decoded_video)), axis=2)
            
            imageio.mimsave(f"{out_path}/{rd_point}_anim_enh.mp4",comp_vid, fps=10)

            metrics = monitor.compute_metrics(input_data.original_video,output_data.decoded_video)
            metrics.update({'bitrate':output_data.get_bitrate(input_data.fps, input_data.num_frames)})
            all_metrics[name[0]] = metrics
            print(metrics)
            break
    # with open(f"{kwargs['log_dir']}/metrics_{rd_point}.json", 'w') as f:
    #     json.dump(all_metrics, f, indent=4)


def test_hdac(config,dataset,animation_model_arch, kp_detector_arch,**kwargs ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']

    visualizer = Visualizer(**config['visualizer_params'])
    monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['per_frame_metrics'])
    #get a pretrained dac_model based on current rdac config
    pretrained_cpk_path = config['dataset_params']['cpk_path']
    rd_point = config['eval_params']['base_layer_qp']


    if pretrained_cpk_path is not None:
        animation_model = load_pretrained_model(animation_model_arch, path=pretrained_cpk_path, device=device)
        kp_detector_model = load_pretrained_model(kp_detector_arch, path=pretrained_cpk_path,name='kp_detector',device=device)
    
    animation_model.ref_coder.update(force=True)
    animation_model.eval()
    kp_detector_model.eval()
    
    if torch.cuda.is_available():
        animation_model = animation_model.cuda()
        kp_detector_model= kp_detector_model.cuda()

    # reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
    
    motion_kp_coder = KpEntropyCoder()

    models = Models(animation_model, kp_detector_model, motion_kp_coder)    

    all_metrics = {}
    with torch.no_grad():
        for x in dataset:
            video = x['video']
            N,h,w,c = video.shape
            n_frames = min(num_frames, N)
            input_data = Inputs(config['eval_params'])
            
            #update codec params for this sequence
            input_data.num_frames = n_frames
            input_data.video = video
            input_data.create_gops()
            name = x['name']
            out_path = os.path.join(kwargs['log_dir'],name.split('.')[0])
            os.makedirs(out_path, exist_ok=True)
            output_data  = Outputs(out_path)

            
            output_data = hybrid_coder(models,input_data,output_data, visualizer)

            imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4",output_data.decoded_video, fps=10)
            imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4",output_data.visualization, fps=10)

            metrics = monitor.compute_metrics(input_data.video[:len(output_data.decoded_video)],output_data.decoded_video)
            metrics.update({'bitrate':output_data.get_bitrate(input_data.fps, input_data.num_frames)})
            all_metrics[name[0]] = metrics
            print(metrics)
    with open(f"{kwargs['log_dir']}/metrics_{rd_point}.json", 'w') as f:
        json.dump(all_metrics, f, indent=4)




def test_mrdac(config,dataset,animation_model_arch, kp_detector_arch,**kwargs ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = kwargs['model_id']
    num_frames = config['eval_params']['num_frames']

    visualizer = Visualizer(**config['visualizer_params'])
    monitor = Metrics(config['eval_params']['metrics'],config['eval_params']['per_frame_metrics'])
    #get a pretrained dac_model based on current rdac config

    pretrained_cpk_path = config['dataset_params']['cpk_path']
    rd_point = config['eval_params']['gop_size']

    if pretrained_cpk_path is not None:
        animation_model = load_pretrained_model(animation_model_arch, path=pretrained_cpk_path, device=device)
        kp_detector_model = load_pretrained_model(kp_detector_arch, path=pretrained_cpk_path,name='kp_detector',device=device)

    animation_model.ref_coder.update(force=True)
    animation_model.eval()
    kp_detector_model.eval()

    kp_detector_model.eval()
    if torch.cuda.is_available():
        animation_model = animation_model.cuda()
        kp_detector_model= kp_detector_model.cuda()

    # reference_image_coder = ImageCoder(config['eval_params']['qp'],config['eval_params']['ref_codec'])
    motion_kp_coder = KpEntropyCoder()

    models = Models(animation_model, kp_detector_model, motion_kp_coder)    

    all_metrics = {}
    with torch.no_grad():
        for x in dataset:
            video = x['video']
            N,h,w,c = video.shape
            n_frames = min(num_frames, N)
            input_data = Inputs(config['eval_params'])
            
            #update codec params for this sequence
            input_data.num_frames = n_frames
            input_data.video = video
            input_data.device = device
            input_data.create_gops()
            name = x['name']
            out_path = os.path.join(kwargs['log_dir'],name.split('.')[0])
            os.makedirs(out_path, exist_ok=True)
            output_data  = Outputs(out_path)

            output_data = mr_animation_coder(models,input_data,output_data, visualizer)
            output_data.f_dec.close()

            imageio.mimsave(f"{out_path}/{rd_point}_enh_video.mp4",output_data.decoded_video, fps=10)
            imageio.mimsave(f"{out_path}/{rd_point}_viz.mp4",output_data.visualization, fps=10)
            
    
            metrics = monitor.compute_metrics(input_data.video[:len(output_data.decoded_video)],output_data.decoded_video)
            metrics.update({'bitrate':output_data.get_bitrate(input_data.fps, input_data.num_frames)})
            all_metrics[name[0]] = metrics

    with open(f"{kwargs['log_dir']}/metrics_{rd_point}.json", 'w') as f:
        json.dump(all_metrics, f, indent=4)
   
test_functions = {'dac': test_dac,
                  'hdac': test_hdac,
                  'hdac_hf': test_hdac,
                  'rdac': test_rdac,
                  'crdac':test_rdac,
                  'mrdac':test_mrdac}
