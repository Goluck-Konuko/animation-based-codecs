import os
import imageio 
import numpy as np
from hevc import HEVC
from vvc import VVC_VTM
from vvenc import VvenC
from multiprocessing import Pool
from metrics.metrics import Metrics


def hevc_encode(args):
    metrics =  ['ms_ssim','fsim','lpips','lpips_vgg','dists','iw_ssim','msVGG', 'vmaf','nlpd','exp'] 

    path = args[0]
    qp = args[1]
    n_frames = args[2]
    gop_size=args[3]
    config_template = "hevc_hm/config_template.cfg"
    video = imageio.get_reader(path)

    seq_name = path.split("/")[-1].split(".")[0]

    fps = video.get_meta_data()['fps']
    frames = []

    for idx, frame in enumerate(video):
        frames.append(frame)

    frames = frames[:n_frames]

    hevc_params = {
                'seq_name':seq_name,
                'qp':qp,
                'sequence':frames,
                'gop_size':gop_size,
                'fps':fps,
                'config':config_template
            }
    hevc_coder = HEVC(**hevc_params,)
    info_out = hevc_coder.run()
    dec_frames = info_out['dec_frames']
    # # self.res_hat.extend(res_dec-128)
    bits = info_out['bitstring_size']
    bitrate = (bits*fps)/(1000*n_frames)

    monitor = Metrics(metrics=metrics)
    metrics = monitor.compute_metrics(frames, dec_frames, temporal=True)
    metrics['bitrate'] = bitrate
    print(metrics)




if __name__ == "__main__":
    config_template = "hevc_hm/config_template.cfg"

    dataset = "vox"

    videos = os.listdir(f"../../../datasets/inference/{dataset}")
    gop_size = 8
    n_frames = 8
    qps = [50,40,35,30,25]

    encodings = []
    for v in videos:
        path = f"../../../datasets/inference/{dataset}/{v}"
        for qp in qps:
            encodings.append([path,qp,n_frames, gop_size])
    
    hevc_encode(encodings[0])