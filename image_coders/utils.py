import os
import json
import torch
import imageio
import numpy as np
from typing import List, Dict, Any


def tensor2frame(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().cpu().squeeze().numpy().transpose(1,2,0) * 255.0).astype(np.uint8)

def frame2tensor(frame:np.ndarray, device='cpu') -> torch.Tensor:
    frame =  torch.from_numpy(frame.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    if device=='cuda' and torch.cuda.is_available():
        frame = frame.cuda()
    return frame

def to_cuda(x: torch.Tensor) -> torch.Tensor:
    '''move tensor to cuda for gpu computation'''
    if torch.cuda.is_available():
        x = x.cuda()
    return x

from pathlib import Path
def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return os.path.getsize(filepath)*8

def read_bitstring(filepath:str):
    '''
    input: Path to a binary file
    returns: binary string of file contents
    '''
    with open(filepath, 'rb') as bt:
        bitstring = bt.read()
    return bitstring


def compute_bitrate(bits: int, fps: float, frames: int) -> float:
    return round(((bits*fps)/(1000*frames)), 2)

def save_videos(path: str, videos: Dict[str, List[np.ndarray]], metadata: Dict[str, Any]) -> None:
    out_path = os.path.join(path, metadata['c_name'], metadata['name'])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #save decoded video
    imageio.mimsave(f"{out_path}/{metadata['l_name']}_dec.mp4", videos['decoded'], fps=metadata['fps'])
    #save visualization
    imageio.mimsave(f"{out_path}/{metadata['l_name']}_vis.mp4", videos['visualization'], fps=metadata['fps'])
    if 'mask' in videos:
        imageio.mimsave(f"{out_path}/{metadata['l_name']}_mask.mp4", videos['mask'], fps=metadata['fps'])

def save_metrics(path: str, metrics: Dict[str, List[float]], metadata: Dict[str, Any]) -> None:
    out_path = os.path.join(path,metadata['c_name'])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    file_path = f"{out_path}/{metadata['c_name']}_metrics.json"
    if os.path.exists(file_path):
        #read and update the existing file
        with open(file_path, 'r') as dt:
            all_metrics = json.load(dt)

        l_metrics = {'fps': metadata['fps'],
                    'bitrate' : metrics['bitrate'] ,
                    **metrics['metrics']}
        if metadata['name'] in all_metrics:
            all_metrics[metadata['name']].update({metadata['l_name']: l_metrics})
        else:
            all_metrics.update({metadata['name']: {metadata['l_name']: l_metrics}})
    else:
        #create new
        all_metrics = {metadata['name']: {metadata['l_name']: {'fps': metadata['fps'],
                                                            'bitrate' : metrics['bitrate'] ,
                                                            **metrics['metrics']}}}
    with open(file_path, 'w') as out:
        json.dump(all_metrics, out)
