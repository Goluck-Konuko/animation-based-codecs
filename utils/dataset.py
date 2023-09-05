import os
import json
import numpy as np
import imageio.v3 as iio
from skimage import img_as_float32
from torch.utils.data import Dataset
from utils.augmentation import AllAugmentationTransform
from typing import Dict, Any

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    def __init__(self, train_dir: str,test_dir: str, frame_shape: tuple =(256, 256, 3), is_train: bool=True,
                 base_layer: bool=False,  augmentation_params: Dict[str, Any]=None, 
                 num_sources: int=2, use_saliency_map: bool=False,base_layer_params = None,
                 target_delta: int=2, **kwargs):
        print("LOADING DATASET..")
        self.is_train = is_train
        self.frame_shape = tuple(frame_shape)
        if self.is_train:
            self.root_dir = train_dir
        else:
            self.root_dir = test_dir
        self.videos = os.listdir(self.root_dir)
                
        self.num_sources = num_sources

        self.base_layer = base_layer
        self.base_layer_params = base_layer_params
        self.tgt_delta = target_delta

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self)->int:
        return len(self.videos)

    def __getitem__(self, idx :int)->Dict[str, Any]:
        name = self.videos[idx]
        path = os.path.join(self.root_dir, name)
        out = {}
        if self.is_train:
            if self.base_layer:
                bl_qp = np.random.choice(list(self.base_layer_params['qp_values'].keys()))
                bl_path = os.path.join(f"{self.base_layer_params['dir']}_{bl_qp}", name)
                out.update({'lambda_value': self.base_layer_params['qp_values'][bl_qp]})
                
            video = iio.imread(f"{path}", plugin='pyav')
            n_frames = len(video) 

            src_idx = np.random.choice(n_frames//2)
            drv_idx = np.random.choice(range(n_frames//2, n_frames-(self.num_sources*self.tgt_delta)))
            frame_idx = [src_idx, drv_idx]
            
            #add intermediate frames backwards from the target if necessary
            if self.num_sources > 2:
                for idx in range(self.num_sources-2):
                    drv_idx += self.tgt_delta
                    frame_idx.append(drv_idx)   
            
            video_array = []
            for idx in frame_idx:
                video_array.append(video[idx])
            video_array = img_as_float32(video_array)

            
            if self.transform is not None:
                video_array = self.transform(video_array)
                
            if self.base_layer:
                bl_video = iio.imread(f"{bl_path}", plugin="pyav")
                bl_video_array = []
                for idx in frame_idx:
                    bl_video_array.append(bl_video[idx])
                bl_video_array = img_as_float32(bl_video_array)

            
            for idx in range(self.num_sources):
                frame = video_array[idx]
                if idx == 0 :
                    #first source frame
                    out['reference'] = frame.transpose((2, 0, 1))
                else:
                    out[f'target_{idx-1}'] = frame.transpose((2, 0, 1))
                    if self.base_layer:
                        out[f'base_layer_{idx-1}'] = bl_video_array[idx].transpose((2, 0, 1))
        else:
            video = np.array(iio.imread(f"{path}",plugin="pyav"))
            out.update({'video':video, 'name':name})
        return out



class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset: FramesDataset, num_repeats: int=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self)->int:
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx)->FramesDataset:
        return self.dataset[idx % self.dataset.__len__()]
