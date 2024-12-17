import os
import numpy as np
import imageio.v3 as iio
from skimage import io, img_as_float32
from skimage.transform import resize
from torch.utils.data import Dataset
from .augmentation import AllAugmentationTransform
from typing import Dict, Any
from functools import partial

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    def __init__(self, train_dir: str,test_dir: str, frame_shape: tuple =(256, 256, 3), is_train: bool=True,
                 augmentation_params: Dict[str, Any]=None, num_sources: int=2, target_delta: int=2,use_audio=False, **kwargs):
        
        self.is_train = is_train
        self.frame_shape = tuple(frame_shape)
        if self.is_train:
            self.root_dir = train_dir
        else:
            self.root_dir = test_dir
        self.videos = os.listdir(self.root_dir)
 
        self.num_sources = num_sources
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
            if os.path.isdir(path):
                frames = os.listdir(path)
                n_frames = len(frames) 
            else:
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

            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32
                    
            if os.path.isdir(path):
                if type(frames[0]) is bytes:
                    video_array = [resize_fn(io.imread(os.path.join(path, frames[idx].decode('utf-8')))) for idx in
                                frame_idx]
                else:
                    video_array = [resize_fn(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            else:
                video_array = []
                for idx in frame_idx:
                    video_array.append(resize_fn(video[idx]))

            if self.transform is not None:
                video_array = self.transform(video_array)
                
            for idx in range(self.num_sources):
                frame = video_array[idx]
                if idx == 0 :
                    out['reference'] = frame.transpose((2, 0, 1))
                else:
                    out[f'target_{idx-1}'] = frame.transpose((2, 0, 1))

        else:
            video = np.array(iio.imread(f"{path}",plugin="pyav"))
            out.update({'video':video, 'name':name.split('.')[0]})
        return out



class MRFramesDataset(FramesDataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    def __init__(self, train_dir: str,test_dir: str, frame_shape: tuple =(256, 256, 3), is_train: bool=True,
                 augmentation_params: Dict[str, Any]=None, num_sources: int=2, target_delta: int=2, **kwargs):
        super(MRFramesDataset, self).__init__(train_dir,test_dir, frame_shape, is_train,
                                            augmentation_params, num_sources, target_delta, **kwargs)
        self.is_train = is_train
        self.frame_shape = tuple(frame_shape)
        if self.is_train:
            self.root_dir = train_dir
        else:
            self.root_dir = test_dir
        self.videos = os.listdir(self.root_dir)
 
        self.num_sources = num_sources
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
           
            if os.path.isdir(path):
                frames = os.listdir(path)
                n_frames = len(frames) 
            else:
                video = iio.imread(f"{path}", plugin='pyav')
                n_frames = len(video) 

            frame_idx = np.random.choice(range(n_frames), size=self.num_sources, replace=False)
            out['rf_weights'] = frame_idx
            

            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32

            if os.path.isdir(path):
                if type(frames[0]) is bytes:
                    video_array = [resize_fn(io.imread(os.path.join(path, frames[idx].decode('utf-8')))) for idx in
                                frame_idx]
                else:
                    video_array = [resize_fn(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
            else:
                video_array = []
                for idx in frame_idx:
                    video_array.append(resize_fn(video[idx]))
   
            if self.transform is not None:
                video_array = self.transform(video_array)
                
            for idx in range(self.num_sources):
                frame = video_array[idx]
                # print(frame.shape)
                if idx < self.num_sources-1 :
                    out[f'reference_{idx}'] = frame.transpose((2, 0, 1))
                else:
                    out[f'target_0'] = frame.transpose((2, 0, 1))

        else:
            video = np.array(iio.imread(f"{path}",plugin="pyav"))
            out.update({'video':video, 'name':name.split('.')[0]})
        return out

class HDACFramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    def __init__(self, train_dir: str,test_dir: str, frame_shape: tuple =(256, 256, 3), is_train: bool=True,
                 base_layer: bool=False,  augmentation_params: Dict[str, Any]=None, 
                 num_sources: int=2, base_layer_params = None,target_delta: int=2, **kwargs):
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
                if self.base_layer_params['variable_quality']:
                    bl_qp = np.random.choice(list(self.base_layer_params['qp_values'].keys())) 
                else:
                    #Use the lowest QP only
                    bl_qp = '50'
                bl_path = os.path.join(f"{self.base_layer_params['dir']}/{self.base_layer_params['bl_codec']}_bl/{bl_qp}", name)
                out.update({'lambda_value': self.base_layer_params['qp_values'][bl_qp]['lmbda'],
                            'bitrate': self.base_layer_params['qp_values'][bl_qp]['bitrate']})
   
            video = iio.imread(f"{path}", plugin='pyav')
            n_frames = len(video) 

            src_idx = np.random.choice(n_frames//2)
            drv_idx = np.random.choice(range(n_frames//2, n_frames-(self.num_sources*self.tgt_delta)))
            frame_idx = [src_idx, drv_idx]
            
            #add intermediate frames backwards from the target if necessary
            if self.num_sources > 2:
                for idx in range(self.num_sources-2):
                    drv_idx += np.random.choice(range(self.tgt_delta))
                    frame_idx.append(drv_idx)   
            
            video_array = []
            for idx in frame_idx:
                video_array.append(video[idx])

            video_array = img_as_float32(video_array)
            if self.base_layer:
                bl_video = iio.imread(f"{bl_path}", plugin="pyav")
                bl_video_array = []
                for idx in frame_idx:
                    bl_video_array.append(bl_video[idx])
                bl_video_array = img_as_float32(bl_video_array)
                video_array = np.concatenate([video_array,bl_video_array],axis=0)

            if self.transform is not None:
                video_array = self.transform(video_array)
                
            
            for idx in range(self.num_sources):
                frame = video_array[idx]
                if idx == 0 :
                    #first source frame
                    out['reference'] = frame.transpose((2, 0, 1))
                else:
                    out[f'target_{idx-1}'] = frame.transpose((2, 0, 1))
    
                    if self.base_layer:
                        bl_frame = video_array[idx+self.num_sources]
                        out[f'base_layer_{idx-1}'] = bl_frame.transpose((2, 0, 1))
        else:
            video = np.array(iio.imread(f"{path}",plugin="pyav"))
            out.update({'video':video, 'name':name.split('.')[0]})
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
