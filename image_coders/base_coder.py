import torch
import numpy as np
from .bpg_codec import BPG
from .dl_codec import AImageCodec
from .utils import tensor2frame, frame2tensor
 
class ImageCoder:
    def __init__(self, qp=1, name='bpg') -> None:
        self.name = name
        if name == 'bpg':
            self.ref_coder = BPG()
        else:
            self.ref_coder = AImageCodec(qp=qp)
            
    def __call__(self, frame, s_qp: int=30):
        if self.name == 'bpg':
            if isinstance(frame, torch.Tensor):
                frame = tensor2frame(frame)
            out = self.ref_coder.run(frame, s_qp)
        else:
            if isinstance(frame, np.ndarray):
                frame = frame2tensor(frame)
            out = self.ref_coder.run(frame)
        return out