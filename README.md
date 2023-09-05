# Ultra-low bitrate video compression using deep animation models

This repository contains the source code for the papers
[ULTRA-LOW BITRATE VIDEO CONFERENCING USING DEEP IMAGE ANIMATION](https://arxiv.org/abs/2012.00346v1),
[A HYBRID DEEP ANIMATION CODEC FOR LOW-BITRATE VIDEO CONFERENCING](https://arxiv.org/abs/2207.13530) and 
[PREDICTIVE CODING FOR ANIMATION-BASED VIDEO COMPRESSION](https://arxiv.org/abs/2307.04187)


## Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

## Assets
### YAML Config
Describes the configuration settings for for training and testing the models. 
See ```config/dac.yaml```, ```config/hdac.yaml```,```config/rdac.yaml```.
Use ```--mode test``` at inference time with the same config file after changing the ```eval_params``` appropriately.


### Datasets
 	**VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

 	**Creating your own videos**. 
 	The input videos should be cropped to target the speaker's face at a resolution of 256x256 (Updates are underway to add higher resolution). 

    **Pre-processed videos (256x256 px)**
    We provide preprocessed videos at the following link: [google-drive](https://drive.google.com/drive/folders/1g0U1ZCTszm3yrmIewg7FahXsxyMBfxKj?usp=sharing)

    Download put the videos in ```datasets/train``` and ```datasets/inference``` folders.


### Pre-trained checkpoint
Checkpoints can be found under following link: [google-drive](https://drive.google.com/drive/folders/1DNcgE0ytZiFWoATB6VqdANpLmoYqqOJN?usp=drive_link). Download and place in the ```checkpoints/``` directory.



#### Metrics
We include a metrics module combining the suggestions from JPEG-AI with popular quantiative metrics used in computer vision and beyond.
Supported metrics: 'psnr', 'psnr-hvs','fsim','iw_ssim','ms_ssim','vif','nlpd', 'vmaf','lpips'


## Training
Set the ```config/[MODEL_NAME].yaml``` parameters appropriately or use default (to reproduce our results) and run ```bash script_training.sh [MODEL_NAME]```. 
The default setup uses a single GPU (NVIDIA-A100). However, training DAC and HDAC can be trained on multiple GPUs by using distributed dataparallel and setting ```--device_ids``` parameter as desired.

## Testing
Set the ```eval_params``` on the ```config/[MODEL_NAME].yaml``` file and run ```bash script_test.sh [MODEL_NAME]```


### Attributions
This code base  contains source code from the following works:
1.  [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation) for the base architecture of deep image animation with unsupervised keypoints.
2. [Compressai](https://github.com/InterDigitalInc/CompressAI) for Learned image compression.
3. [JPEG-AI](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-qaf) for evaluation metrics.