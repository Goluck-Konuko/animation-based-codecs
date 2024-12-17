# Ultra-low Bitrate Video Compression Using Deep Animation Models

Welcome to the repository for **Ultra-low Bitrate Video Compression Using Deep Animation Models**. This codebase implements methods and models described in cutting-edge research on low-bitrate video conferencing and animation-based video compression. The repository is designed to serve researchers and developers interested in leveraging deep learning for video compression.

---

## 📚 Related Publications
This repository accompanies the following papers:

1. **[Ultra-Low Bitrate Video Conferencing Using Deep Image Animation](https://arxiv.org/abs/2012.00346v1)**
2. **[A Hybrid Deep Animation Codec for Low-Bitrate Video Conferencing](https://ieeexplore.ieee.org/abstract/document/10458867)**
3. **[Improving Reconstruction Fidelity in Generative Face Video Coding Using High-Frequency Shuttling](https://ieeexplore.ieee.org/abstract/document/10458867)**
4. **[Predictive Coding for Animation-Based Video Compression](https://ieeexplore.ieee.org/abstract/document/10222205)**
5. **[Improved Predictive Coding for Animation-Based Video Compression](https://ieeexplore.ieee.org/abstract/document/10772980)**
6. **[Multi-Reference Generative Face Video Compression with Contrastive Learning](https://arxiv.org/html/2409.01029v1)**

---

## ⚙️ Installation
This repository supports Python 3. To set up the environment, clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Assets

### YAML Configuration
The YAML configuration files are used to define the settings for training and testing the models. Example files are located in the `train/test config` directory:
- `[train/test]_config/dac.yaml`
- `[train/test]_config/hdac.yaml`
- `[train/test]_config/rdac.yaml`

During inference, use the `--mode test` flag with the same configuration file after updating the `eval_params` section appropriately.

### Datasets
- **VoxCeleb**: Follow the instructions in the [video-preprocessing repository](https://github.com/AliaksandrSiarohin/video-preprocessing) to prepare the dataset.
- **Creating Your Own Videos**: Ensure that input videos are cropped to focus on the speaker’s face at a resolution of 256x256 pixels. (Support for higher resolutions is under development.)
- **Pre-processed Videos (256x256 px)**: Pre-processed videos are available for download from our [Google Drive link](https://drive.google.com/drive/folders/1g0U1ZCTszm3yrmIewg7FahXsxyMBfxKj?usp=sharing). Place these videos in the following folders:
  - `datasets/train`
  - `datasets/inference`

### Evaluation Metrics
Our metrics module incorporates suggestions from **JPEG-AI** alongside popular quantitative metrics used in computer vision. Supported metrics include:
- `psnr`, `psnr-hvs`, `fsim`, `iw_ssim`, `ms_ssim`
- `vif`, `nlpd`, `vmaf`, `lpips`, `msVGG`

---

## 🚀 Training
To train a model, update the relevant parameters in the corresponding `train_config/[MODEL_NAME].yaml` file or use the default configuration (to reproduce our results). Run the following command:

```bash
bash training_script.sh [MODEL_NAME]
```

> **Note:** The default setup requires 2 x A40 GPUs. Adjust the batch size in the configuration file if using a different hardware setup.

---

## 🧪 Testing
To test a model, update the `eval_params` in the corresponding `test_config/[MODEL_NAME].yaml` file and run:

```bash
bash test_script.sh [MODEL_NAME]
```
Refer to JVET_AH0114 and subsequent documentation and [Reference software for CTC implementations](https://vcgit.hhi.fraunhofer.de/jvet-ahg-gfvc/gfvc_v1 ) and benchmark evaluation against other GFVC frameworks.
---

## 🙏 Attributions
This codebase includes components adapted from the following projects:

1. **[First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation)**: For the base architecture of deep image animation using unsupervised keypoints.
2. **[CompressAI](https://github.com/InterDigitalInc/CompressAI)**: For learned image compression.
3. **[JPEG-AI](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-qaf)**: For evaluation metrics.

---

## 📬 Contact
For any questions, feedback, or collaboration opportunities, feel free to contact the maintainers or open an issue in this repository.

---

## 🌟 Acknowledgments
We appreciate the contributions of the research community that enabled this work. If you use this repository or find it helpful, please consider citing the relevant papers.

---

### ⭐ Star This Repository
If you find this project useful, give it a star on GitHub to support further development!
