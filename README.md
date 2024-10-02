# OR-FCOS

Code for the paper "An Enhanced Fully Convolutional One-Stage Approach for Object Detection: Advancing Performance and Efficiency in Computer Vision". This repository includes the implementation of our proposed model.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13832203.svg)](https://doi.org/10.5281/zenodo.13832203)

## Dataset Access

The dataset is uploading and will be publicly available soon.

To use the data to train our model, first create a folder named 'data' and unzip the ORaph8K.zip.

## Installation

```
# Ensure that Miniconda is installed

# Create virtual environment
conda create -n oraph python=3.10
conda activate oraph

# Install PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install OpenMMLab Dependencies
pip install cmake lit
pip3 install openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.1.0"

# Clone and install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
cd ..

# Clone and install mmdetection
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
cd ..
```

After installation, the folders should be structured as follows :

```
.
├── mmdetection
├── mmpretrain
└── data
    └── ORaph8K
        ├── val
        ├── train
        └── test
```

## Training

Place the `OR_FCOS.py` file in this repository to the folder `mmdetection/configs` , then use `mmdetection/train.py`to train.

