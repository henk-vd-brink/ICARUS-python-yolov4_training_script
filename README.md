## About the Project
This repository contains a script to train a YoloV4 model using Pytorch.

## Getting Started

### Prerequisites
- CUDA

### Installation
```
git clone https://github.com/henk-vd-brink/python-ml-yolov4_training.git
cd python-ml-yolov4_training
```

### Run
```
cd app
./scripts/train.sh
```

## Structure
```
.
├── data
│   └── american_signlanguage
│       ├── README.dataset.txt
│       ├── README.roboflow.txt
│       ├── test
│       │   ├── A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg
│       │   ├── B14_jpg.rf.ed5ba6d44f55ab03e62d2baeac4aa1aa.jpg
│       │   ├── _annotations.txt
│       │   └── _classes.txt
│       ├── train
│       │   ├── A0_jpg.rf.292a080422ba984985192f413101af41.jpg
│       │   ├── Z8_jpg.rf.81357aeb1a89b914fc16d5a0418db9ff.jpg
│       │   ├── _annotations.txt
│       │   └── _classes.txt
│       └── valid
│           ├── A10_jpg.rf.470b1af0feaa190a2d29fcafd6fe747d.jpg
│           ├── A14_jpg.rf.392d1ac7f954c7a26a3bf99762e281e3.jpg
│           ├── _annotations.txt
│           └── _classes.txt
├── models
├── pytorch
│   ├── dataset.py
│   ├── logs
│   ├── model.py
│   └── train.py
├── scripts
│   └── train.sh
└── tree.txt
```
