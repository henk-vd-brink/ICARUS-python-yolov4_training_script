## About the Project
This repository contains code for a dockerized training script. Since all the correct CUDA and CUDNN versions are installed in the container, the script can run on almost any CUDA enabled machine.

## Getting Started
### Prerequisites 
- Docker
- GPU with CUDA (installation not necessary)

### Installation
```
git clone https://github.com/henk-vd-brink/ICARUS-python-yolov4_training_script.git
cd ICARUS-python-yolov4_training_script
```

### Build
```
docker build -t training_image .
```

### Run
```
docker run --gpus all --shm-size 50G --privileged -d --name tr --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm training_image
```
