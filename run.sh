#!/bin/bash

sudo docker run -v app:/code/app --gpus all --shm-size 50G --privileged -d --name tr --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm train

