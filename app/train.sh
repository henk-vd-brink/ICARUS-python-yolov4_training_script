#!/bin/bash

DATASET_NAME="american_sign_language"

python3 pytorch-YOLOv4/train.py -b 2 \
                                -s 1 \
                                -l 0.001 \
                                -g 0 \
                                -pretrained pytorch-YOLOv4/cfg/yolov4.conv.137.pth \
                                -classes 26 \
                                -dir data/$DATASET_NAME/train \
                                -train_label_path data/$DATASET_NAME/train/_annotations.txt \
				-val_label_path data/$DATASET_NAME/valid/_annotations.txt \
                                -epochs 100
