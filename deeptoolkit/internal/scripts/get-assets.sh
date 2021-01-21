#!/usr/bin/env bash

if [ ! -z "$(ls -A $1)" ]; then
  echo "Delete existing files from directory? [y/n]"
  read -r input
  if [ "$input" = "y" ]; then
    rm $1/*
  else [ "$input" = "n" ]
    echo "Aborting."
    exit
  fi
fi

# Get model for DNN.
wget -O $1/model.prototxt \
        https://raw.githubusercontent.com/opencv/opencv/ea667d82b30a19b10a6c00edf8acc6e9dd85c429/samples/dnn/face_detector/deploy.prototxt
# Get caffemodel for DNN.
wget -O $1/dnn-weights.caffemodel \
        https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
