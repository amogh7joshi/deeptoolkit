#!/usr/bin/env bash

# A copy of the deeptoolkit/internal/scripts/get-resources.sh script for higher-level testing and compatibility.

if [ -z "$(ls -A ../deeptoolkit/internal/assets/)" ]; then
  echo "The assets directory should be empty if calling the asset configuration script."
  echo "Delete existing files from directory? [y/n]"
  read -r input
  if [ "$input" = "y" ]; then
    rm ../deeptoolkit/internal/assets/*
  else [ "$input" = "n" ]
    echo "Aborting."
    exit
  fi
fi

# Get model for DNN.
wget -O ../deeptoolkit/internal/assets/model.prototxt \
        https://raw.githubusercontent.com/opencv/opencv/ea667d82b30a19b10a6c00edf8acc6e9dd85c429/samples/dnn/face_detector/deploy.prototxt
# Get caffemodel for DNN.
wget -O ../deeptoolkit/internal/assets/dnn-weights.caffemodel \
        https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
