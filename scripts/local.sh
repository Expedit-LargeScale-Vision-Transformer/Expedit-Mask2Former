#!/bin/bash

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
IMAGE="weicong/mask2former"
CONFIG=$1

read -d '' CMD <<EOF
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export DETECTRON2_DATASETS=/data

python train_net.py \
  --config-file $CONFIG \
  --num-gpus 1 \
  --eval-only MODEL.WEIGHTS /outputs/ade20k/panoptic-segmentation/vit/maskformer2_single_scale_vit_384_bs16_160k/model_0159999.pth \
  SOLVER.IMS_PER_BATCH 2 
EOF

echo "=========== COMMAND ==========="
echo "$CMD"
echo "==============================="

sudo nvidia-docker run \
    --rm --ipc=host -it \
    -v "$PWD":/workspace \
    -v /home/openseg/jiading_blob_fix/data:/data/ \
    -v /home/openseg/jiading_blob_fix/outputs-aml/mask2former-weicong:/outputs/ \
    -u $(id -u):$(id -g) \
    "${IMAGE}" \
    bash -c "$CMD"
