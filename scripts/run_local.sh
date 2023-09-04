PYTHON="python"
CONFIG="configs/ade20k/panoptic-segmentation/vit/maskformer2_conditional_vit_384_bs16_160k.yaml"
GPU_NUM=1

export DETECTRON2_DATASETS=/teamdrive/dataset/

CUDA_VISIBLE_DEVICES=1 $PYTHON train_net.py \
                        --config-file $CONFIG \
                        --num-gpus $GPU_NUM \
                        SOLVER.IMS_PER_BATCH 2 \
