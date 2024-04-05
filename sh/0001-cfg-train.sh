DISKPATH=40005
export HF_HOME=/group/$DISKPATH/yaoweili/tmp/.cache/
export TORCH_HOME=/group/$DISKPATH/yaoweili/tmp/.cache/torch
PYTHON=${PYTHON:-"/data/miniconda3/envs/svd/bin/python"}
export PYTHONPATH=.:$PYTHONPATH

#### node settings
DEBUG=false
if [ $DEBUG == true ]; then
    HOST_GPU_NUM=1
    VISIABLE_DEVICES='0'
    PRE_NAME='DEUBUG'
else
    HOST_GPU_NUM=8
    VISIABLE_DEVICES='0,1,2,3,4,5,6,7'
    PRE_NAME=''
fi



#### hypeparameters
TYPE=CFG0.1_ImgaNet64
BTS=200
TOTAL_BTS=$BTS*$HOST_GPU_NUM
echo "Batch size: "$TOTAL_BTS
LR=1e-4

DATA_DIR=datasets/imagenet64/imagenet64_caffe.txt

SAVE_NAME=$PRE_NAME"0001_LR_"$LR"_BTS_"$TOTAL_BTS"_"$TYPE
SAVE_ROOT=/group/40005/yaoweili/checkpoints
UCG_RATE=0.1
NUM_WORKERS=14

mkdir -p $SAVE_ROOT/$SAVE_NAME
zip -ry $SAVE_ROOT/$SAVE_NAME/code.zip `pwd` -x "`pwd`/.git" > /dev/null 2>&1

# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_SOCKET_TIMEOUT=4200
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
#### run
CUDA_VISIBLE_DEVICES=$VISIABLE_DEVICES mpiexec -n $HOST_GPU_NUM python ./scripts/image_train.py \
--data_dir $DATA_DIR \
--attention_resolutions 32,16,8 \
--class_cond True \
--dropout 0.1 \
--image_size 64 \
--num_channels 192 \
--num_res_blocks 3 \
--learn_sigma True \
--num_head_channels 64 \
--resblock_updown True \
--use_new_attention_order True \
--use_fp16 True \
--use_scale_shift_norm True \
--diffusion_steps 1000 \
--noise_schedule cosine \
--rescale_learned_sigmas False \
--lr $LR \
--batch_size $BTS \
--ucg_rate $UCG_RATE \
--num_workers $NUM_WORKERS \
--outputs_dir $SAVE_ROOT/$SAVE_NAME 2>&1 | tee logs/$SAVE_NAME.log


# 直接删除cudnn
# cd /usr/local/cuda-12.1/lib64
# sudo rm -f libcudnn*
# cd /usr/local/cuda-12.1/include
# sudo rm -f cudnn*