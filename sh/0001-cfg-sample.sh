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
SAVE_PATH=sample
SAVE_NAME=0001-CFG0.1_ImgaNet64_ema_150000


BTS=256
TOTAL_BTS=$BTS*$HOST_GPU_NUM
echo "Batch size: "$TOTAL_BTS


MODEL_PATH=checkpoints/0001_LR_1e-4_BTS_200*8_CFG0.1_ImgaNet64/ema_0.9999_150000.pt

#### run
CUDA_VISIBLE_DEVICES=$VISIABLE_DEVICES  mpiexec -n $HOST_GPU_NUM python ./scripts/image_sample.py \
--model_path $MODEL_PATH \
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
--num_samples 50000 \
--timestep_respacing 250 \
--batch_size $BTS \
--outputs_dir $SAVE_PATH/$SAVE_NAME 2>&1 | tee logs/sample/$SAVE_NAME.log


# 直接删除cudnn
# cd /usr/local/cuda-12.1/lib64
# sudo rm -f libcudnn*
# cd /usr/local/cuda-12.1/include
# sudo rm -f cudnn*