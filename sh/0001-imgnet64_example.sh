DISKPATH=40034
# DISKPATH=30098
export HF_HOME=/group/$DISKPATH/yaoweili/tmp/.cache/
export TORCH_HOME=/group/$DISKPATH/yaoweili/tmp/.cache/torch


# sample 64*64
SAMPLE_FLAGS="--batch_size 4 --num_samples 100 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"

python scripts/classifier_sample.py --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt --batch_size 4 --num_samples 100 --timestep_respacing 250


# classifier train
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 8 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 64 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

mpiexec -n N python scripts/classifier_train.py --data_dir datasets/imagenet64/raw $TRAIN_FLAGS $CLASSIFIER_FLAGS


# diffusion train

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --num_head_channels 64 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 2048"

python scripts/image_train.py --data_dir datasets/imagenet64/raw --attention_resolutions 32,16,8 --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --num_head_channels 64 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True --diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --lr 3e-4 --batch_size 8 --ucg_rate 0.1 --outputs_dir /group/40033/public_datasets/WebVid_process/cfg_step/Debug0001

"

## eval
python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz datasets/admnet_imagenet64.npz 

