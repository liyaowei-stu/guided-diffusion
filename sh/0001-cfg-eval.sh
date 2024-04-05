# ddpm 5w w. ema
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_050000/samples_50000x64x64x3.npz --batch_size 128

# ddpm 5w w.o. ema
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_050000/samples_50000x64x64x3.npz --batch_size 128


# ddpm w. ema 6w
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_060000/samples_50000x64x64x3.npz --batch_size 128

# ddim w. ema 6w
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_060000_ddim/samples_50000x64x64x3.npz --batch_size 128

# official iddpm
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz datasets/iddpm_imagenet64.npz --batch_size 128

# official adm
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz datasets/admnet_imagenet64.npz --batch_size 128

# sample using ddpm on guided diffusion official ckpt
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-gudied_diffusion_iddpm/samples_50000x64x64x3.npz --batch_size 128


# ddpm w. ema 7w
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_070000/samples_50000x64x64x3.npz --batch_size 128

# ddpm w. ema 8w
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_080000/samples_50000x64x64x3.npz --batch_size 128

# ddpm w. ema 10w
# python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_100000/samples_50000x64x64x3.npz --batch_size 128

# ddpm w. ema 15w
python evaluations/evaluator.py datasets/VIRTUAL_imagenet64_labeled.npz sample/0001-CFG0.1_ImgaNet64_ema_150000/samples_50000x64x64x3.npz --batch_size 128
