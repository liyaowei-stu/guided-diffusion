import numpy as np
from utils import load_databatch, load_raw_data
from PIL import Image


if __name__=="__main__":
    # Load the data
    probs_path="datasets/imagenet64/imagenet64_pickle/probs/train_data_batch_probs1.npy"
    data_path="datasets/imagenet64/imagenet64_pickle/train_data_batch_1"
    
    X_train, Y_train = load_raw_data(data_path, img_size=64) 
    
    import ipdb;ipdb.set_trace()
    
    probs = np.load(probs_path)
   
    vis_train = X_train[:32].transpose(0,2,3,1)

    vis_train = vis_train.astype(np.uint8)
    
    vis_num, h, w, c=vis_train.shape
    nrow = 4
    ncol = vis_num // nrow
    grid = vis_train.reshape(nrow, ncol, h, w, c)
    grid = np.concatenate(grid, axis=1)  # 沿着列方向拼接
    grid = np.concatenate(grid, axis=1)  # 沿着行方向拼接
    
    Image.fromarray(grid).save("train_0.png")
   
   