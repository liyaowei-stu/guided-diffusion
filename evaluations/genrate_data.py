import numpy as np
from utils import load_databatch, load_raw_data
from PIL import Image
import os, sys
import time
from tqdm import tqdm
import argparse


if __name__=="__main__":
    X_train_all, Y_train_all, Probs = [], [], []

    for flag in range(1,11):
    # Load the data
        probs_path=f"datasets/imagenet64/imagenet64_pickle/probs/train_data_batch_probs{flag}.npy"
        data_path=f"datasets/imagenet64/imagenet64_pickle/train_data_batch_{flag}"
        
        X_train, Y_train = load_raw_data(data_path, img_size=64) 
        probs = np.load(probs_path)

        X_train_all.append(X_train)
        Y_train_all.append(Y_train)
        Probs.append(probs)
   
    X_train_all = np.concatenate(X_train_all, axis=0)
    Y_train_all = np.concatenate(Y_train_all, axis=0)
    Probs =  np.concatenate(Probs, axis=0)

    num_samples = len(Y_train_all)

    save_dir = "/group/40033/public_datasets/imagenet64"
    
    # import ipdb; ipdb.set_trace()

    label_idx_list =[0]*1000
    # import ipdb; ipdb.set_trace()

    for sample_idx in tqdm(range(num_samples)):
        
        vis = X_train_all[sample_idx].transpose(1,2,0)
        vis = vis.astype(np.uint8)

        label = Y_train_all[sample_idx]
        probs_ = Probs[sample_idx]

        label_dir = f"{save_dir}/raw/{label:04d}"

        
        label_idx = label_idx_list[label - 1]
        sample_dir = f"{label_dir}/{label:04d}_{label_idx:05d}.png"

        try:
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)

            record_dir = f"{save_dir}/imagenet_f2p.txt"
            output_string = f"{sample_dir},{probs_}"
            with open(record_dir, 'a') as file:
                file.write(output_string + '\n') 

            Image.fromarray(vis).save(sample_dir)

            label_idx_list[label - 1] += 1 

            if (sample_idx+1)% 100000 ==0:
                print("Sequence {0} finished.".format(sample_idx+1))

        except:
            print(f"Sequence {sample_dir} failed.")


    
   
   