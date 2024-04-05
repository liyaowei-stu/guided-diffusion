import pickle
import numpy as np
import glob
import os,sys
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_flie, img_size=64):
    
    d = unpickle(data_flie)
    
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    # y = [i-1 for i in y]
    data_size = x.shape[0]
    
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2) # b,h,w,c ==> b,c,h,w

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]

    # arr2img(X_train[0].transpose(1,2,0),"train0.jpg")
    return X_train, Y_train, mean_image


def load_raw_data(data_flie, img_size=64):
    
    d = unpickle(data_flie)
    
    x = d['data']
    y = d['labels']

    # Labels are indexed from 1, shift it so that indexes start at 0
    # y = [i-1 for i in y]
    data_size = x.shape[0]
    
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2) # b,h,w,c ==> b,c,h,w

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]

    return X_train, Y_train



if __name__ == "__main__":
    image_paths = "datasets/imagenet64/imagenet64_pickle/"
    data_flies = sorted(glob.glob(os.path.join(image_paths, 'train_data_batch_*')))
    X_train_all, Y_train_all = [], []
    for data_flie in data_flies:
        import ipdb; ipdb.set_trace()
        X_train, Y_train, mean_image = load_databatch(data_flie, img_size=64)
        Image.fromarray(np.uint8((((X_train[0].transpose(1,2,0) + 1.) / 2)*255.))).save("train0.png")
        
        X_train_all.append(X_train)
        Y_train_all.append(Y_train)