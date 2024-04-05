import math
import random
import torch
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import json

def load_data(
    *,
    data_dir,
    probs_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    num_workers=0,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data file")
    
    with open(data_dir, 'r') as f:
        sample_data = f.readlines()
        sample_data = [x.strip() for x in sample_data if x.strip()]
    sample_data = sorted(sample_data, key=lambda s: s.split('/')[-1].split('.')[0])

   

    with open(probs_dir, 'r') as f:
        probs_data = f.readlines()
        probs_data = [x.strip() for x in probs_data if x.strip()]
    probs_data = sorted(probs_data, key=lambda s: s.split('/')[-1].split('.')[0])


    all_files = [x.split(",")[0] for x in sample_data]


    # # 初始化一个空字典来存储每个目录下的文件数
    # sample_data_counts = {}
    # # 遍历列表，统计每个目录下的文件数
    # for s in all_files:
    #     # 提取目录名
    #     dir_name = s.split('/')[-2]
    #     # 如果这个目录名还没有在字典中，就添加它并设置初始值为 1
    #     # 如果这个目录名已经在字典中，就将它的值加 1
    #     sample_data_counts[dir_name] = sample_data_counts.get(dir_name, 0) + 1 
    # with open('imagenet1k_every_class_num.json', 'w') as handle:
    #     json.dump(sample_data_counts, handle)

    with open('imagenet1k_every_class_num.json', 'r') as f:
        sample_data_counts = json.load(f)

    probs_classes = [[] for _ in range(len(sample_data_counts))]
    last_num = 0
    for cond_idx, cond_num in sample_data_counts.items():
        idx = int(cond_idx)-1
        current_num = last_num + int(cond_num)
        probs_classes[idx].extend(probs_data[last_num: current_num])
        last_num += int(cond_num)

    import ipdb; ipdb.set_trace()
    Total_num = sum([len(prob) for prob in probs_classes])
    assert Total_num == len(probs_data), 'data error, please check total number'


    with open('imagenet1k_f2p_sorted_classes.json', 'w') as handle:
        json.dump(probs_classes, handle)


    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [x.split(",")[1] for x in sample_data]

    # sample_idx = []
    # for cond, cond_num in sample_data_counts:
    #         for idx in cond_num:

    #             sample_idx.append(cond)


    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    sampler = ClassCondSampler(all_files, probs_data, sample_data_counts)


    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    # while True:
    #     yield from loader




class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        # np.uint8(np.transpose(((arr+1)/2)*255, [1,2,0]))

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class ClassCondSampler(Sampler):
    def __init__(self, data_source, probs_data, sample_data_counts) -> None:
        self.data_source = data_source
        self.class_prob = probs_data
        self.sample_data_counts = sample_data_counts


    def __iter__(self):
        sample_idx = []
        for cond, cond_num in self.sample_data_counts:
            for idx in cond_num:
                sample_idx.append(cond)

        return iter(sample_idx)

    def __len__(self):
        return len(self.data_source)
    

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


if __name__=="__main__":
    
    data_dir="datasets/imagenet64/imagenet64_caffe.txt"
    probs_dir="datasets/imagenet64/imagenet_f2p.txt"
    batch_size=8
    image_size=64
    class_cond=True
    num_workers=0
    print("creating data loader...")
    data = load_data(
        data_dir=data_dir,
        probs_dir=probs_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        num_workers=num_workers,
    )
