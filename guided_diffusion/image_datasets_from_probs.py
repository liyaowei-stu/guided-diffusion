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
    sample_alpah=5,
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


    all_files = [x.split(",")[0] for x in sample_data]

    with open(probs_dir, 'r') as f:
        file_probs_classes = json.load(f)
    
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        classes = [x.split(",")[1] for x in sample_data]

    flat_probs = [item for sublist in file_probs_classes for item in sublist]
    dataset = ImageDataset(
        image_size,
        all_files,
        flat_probs=flat_probs,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )

    class_cond_sampler = ClassCondSampler(all_files, file_probs_classes, sample_alpah)


    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=class_cond_sampler, num_workers=num_workers, drop_last=False
        )

    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        flat_probs=None,
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
        self.flat_probs = None if flat_probs is None else flat_probs[shard:][::num_shards]

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

        # probs = float(self.flat_probs[idx].split(",")[-1])

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # out_dict["probs"] = probs
        return np.transpose(arr, [2, 0, 1]), out_dict


class ClassCondSampler(Sampler):
    def __init__(self, data_source, file_probs_classes, sample_alpah=5) -> None:
        self.data_source = data_source
        self.file_probs_classes = file_probs_classes
        self.sample_alpah = sample_alpah


    def __iter__(self):
        probs_classes = self.file_probs_classes
        sample_list = []
        for cond_idx in range(len(probs_classes)):
            file_probs = probs_classes[cond_idx]
            classes_probs = [float(probs.split(",")[-1])  for probs in file_probs]
            classes_probs = np.array(classes_probs)
            classes_probs = np.power(classes_probs, self.sample_alpah)
            classes_probs = classes_probs / np.sum(classes_probs)
            select_idx = np.random.choice(np.arange(len(classes_probs)), size=len(classes_probs), p=classes_probs, replace=True)
            select_idx = select_idx + len(sample_list)
            sample_list.extend(select_idx.tolist())

        return iter(sample_list)

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
    probs_dir="datasets/imagenet64/imagenet1k_f2p_sorted_classes.json"
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

    for i, batch in enumerate(data):
        batch = batch
        pass 