#!/bin/bash

gpu_id=7
index=7

CUDA_VISIBLE_DEVICES=$gpu_id python evaluations/generate_p.py \
datasets/imagenet64/imagenet64_pickle $index 2>&1 | tee "logs/"$index"_extract_p.log"

if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
