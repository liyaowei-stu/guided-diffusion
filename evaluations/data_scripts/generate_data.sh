#!/bin/bash



python evaluations/genrate_data.py 2>&1 | tee "logs/extract_data.log"

if [ $? != 0 ]; then
   echo "Fail! Exit with 1"
   exit 1
else
   echo "Success! Exit with 0"
   exit 0
fi
