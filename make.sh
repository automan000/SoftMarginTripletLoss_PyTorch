#!/usr/bin/env bash

#NVCC=/usr/local/cuda/bin/nvcc
#
#
cd online_triplet_loss/src

echo "Compiling online_triplet_loss.cpp..."
gcc -c -o online_triplet_loss.o online_triplet_loss.cpp -fPIC -std=c++11
echo "python build.py"
cd ../
python build.py
