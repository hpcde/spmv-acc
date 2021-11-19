#!/bin/sh
#this file is used for load hola from bitbucket.org

wget https://bitbucket.org/gpusmack/holaspmv/get/master.tar.gz

tar -xf gpusmack-holaspmv-e1df708ac3b8.tar.gz
mv gpusmack-holaspmv-e1df708ac3b8 holaspmv

#load cub 1.8.0
cd holaspmv/deps/
wget https://github.com/NVlabs/cub/archive/refs/tags/v1.8.0.tar.gz
tar -xf v1.8.0.tar.gz
mv cub-v1.8.0 cub
cd ../..

#todo: auto modify dVector.h dCSR.h dCSR.cpp