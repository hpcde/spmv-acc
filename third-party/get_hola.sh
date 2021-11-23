#!/bin/sh
#this file is used for load hola from bitbucket.org

git clone https://bitbucket.org/gpusmack/holaspmv.git
cd holaspmv
git checkout e1df708ac3b8d09efe1c1971f477f2ffde233122
git apply ../holaspmv.patch

# load cub 1.8.0
cd deps/
curl https://github.com/NVlabs/cub/archive/refs/tags/1.8.0.tar.gz -L -o cub-1.8.0.tar.gz
tar -xf cub-1.8.0.tar.gz
ln -sf cub-1.8.0 cub
rm -rf cub-1.8.0.tar.gz

# or use system cub:
# ln -s /usr/local/cuda/include/cub ./
