#!/bin/sh
set -e

#this file is used for load hola from git.hpcer.dev or github

# or mirror from: https://github.com/hpcde/hola-hip.git
git clone https://git.hpcer.dev/PRA/hola-hip.git holahip

cd holahip
git checkout 08a2b06ceb5b5605dabf6842b90ab63b931a5607

echo "completed."
echo "\033[31m Please edit file holahip/hip-hola/utils/common.hpp to set the WARP_SIZE to the desired value
(usually 32 for NVIDIA GPU and 64 for AMD GPU). \033[0m" 
