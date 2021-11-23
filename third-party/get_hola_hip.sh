#!/bin/sh
#this file is used for load hola from git.hpcer.dev or github

# or mirror from: https://github.com/hpcde/hola-hip.git
git clone https://git.hpcer.dev/PRA/hola-hip.git holahip

cd holahip
git checkout 20b1669b9c397aca8f629099eb70890c53940da0

echo "completed."
echo -e "\033[31m Please edit file hip-hola/utils/common.hpp to set the WARP_SIZE to the desired value
(usually 32 for NVIDIA GPU and 64 for AMD GPU). \033[0m" 
