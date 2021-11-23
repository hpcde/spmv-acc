#!/bin/sh
#this file is used for load hola from bitbucket.org

git clone https://git.hpcer.dev/PRA/hola-hip.git

mv hola-hip holahip

cd holahip/hip-hola

# modify d_csr.h
sed -i 's/^.*alloc/\/\/&/g' d_csr.h
sed -i 's/^.*convert/\/\/&/g' d_csr.h

# modify d_csr.cpp
sed -i 's/^.*/\/\/&/g' d_csr.cpp
