#!/bin/sh
#this file is used for load clipp.h file from github.com

mkdir -p include
cd include
wget https://raw.githubusercontent.com/muellan/clipp/v1.2.3/include/clipp.h

# or from fastgit.org mirror:
# wget https://raw.fastgit.org/muellan/clipp/v1.2.3/include/clipp.h
cd ../
