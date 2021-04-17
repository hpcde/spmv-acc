#!/bin/bash
#SBATCH -J spmv-hip
#SBATCH -o log-%j.log
#SBATCH -e log-%j.err
#SBATCH -N 1
#SBATCH -n 1
##SBATCH -c 1
#SBATCH --gres=dcu:4
#SBATCH -p PilotCup
#SBATCH -t 00:05:00
#SBATCH --export=ALL

BIN_PATH=./spmv-hip
#BIN_PATH=./Csrsparse # demo

module purge
module load compiler/rocm/3.9.1

#ldd ${BIN_PATH}
ulimit -c unlimited
${BIN_PATH} 3000 3000 0.5
