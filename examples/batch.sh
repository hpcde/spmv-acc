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

echo $SLURM_JOB_NODELIST

#ldd ${BIN_PATH}
ulimit -c unlimited

echo "23560 * 23560"
${BIN_PATH} ./data/af23560.csr

echo "13436 * 13436"
${BIN_PATH} ./data/bayer10.csr

echo "11948 * 11948"
${BIN_PATH} ./data/bcsstk18.csr

echo "9540 * 9540"
${BIN_PATH} ./data/coater2.csr

echo "8192 * 8192"
${BIN_PATH} ./data/dw4096.csr

echo "14734 * 14734"
${BIN_PATH} ./data/epb1.csr

echo "6001 * 6001"
${BIN_PATH} ./data/exdata_1.csr

echo "9506 * 9506"
${BIN_PATH} ./data/nemeth03.csr

echo "15575 * 15575"
${BIN_PATH} ./data/poli_large.csr

echo "7602 * 7602"
${BIN_PATH} ./data/rajat03.csr
