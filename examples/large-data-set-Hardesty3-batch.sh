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

module purge
module load compiler/rocm/3.9.1

echo $SLURM_JOB_NODELIST

#ldd ${BIN_PATH}
ulimit -c unlimited

# rows, cols, non-zeros, percent of non-zeros, average non-zeros per row
echo "Hardesty3: (381,689	381,689	37,464,962	0.02572%	98.15573)"
${BIN_PATH} ./large-data-set/Hardesty3.csr
