#!/bin/sh
#SBATCH -J spmv-bench
#SBATCH -o log-%j.log
#SBATCH -e log-%j.err
#SBATCH -N 1
#SBATCH -n 1
##SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -p lzhgnormal
#SBATCH -t 15:00:00
#SBATCH --export=ALL

echo $SLURM_JOB_NODELIST
which rocm-smi && rocm-smi
which nvidia-smi && nvidia-smi

APP=./cmake-build-release/bin/spmv-gpu-benchmark

echo running {{.Size}} matrices

# note: you can also use `.MtxName`, `FileName` and `.Path` fields in `.Mates`.

{{range .Metas}}
${APP} {{.AbsPath}} -f bin
{{end}}