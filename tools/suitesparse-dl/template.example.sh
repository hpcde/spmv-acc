#!/bin/sh
#SBATCH -J spmv_cli
#SBATCH -o log-%j.log
#SBATCH -e log-%j.err
#SBATCH -N 1
#SBATCH -n 1
##SBATCH -c 1
#SBATCH --exclusive
#SBATCH -p normal
#SBATCH -t 05:00:00
#SBATCH --export=ALL

APP=./build/bin/spmv_cli

echo running {{.Size}} matrices

# note: you can also use `.Name` and `.Path` fields in `.Mates`.

{{range .Metas}}
${APP} {{.AbsPath}}  -f mm
{{end}}