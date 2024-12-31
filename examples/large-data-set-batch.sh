#!/bin/bash
#SBATCH -J spmv-cli
#SBATCH -o log-%j.log
#SBATCH -e log-%j.err
#SBATCH -N 1
#SBATCH -n 1
##SBATCH -c 1
#SBATCH --gres=dcu:4
#SBATCH -p PilotCup
#SBATCH -t 00:05:00
#SBATCH --export=ALL

BIN_PATH=./spmv-cli

module purge
module load compiler/rocm/3.9.1

echo $SLURM_JOB_NODELIST

#ldd ${BIN_PATH}
ulimit -c unlimited

# rows, cols, non-zeros, percent of non-zeros, average non-zeros per row
echo "boneS10: (914,898/914,898 28,191,660 0.00337% 30.81)"
${BIN_PATH} ./large-data-set/boneS10.csr -f csr

echo "Bump_2911: (2,911,419/2,911,419 65,320,659 0.00077% 22.44)"
${BIN_PATH} ./large-data-set/Bump_2911.csr -f csr

echo "Cube_Coup_dt6: (2,164,760/2,164,760 64,685,452 0.00138% 29.88)"
${BIN_PATH} ./large-data-set/Cube_Coup_dt6.csr -f csr

echo "dielFilterV3real: (1,102,824/1,102,824 45,204,422 0.00372% 40.99)"
${BIN_PATH} ./large-data-set/dielFilterV3real.csr -f csr

echo "Ga41As41H72: (268,096/268,096 9,378,286 0.01305% 34.98)"
${BIN_PATH} ./large-data-set/Ga41As41H72.csr -f csr

echo "Hardesty3: (8,217,820/7,591,564 40,451,632 0.00006% 4.92)"
${BIN_PATH} ./large-data-set/Hardesty3.csr -f csr

echo "largebasis: (440,020/440,020 5,560,100 0.00287% 12.64)"
${BIN_PATH} ./large-data-set/largebasis.csr -f csr

echo "RM07R: (381,689/381,689 37,464,962 0.02572% 98.16)"
${BIN_PATH} ./large-data-set/RM07R.csr -f csr

echo "TSOPF_RS_b2383: (38,120/38,120 16,171,169 1.11285% 424.22)"
${BIN_PATH} ./large-data-set/TSOPF_RS_b2383.csr -f csr

echo "vas_stokes_2M: (2,146,677/2,146,677 65,129,037 0.00141% 30.34)"
${BIN_PATH} ./large-data-set/vas_stokes_2M.csr -f csr
