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
echo "boneS10: (914,898	914,898	40,878,708	0.00488%	44.68116)"
${BIN_PATH} ./large-data-set/boneS10.csr

echo "Cube_Coup_dt6: (2,911,419	2,911,419	127,729,899	0.00151%	43.87204)"
${BIN_PATH} ./large-data-set/Cube_Coup_dt6.csr

echo "Ga41As41H72: (2,164,760	2,164,760	124,406,070	0.00265%	57.46876)"
${BIN_PATH} ./large-data-set/Ga41As41H72.csr

echo "largebasis: (1,102,824	1,102,824	89,306,020	0.00734%	80.97939)"
${BIN_PATH} ./large-data-set/largebasis.csr

echo "TSOPF_RS_b2383: (268,096	268,096	18,488,476	0.02572%	68.96215)"
${BIN_PATH} ./large-data-set/TSOPF_RS_b2383.csr

echo "Bump_2911: (8,217,820	7,591,564	40,451,632	0.00006%	5.32850)"
${BIN_PATH} ./large-data-set/Bump_2911.csr

echo "dielFilterV3real: (440,020	440,020	5,240,084	0.00271%	11.90874)"
${BIN_PATH} ./large-data-set/dielFilterV3real.csr

echo "Hardesty3: (381,689	381,689	37,464,962	0.02572%	98.15573)"
${BIN_PATH} ./large-data-set/Hardesty3.csr

echo "RM07R: (38,120	38,120	16,171,169	1.11285%	424.21744)"
${BIN_PATH} ./large-data-set/RM07R.csr

echo "vas_stokes_2M: (2,146,677	2,146,677	65,129,037	0.00141%	30.33947)"
${BIN_PATH} ./large-data-set/vas_stokes_2M.csr
