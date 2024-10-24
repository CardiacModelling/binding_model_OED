#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=06:00:00
#SBATCH --array=1-15

module use $HOME/.local/easybuild/modules/all
module load gcc-uoneasy/12.3.0
module load SUNDIALS/6.6.0-foss-2023a
module load anaconda-uoneasy/2023.09-0
source ~/.bashrc
conda activate env

dir=$1
compound=$2
repeats=$3
herg_pars=$4
herg_pars="${herg_pars}.csv"
protocol=protocols/Milnes_16102024_MA1_FP_RT.mmt
shift 4
models=("$@")

# Define protocol details
ttime=15350
#wins_str="[[1350,4690],[8020,11350],[17930,27930]]"
wins_str="[[1350,11350]]"

python -u src/fit_models.py -r ${repeats} -m ${models[SLURM_ARRAY_TASK_ID-1]} -p ${herg_pars} -v ${protocol} -o ${dir} -c ${compound} > ${dir}/fitting_output/model_${models[SLURM_ARRAY_TASK_ID-1]}_output.txt -t $ttime -b "$wins_str"
