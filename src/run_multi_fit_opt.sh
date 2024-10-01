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
protocol=${dir}/opt_prot.mmt
shift 4
models=("$@")

# Read in protocol details
filename=${dir}/prot_details.csv
win_list=$(sed -n '1p' "$filename")
IFS=',' read -r -a win <<< "$win_list"
win_list_alt=$(sed -n '2p' "$filename")
IFS_alt=',' read -r -a win_alt <<< "$win_list_alt"
ttime=$(sed -n '3p' "$filename")
win_str="[${win[@]}]"
win_str=$(echo $win_str | sed 's/ /,/g')
alt_win_str="[${win_alt[@]}]"
alt_win_str=$(echo $alt_win_str | sed 's/ /,/g')
wins_str="[$win_str,$alt_win_str]"

python -u src/fit_models.py -r ${repeats} -m ${models[SLURM_ARRAY_TASK_ID-1]} -p ${herg_pars} -v ${protocol} -o ${dir}/opt_synth_data -c ${compound} > ${dir}/opt_synth_data/fitting_output/model_${models[SLURM_ARRAY_TASK_ID-1]}_output.txt -t $ttime -b "$wins_str"
