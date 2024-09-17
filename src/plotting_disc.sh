#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=38g
#SBATCH --time=00:10:00

module use $HOME/.local/easybuild/modules/all
module load gcc-uoneasy/12.3.0
module load SUNDIALS/6.6.0-foss-2023a
module load anaconda-uoneasy/2023.09-0
source ~/.bashrc
conda activate env

# Define experiment
compound=$1
true_model=$2
herg='kemp'
models="['1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13']"

# Set output directory
dir="outputs/${compound}/model_${true_model}_${compound}_disc"

# Plot fitting results
python src/post_fit_plot.py -m ${models} -p 'protocols/Milnes_Phil_Trans.mmt' -o ${dir} -e ${herg} -d ${compound}

# Read in protocol details
filename="${dir}/prot_details.csv"
win_list=$(sed -n '1p' "$filename")
IFS=',' read -r -a alt_win <<< "$win_list"
ttime=$(sed -n '2p' "$filename")
alt_win_str="[${alt_win[@]}]"
alt_win_str=$(echo $alt_win_str | sed 's/ /,/g')
wins_str="[[1000,11000],$alt_win_str]"

# Plot fitting results
python src/post_fit_plot.py -m ${models} -p "${dir}/opt_prot.mmt" -t $ttime -b "$wins_str" -o "${dir}/opt_synth_data" -e ${herg} -d ${compound} -c
