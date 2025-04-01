#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3g
#SBATCH --time=10:00:00

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
models="['7','10','11','13']"
fit_reps=10

# Convert models string for bash looping
models_bash=(${models//[\[\]\'\,]/ })

# Set output directory
dir="outputs_lowdim_stepvar_flexswp_newprot_fitboth/${compound}/model_${true_model}_${compound}_disc"

# Generate synthetic data
#echo "Generating synthetic data..."
#python src/generate_synthetic_data.py -m ${true_model} -p 'protocols/Milnes_Phil_Trans.mmt' -o ${dir} -e ${herg} -c ${compound}

# Fit splines to synthetic data control sweeps
#echo "Fitting splines..."
#python src/fit_spline.py -i ${dir} -o ${dir}

herg='2019_37C'

# Fit models to synthetic data
#JOB_ID=$(sbatch src/run_multi_fit.sh ${dir} ${compound} ${fit_reps} ${herg} "${models_bash[@]}" | awk '{print $4}')

# Function to check if job is still running
is_job_running() {
    squeue -j $JOB_ID > /dev/null 2>&1
    return $?
}

# Wait for the job to complete
#while is_job_running; do
#    echo "Waiting for fitting job $JOB_ID to complete..."
#    sleep 60
#done

# Optimise protocol
#echo "Optimising protocol..."
#python src/optimise_protocol.py -m ${models} -o ${dir} -e ${herg} -c ${compound}

# Read in protocol details
filename="${dir}/prot_details.csv"
win_list=$(sed -n '1p' "$filename")
IFS=',' read -r -a win <<< "$win_list"
win_list_alt=$(sed -n '2p' "$filename")
IFS_alt=',' read -r -a win_alt <<< "$win_list_alt"
ttime=$(sed -n '3p' "$filename")
t_list=$(sed -n '4p' "$filename")
IFS_t_list=',' read -r -a t_list <<< "$t_list"
win_str="[${win[@]}]"
win_str=$(echo $win_str | sed 's/ /,/g')
alt_win_str="[${win_alt[@]}]"
alt_win_str=$(echo $alt_win_str | sed 's/ /,/g')
t_list_str="[${t_list[@]}]"
t_list_str=$(echo $t_list_str | sed 's/ /,/g')
wins_str="[$win_str,$alt_win_str]"

herg='kemp'

# Generate new synthetic data
#echo "Generating new synthetic data..."
#python src/generate_synthetic_data.py -m ${true_model} -p "${dir}/opt_prot.mmt" -t $ttime -b "$wins_str" -o "${dir}/opt_synth_data" -e ${herg} -c ${compound}

# Fit splines to new synthetic data control sweeps
#echo "Fitting splines..."
#python src/fit_spline.py -i "${dir}/opt_synth_data" -o "${dir}/opt_synth_data" -m "opt" -s "$t_list_str" -t $ttime

herg='2019_37C'

models="['1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13']"
models_bash=(${models//[\[\]\'\,]/ })

# Fit models to synthetic data
JOB_ID=$(sbatch src/run_multi_fit_opt.sh ${dir} ${compound} ${fit_reps} ${herg} "${models_bash[@]}" | awk '{print $4}')

# Wait for the job to complete
while is_job_running; do
    echo "Waiting for fitting job $JOB_ID to complete..."
    sleep 60
done
