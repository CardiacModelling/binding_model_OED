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
herg='2019_37C'
models="['1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13']"
fit_reps=10

# Convert models string for bash looping
models_bash=(${models//[\[\]\'\,]/ })

# Set output directory
dir="outputs/${compound}/model_${true_model}_${compound}_disc"

# Generate synthetic data
echo "Generating synthetic data..."
python src/generate_synthetic_data.py -m ${true_model} -p 'protocols/Milnes_Phil_Trans.mmt' -o ${dir} -e ${herg} -c ${compound}

# Fit splines to synthetic data control sweeps
echo "Fitting splines..."
python src/fit_spline.py -i ${dir} -o ${dir}

herg='kemp'

# Fit models to synthetic data
JOB_ID=$(sbatch src/run_multi_fit.sh ${dir} ${compound} ${fit_reps} ${herg} "${models_bash[@]}" | awk '{print $4}')

# Function to check if job is still running
is_job_running() {
    squeue -j $JOB_ID > /dev/null 2>&1
    return $?
}

# Wait for the job to complete
while is_job_running; do
    echo "Waiting for fitting job $JOB_ID to complete..."
    sleep 60
done

# Optimise protocol
echo "Optimising protocol..."
python src/optimise_protocol.py -m ${models} -o ${dir} -e ${herg} -c ${compound}

# Read in protocol details
filename="${dir}/prot_details.csv"
win_list=$(sed -n '1p' "$filename")
IFS=',' read -r -a alt_win <<< "$win_list"
ttime=$(sed -n '2p' "$filename")
alt_win_str="[${alt_win[@]}]"
alt_win_str=$(echo $alt_win_str | sed 's/ /,/g')
wins_str="[[1000,11000],$alt_win_str]"

herg='2019_37C'

# Generate new synthetic data
echo "Generating new synthetic data..."
python src/generate_synthetic_data.py -m ${true_model} -p "${dir}/opt_prot.mmt" -t $ttime -b "$wins_str" -o "${dir}/opt_synth_data" -e ${herg} -c ${compound}

# Fit splines to new synthetic data control sweeps
echo "Fitting splines..."
python src/fit_spline.py -i "${dir}/opt_synth_data" -o "${dir}/opt_synth_data" -m "opt"

herg='kemp'

# Fit models to synthetic data
JOB_ID=$(sbatch src/run_multi_fit_opt.sh ${dir} ${compound} ${fit_reps} ${herg} "${models_bash[@]}" | awk '{print $4}')

# Wait for the job to complete
while is_job_running; do
    echo "Waiting for fitting job $JOB_ID to complete..."
    sleep 60
done
