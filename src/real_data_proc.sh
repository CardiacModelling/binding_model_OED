#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2g
#SBATCH --time=01:00:00

module use $HOME/.local/easybuild/modules/all
module load gcc-uoneasy/12.3.0
module load SUNDIALS/6.6.0-foss-2023a
module load anaconda-uoneasy/2023.09-0
source ~/.bashrc
conda activate env

# Get drug
compound=$1

# Set output directory
dir="outputs_real/${compound}"

# Load in real data and save
python src/syncro_export.py -o ${dir} -d ${compound}

# Fit splines to new synthetic data control sweeps
echo "Fitting splines..."
python src/fit_spline.py -i ${dir} -o ${dir} -m "real"

# Function to check if job is still running
is_job_running() {
    squeue -j $JOB_ID > /dev/null 2>&1
    return $?
}

# Define models and fitting reps
herg='2024_Joey_sis_25C'
models="['1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13']"
fit_reps=10

# Convert models string for bash looping
models_bash=(${models//[\[\]\'\,]/ })

# Fit models to synthetic data
JOB_ID=$(sbatch src/run_multi_fit_real.sh ${dir} ${compound} ${fit_reps} ${herg} "${models_bash[@]}" | awk '{print $4}')

# Wait for the job to complete
while is_job_running; do
    echo "Waiting for fitting job $JOB_ID to complete..."
    sleep 60
done

ttime=32000
wins_str="[[1350,4690],[8020,11350],[17930,27930]]"
python src/post_fit_plot.py -m ${models} -p "protocols/3_drug_protocol.mmt" -t $ttime -b "$wins_str" -o "${dir}" -e ${herg} -d ${compound}
