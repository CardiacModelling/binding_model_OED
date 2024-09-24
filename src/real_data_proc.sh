#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3g
#SBATCH --time=04:00:00

module use $HOME/.local/easybuild/modules/all
module load gcc-uoneasy/12.3.0
module load SUNDIALS/6.6.0-foss-2023a
module load anaconda-uoneasy/2023.09-0
source ~/.bashrc
conda activate env

# Define experiment
compound=$1
herg='2019_37C'
models="['1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13']"
fit_reps=10

# Convert models string for bash looping
models_bash=(${models//[\[\]\'\,]/ })

# Set output directory
dir="outputs_real/${compound}/"

# Load in real data and save
python src/syncro_export.py -o ${dir} -d ${compound}

# Fit splines to new synthetic data control sweeps
echo "Fitting splines..."
python src/fit_spline.py -i ${dir} -o ${dir} -m "real"
