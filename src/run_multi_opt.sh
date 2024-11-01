#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=01:00:00
#SBATCH --array=1-5

module use $HOME/.local/easybuild/modules/all
module load gcc-uoneasy/12.3.0
module load SUNDIALS/6.6.0-foss-2023a
module load anaconda-uoneasy/2023.09-0
source ~/.bashrc
conda activate env

dir="outputs_real_16102024_MA1_FP_RT"

python -u src/optimise_protocol_3drug.py -m "['7','10','11','12']" -o ${dir} -e "2024_Joey_sis_25C" -c "['bepridil','verapamil','terfenadine']" -r -n $SLURM_ARRAY_TASK_ID > ${dir}/opt_out_${SLURM_ARRAY_TASK_ID}_output.txt
