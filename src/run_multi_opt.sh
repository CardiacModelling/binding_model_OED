#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500m
#SBATCH --time=02:00:00
#SBATCH --array=1-10

module use $HOME/.local/easybuild/modules/all
module load gcc-uoneasy/12.3.0
module load SUNDIALS/6.6.0-foss-2023a
module load anaconda-uoneasy/2023.09-0
source ~/.bashrc
conda activate env

dir="outputs_real_20241114_MA_FP_RT"

python -u src/optimise_protocol_3drug.py -m "['1','3','5','6','7','8','11']" -o ${dir} -e "2024_Joey_sis_25C" -c "['diltiazem','quinidine','chlorpromazine']" -r -n $SLURM_ARRAY_TASK_ID -t 22300 -b "[[1000,3900],[5200,8300]]" -y > ${dir}/opt_out_${SLURM_ARRAY_TASK_ID}_output.txt
