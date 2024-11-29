#!/bin/bash

for i in {'bepridil','verapamil','terfenadine'}
do
    sbatch src/real_data_proc.sh $i
done
