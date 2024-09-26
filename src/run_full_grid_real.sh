#!/bin/bash

for i in {'bepridil','quinidine'}
do
    sbatch src/real_data_proc.sh $i
done
