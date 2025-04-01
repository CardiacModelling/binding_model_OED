#!/bin/bash

for i in {'diltiazem','chlorpromazine','quinidine'}
do
    sbatch src/real_data_proc.sh $i
done
