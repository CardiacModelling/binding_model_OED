#!/bin/bash

for i in {'terfenadine','bepridil','verapamil'}
do
    sbatch src/real_data_proc.sh $i
done
