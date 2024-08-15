#!/bin/bash

for i in {'verapamil','bepridil','dofetilide','chlorpromazine'}
do
    for j in {'1','2','2i','3','4','5','5i','6','7','8','9','10','11','12','13'}
    do
    	sbatch src/whole_procedure.sh $i $j
        sbatch src/whole_procedure_disc.sh $i $j
    done
done
