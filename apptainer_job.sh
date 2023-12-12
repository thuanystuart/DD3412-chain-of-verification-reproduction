#!/bin/bash
#SBATCH --gpus 1 -C "thin" -t 48:00:00


echo "Hello cluster computing world!"

echo "JOB: ${SLURM_JOB_ID}"

echo "The following is RAM info."
free -h

echo "The following is GPU info."
nvidia-smi

echo "Launching experiments with apptainer."

echo "PYTHONPATH is: $PYTHONPATH"

echo "Current working directory is: $(pwd)"

apptainer exec --nv my_apptainer.sif ./scripts/wikidata.sh