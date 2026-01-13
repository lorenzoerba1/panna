#!/bin/bash
# SBATCH script to run datasets_test.py as an array job

#SBATCH --job-name=scal_test
#SBATCH --output=logs/slurm-datasets-%A_%a.out
#SBATCH --error=logs/slurm-datasets-%A_%a.err
#SBATCH --partition lovelace
#SBATCH --mem 0
#SBATCH --exclusive
#SBATCH --time 480
#SBATCH --array=0-5


# List of dataset filenames (must match the paths used by the test script)
paths=(
  "fashion-mnist-784-euclidean.hdf5"
  "glove-100-angular.hdf5"
  "nytimes-256-angular.hdf5"
  "gist-960-euclidean.hdf5"
  "simplewiki-openai-3072-normalized.hdf5"
  "deep-image-96-angular.hdf5"
)

path=${paths[$SLURM_ARRAY_TASK_ID]}

export APPTAINER_BIND="results:/results,datasets:/datasets,tests:/tests"
srun panna_ultimate.sif tests/scalability_datasets_test.py --path "$path"
