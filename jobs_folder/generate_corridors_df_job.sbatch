#!/bin/bash

#SBATCH --job-name=generate_corridors_df
#SBATCH --time=010:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

cd "$SLURM_SUBMIT_DIR"

module load python/3.11.5
source ~/envs/thesis/bin/activate

python -u complete_connections_corridors_df_generation_script.py