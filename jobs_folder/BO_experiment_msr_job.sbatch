#!/bin/bash

#SBATCH --job-name=msr
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-8%9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "$SLURM_SUBMIT_DIR"

module load python/3.11.5
source ~/envs/thesis/bin/activate


export SUMO_HOME="$HOME/sumo_install/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

export LD_PRELOAD="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/gcc/x86_64-pc-linux-gnu/13/libatomic.so.1"

echo "Python : $(which python)"
python --version

echo "SUMO : $(which sumo)"
sumo --version


python -u bo_experiment_scripts.py --method msr --initial_lengthscale 0.1 --scaled --experiment_id $SLURM_ARRAY_TASK_ID
