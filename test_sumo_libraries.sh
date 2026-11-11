#!/bin/bash
#SBATCH --job-name=test_sumo
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/test_sumo-%j.out
#SBATCH --error=logs/test_sumo-%j.err

cd "$SLURM_SUBMIT_DIR"

module load python/3.11.5
source ~/envs/thesis/bin/activate

export SUMO_HOME="$HOME/sumo_install/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

echo "===== NODE ====="
hostname

echo "===== PYTHON ====="
which python

echo "===== SUMO ====="
which sumo

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH"

echo "===== LIBRARIES ====="
ldd "$SUMO_HOME/bin/sumo" | grep -E 'atomic|not found'

echo "===== SUMO VERSION ====="
sumo --version
