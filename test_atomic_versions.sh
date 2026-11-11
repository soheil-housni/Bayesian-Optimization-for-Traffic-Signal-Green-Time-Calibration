#!/bin/bash
#SBATCH --job-name=test_atomic
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/test_atomic-%j.out
#SBATCH --error=logs/test_atomic-%j.err

cd "$SLURM_SUBMIT_DIR"

module load python/3.11.5

export SUMO_HOME="$HOME/sumo_install/sumo"
export PATH="$SUMO_HOME/bin:$PATH"

unset LD_LIBRARY_PATH

LIBDIR="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/gcc/x86_64-pc-linux-gnu"

echo "===== NODE ====="
hostname

echo "===== GLIBC ====="
ldd --version | head -n 1

for version in 11 12 13 14; do

    LIBATOMIC="$LIBDIR/$version/libatomic.so.1"

    echo ""
    echo "===== TEST GCC $version ====="
    echo "Library: $LIBATOMIC"

    if [ ! -f "$LIBATOMIC" ]; then
        echo "Library not found"
        continue
    fi

    echo "--- Dependencies ---"
    ldd "$LIBATOMIC"

    echo "--- SUMO test ---"
    env LD_PRELOAD="$LIBATOMIC" "$SUMO_HOME/bin/sumo" --version

    echo "Exit code: $?"

done
