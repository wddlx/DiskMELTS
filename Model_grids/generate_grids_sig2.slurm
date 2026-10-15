#!/bin/bash
#SBATCH --job-name=grids-sig2
#SBATCH --output=%x-%A_%a.out
#SBATCH --account=Pascucci
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb

set -euo pipefail

module load python/3.11
source /home/u23/cyxie/myenv/sg_fit_py311/bin/activate

# Submit from Model_grids/ so OUTPUT_ROOT = Path(".") retains the notebook's
# output location.
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f Generating-Grids.py ]]; then
    echo "Submit this job from the Model_grids directory." >&2
    exit 1
fi

python3 Generating-Grids.py 