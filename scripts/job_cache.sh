#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH -e slurm-%j.err
#SBATCH --gres=gpu_mem:1024
#SBATCH -o slurm-%j.out

set -e

source /home/users/2ai12_1/miniconda3/etc/profile.d/conda.sh
conda activate benchseg
echo "python: $(which python)"
cd /home/users/2ai12_1/fetai/shell_refineV2

python -u cache.py \
    --images-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Images \
    --labels-dir /home/users/2ai12/Desktop/Datasets_shared/FetalUltrasound/Labels/Head \
    --out-dir cache

echo "=== cache build done, running tests ==="

python -u test_cache.py --round-trip-cases 2
python -u test_geometry.py
