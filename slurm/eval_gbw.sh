#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcyi-delta-gpu
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --job-name=eval_gbw
#SBATCH --output=log/eval_gbw_%j.out

# Load modules and activate environment
conda deactivate || true
module purge
module reset
module load anaconda3_gpu
source activate fairseq

MODEL_PATH=examples/language_model/adaptive_lm_gbw_huge
python -u fairseq_cli/eval_lm.py \
    $MODEL_PATH/data-bin \
    --path $MODEL_PATH/model.pt \
    --sample-break-mode eos \
    --max-tokens 2048 \
    --save-layers -1 \
    --save-probs \
    --batch-size 64 \
    --results-path $MODEL_PATH/results \
    --gen-subset test \