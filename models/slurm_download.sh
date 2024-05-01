#!/bin/bash
#SBATCH --job-name=job_llama_icl
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --output=output/job_%j.out
module load anaconda3 
conda activate llama 
huggingface-cli login --token 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
cd /home/hice1/spark868/scratch/ICL-with-context-distillation-on-llama-3/models 

huggingface-cli download meta-llama/Meta-Llama-3-70B --exclude "original/*" --local-dir Meta-Llama-3-70B --cache-dir /home/hice1/spark868/scratch/.cache/huggingface
