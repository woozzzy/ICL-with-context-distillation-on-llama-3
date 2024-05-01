#!/bin/bash
#SBATCH --job-name=job_llama_icl
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16GB
#SBATCH --output=job_%j.out
module load anaconda3 
conda activate llama 
export $(cat /home/hice1/spark868/scratch/ICL-with-context-distillation-on-llama-3/.env | xargs) &&
cd /home/hice1/spark868/scratch/ICL-with-context-distillation-on-llama-3/models 

huggingface-cli login --token $HF_ACCESS_TOKEN
huggingface-cli download meta-llama/Meta-Llama-3-70B --exclude "original/*" --local-dir Meta-Llama-3-70B --cache-dir /home/hice1/spark868/scratch/.cache/huggingface
