#!/bin/bash
#SBATCH -job_llama_fine_tune                    # Job name
#SBATCH --gres=gpu:H100:4                       # GPU
#SBATCH -N4 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=16G                       # Memory per core
#SBATCH -t02:00:00                              # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3                           # Load module dependencies
conda activate llama
srun ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 main.py --config config/llama-3-70b-qlora.yaml