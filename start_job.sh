#!/bin/bash
#SBATCH -job_llama_fine_tune                    # Job name
#SBATCH --gres=gpu:H100:1                       # GPU
#SBATCH -N4 --ntasks-per-node=4                 # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=4G                        # Memory per core
#SBATCH -t01:00:00                              # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
cd $SLURM_SUBMIT_DIR                            # Change to working directory

module load anaconda3                           # Load module dependencies
conda activate llama
srun python main.py --m train                   # Example Process