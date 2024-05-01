#!/bin/bash
POSITIONAL_ARGS=()
CONFIG="config/config.yaml"
NPROC_PER_NODE=4
BATCH_FILE="output/slurm_job.sh"


while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      CONFIG="$2"
      shift
      shift 
      ;;
    -f|--fsdp)
      ACCELERATE_USE_FSDP=1
      shift 
      ;;
    -t|--torchrun)
      USE_TORCHRUN=1
      shift 
      ;;
    -n|--nproc_per_node)
      NPROC_PER_NODE="$2"
      shift
      shift 
      ;;
    -s|--slurm)
      USE_SLURM=1
      shift 
      ;;
    --clean)
      CLEAN=1
      shift 
      ;;
    -h|--help)
      echo """
Usage: run.sh [OPTIONS]
Options:
  -c, --config <config>       Path to the config file. Default: config/llama-3-8b-qlora.yaml.
  -f, --fsdp                  Use FSDP for training.
  -t, --torchrun              Use torchrun for training. Specify the number of GPUs with --nproc_per_node.
  -n. --nproc_per_node <num>  Number of GPUs to use with torchrun. Default: 4.
  --clean                     Clean the output directory. Does not run the training.
  -s, --slurm                 Dispatch Slurm job. For use on PACE cluster only.
  -h, --help                  Show this help message.
"""
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") 
      shift 
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

RUN_COMMAND=(python main.py --config $CONFIG)

if [ -n "$CLEAN" ]; then
  rm -rf output/*/run_*
  rm data/*.json
  rm output/job_*.out
  rm "$BATCH_FILE"
  echo "Cleaned output directory."
fi

if [ -n "$USE_TORCHRUN" ] && [ -n "$ACCELERATE_USE_FSDP" ]; then
  RUN_COMMAND=(ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=$NPROC_PER_NODE main.py --config $CONFIG)
elif [ -n "$ACCELERATE_USE_FSDP" ]; then
  RUN_COMMAND=(ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 python main.py --config $CONFIG)
elif [ -n "$USE_TORCHRUN" ]; then
  RUN_COMMAND=(torchrun --nproc_per_node=$NPROC_PER_NODE main.py --config $CONFIG)
fi

if [ -n "$USE_SLURM" ]; then  

  if test -f "$BATCH_FILE"; then 
    rm "$BATCH_FILE"
  else 
    touch "$BATCH_FILE"
  fi

  cat <<EOT > "$BATCH_FILE"
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
echo "\$PWD"
echo "${RUN_COMMAND[@]}"
${RUN_COMMAND[@]}
EOT

  sbatch "$BATCH_FILE"
  exit 0
else 
  ${RUN_COMMAND[@]}
  exit 0
fi

exit 1