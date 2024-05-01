#!/bin/bash
POSITIONAL_ARGS=()
CONFIG="config/llama-3-8b-qlora.yaml"
NPROC_PER_NODE=4

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
    -c|--clean)
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
  -c, --clean                 Clean the output directory. Does not run the training.
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
  rm -rf "output/*"
  echo "Cleaned output directory."
  exit 0
elif [ -n "$USE_TORCHRUN" ] && [ -n "$ACCELERATE_USE_FSDP" ]; then
  RUN_COMMAND=(ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=$NPROC_PER_NODE main.py --config $CONFIG)
elif [ -n "$ACCELERATE_USE_FSDP" ]; then
  RUN_COMMAND=(ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 python main.py --config $CONFIG)
elif [ -n "$USE_TORCHRUN" ]; then
  RUN_COMMAND=(torchrun --nproc_per_node=$NPROC_PER_NODE main.py --config $CONFIG)
fi

if [ -n "$USE_SLURM" ]; then
  sbatch -job_llama_fine_tune --gres=gpu:H100:4 -N4 --ntasks-per-node=4 --mem-per-cpu=16G -t02:00:00 -oReport-%j.out "${RUN_COMMAND[@]}"
  exit 0
else 
  ${RUN_COMMAND[@]}
  exit 0
fi

exit 1