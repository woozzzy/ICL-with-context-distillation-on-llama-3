#!/bin/bash
POSITIONAL_ARGS=()
CONFIG="config/llama-3-8b-qlora.yaml"

while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      CONFIG="$2"
      shift
      shift 
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

# python main.py --config $CONFIG
# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 main.py --config $CONFIG