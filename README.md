# ICL-with-context-distillation-on-llama-3

CS 4644 Final Project

## Dependencies:

```bash
conda create -n llama python=3.11 \
conda activate llama \
pip install -U torch torchvision torchaudio \
pip install -U transformers datasets accelerate evaluate bitsandbytes huggingface_hub trl peft \
pip install -U python-dotenv pandas tensorboard
```

## HuggingFace Authentication:

-   Create `.env` file in root directory using `.env.example` as a template.
-   Copy your HuggingFace Access Token to `.env`.
    -   See [HuggingFace Docs](https://huggingface.co/docs/hub/en/security-tokens) for information on how to get token.

## Example Usage with FSDP and Q-LoRA

```bash
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml
```
