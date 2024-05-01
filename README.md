# ICL-with-context-distillation-on-llama-3

CS 4644 Final Project

## To-Do:

-   [ ] Select Task for In-Context Learning
-   [ ] Implement Evaluation Metric for Task
-   [ ] Perform In-Context Learning based on certain Prompts
-   [ ] Implement Context Distillation for Task
-   [ ] Fine Tune via Context Distillation
-   [ ] Evaluate Context Distillation Results

## Dependencies:

```bash
conda create -n llama python=3.11 \
conda activate llama \
pip install -U torch torchvision torchaudio \
pip install -U transformers datasets accelerate evaluate bitsandbytes huggingface_hub trl peft \
pip install python-dotenv pandas \
conda install -c conda-forge cudatoolkit-dev -y \
pip install flash-attn --no-build-isolation
```

## HuggingFace Authentication:

-   Create `.env` file in root directory using `.env.example` as a template.
-   Copy your HuggingFace Access Token to `.env`.
    -   See [HuggingFace Docs](https://huggingface.co/docs/hub/en/security-tokens) for information on how to get token.

## Example Usage:

> _If torchrun is throwing sigterms, it is likely an out of memory issue. Try increasing CPU memory._

#### Using `run.sh` script

```bash
Usage: run.sh [OPTIONS]
Options:
-c, --config <config> Path to the config file. Default: config/llama-3-8b-qlora.yaml.
-f, --fsdp Use FSDP for training.
-t, --torchrun Use torchrun for training.
--nproc_per_node <num> Number of GPUs to use with torchrun. Default: 4.
-c, --clean Clean the output directory. Does not run the training.
-s, --slurm Dispatch Slurm job. For use on PACE cluster only.
-h, --help Show this help message.
```
