# ICL-with-context-distillation-on-llama-3

CS 4644 Final Project

## Environment Setup:

### Conda:

```bash
$ conda create -n icl python=3.11
$ conda activate icl
```

### Pytorch: [See Docs](https://pytorch.org/get-started/locally/)  
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### HuggingFace: 
```bash
$ pip install -U transformers datasets accelerate evaluate bitsandbytes huggingface_hub trl peft nltk rouge-score absl-py
```

### Flash Attention:
```bash
$ conda install -c conda-forge cudatoolkit-dev -y 
$ pip install flash-attn --no-build-isolation 
```

### spaCy:
```bash
$ pip install spacy
$ python -m spacy download en_core_web_md
```

### Misc:
```bash
$ pip install python-dotenv pandas matplotlib scikit-learn
```

### HuggingFace Authentication:

-   Create `.env` file in root directory using `.env.example` as a template.
-   Copy your HuggingFace Access Token to `.env`.
    -   See [HuggingFace Docs](https://huggingface.co/docs/hub/en/security-tokens) for information on how to get token.

## Example Usage:

> _If torchrun is throwing sigterms, it is likely an out of memory issue. Try increasing CPU memory._

### Using `run.sh` script

```
Usage: run.sh [OPTIONS]
Options:
  -h, --help                  Show this help message.
  -c, --config <config>       Path to the config file. Default: config/llama-3-8b-qlora.yaml.
  -f, --fsdp                  Use FSDP for training.
  -t, --torchrun              Use torchrun for training. Specify the number of GPUs with --nproc_per_node.
  -n. --nproc_per_node <num>  Number of GPUs to use with torchrun. Default: 4.
  -s, --slurm                 Dispatch Slurm job. For use on PACE cluster only.
  --gen-only                  Only generate slurm_job.sh file for manual dispatching.
  --clean                     Clean the output directory. Does not run the training.
```

## Config File:

### Script Parameters:

-   `mode`: str - Determines the mode of the script. Options: `'train'`, `'test'`
-   `icl`: str - Specify which task to use ICL for. Only `'summarize'` implemented
-   `distill_process`: bool - Whether to use context distillation via processing the text/
-   `distill_sample`: bool - Whether to use context distillation via selecting better examples. 
-   `max_seq_len`: int - Maximum sequence length for model. Ex: `2048`
-   `model_id`: str - Model ID for HuggingFace model. Ex: `'meta-llama/Meta-Llama-3-8b'`
-   `model_path`: str - Path to local model. Ex: `'output/meta-llama-3-8b-qlora_no_robots/run_1/checkpoints'`
-   `use_local_model`: bool - Whether to use local model.
-   `is_peft`: bool - Whether to use PEFT.
-   `upload_model`: bool - Whether to upload model to HuggingFace.
-   `dataset_id`: str - Dataset ID for HuggingFace dataset. Ex: 'HuggingFaceH4/no_robots'
-   `train_path`: str - Path to local preprocessed training dataset. Ex: `'data/train_data.json'`
-   `test_path`: str - Path to local preprocessed test dataset. Ex: `'data/test_data.json'`
-   `use_local_dataset`: bool - Whether to use local preprocessed data.

### Training Parameters:

-   `output_dir`: str - Output directory for training. Will be set automatically.
-   `learning_rate`: float - Learning rate for training. Ex: `1e-5`
-   `lr_scheduler_type`: str - Learning rate scheduler type. Options: `'linear'`, `'cosine'`, `'cosine_with_restarts'`, `'polynomial'`, `'constant'`
-   `num_train_epochs`: int - Number of training epochs. Ex: `3`
-   `per_device_train_batch_size`: int - Batch size per device for training. Ex: `4`
-   `per_device_eval_batch_size`: int - Batch size per device for evaluation. Ex: `4`
-   `gradient_checkpointing`: bool - Whether to use gradient checkpointing.
-   `gradient_accumulation_steps`: int - Number of gradient accumulation steps. Ex: `1`
-   `optim`: str - Optimizer for training. Use `'adamw_torch'`
-   `weight_decay`: float - Weight decay for training. Ex: `0.01`
-   `max_grad_norm`: float - Maximum gradient norm for training. Ex: `1.0`
-   `warmup_ratio`: float - Warmup ratio for training. Ex: `0.1`
-   `logging_steps`: int - Logging steps for training. Ex: `100`
-   `save_strategy`: str - Save strategy for training. Use `'epoch'`
-   `evaluation_strategy`: str - Evaluation strategy for training. Use `'epoch'`
-   `bf16`: bool - Whether to use bfloat16 precision.
-   `tf32`: bool - Whether to use tf32 precision.
-   `seed`: int - Random seed for training. Ex: `42`
-   `disable_tqdm`: bool - Whether to disable tqdm.
-   `load_best_model_at_end`: bool - Whether to load best model at end of training.
-   `fsdp`: str - Default: `'full_shard_auto_warp offload'`; Remove `offload` if enough GPU memory
-   `fsdp_config`:
    -   `backward_prefetch`: str - Default: `'backward_pre'`
    -   `forward_prefetch`: str - Default: `'false'`
    -   `use_orig_params`: str - Default: `'false'`
