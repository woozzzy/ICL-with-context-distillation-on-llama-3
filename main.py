import random
import torch
import evaluate
from accelerate import Accelerator
from peft import LoraConfig, AutoPeftModelForCausalLM
from pprint import pformat, pprint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from trl.commands.cli_utils import TrlParser
from trl import SFTTrainer

from src.args import ScriptArgs
from src.utils import *
from src.data import *

def test(args, train_args):
    ############################    Dataset    ############################

    dataset = get_dataset(args, train_args, split="train")
    sample_idx = random.sample(range(len(dataset)), 10)
    samples = dataset[sample_idx]["document"]

    ############################    Model    ############################

    model_id = args.model_path if args.use_local_model else args.model_id
    if args.is_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config={"load_in_4bit": True},
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",  # "sdpa"
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False if train_args.gradient_checkpointing else True,
        )

    ############################    Tokenizer    ############################

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side='left', chat_template=get_template(args))
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(
        samples, 
        max_length=args.max_seq_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(model.device)


    ############################    Inference    ############################
    
    logger.info("Beginning generation.")

    outputs = model.generate(
        input_ids,
        max_new_tokens=64,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    logger.info("Finished generation.")

    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=(not args.is_instruct))
    
    prompts, references = [], []
    for sample in dataset.select(sample_idx):
        prompts.append(sample['document'])
        references.append(sample['summary'])

    ############################    Evaluation    ############################

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)

    logger.info(f"Results: {results}")

    ############################    Export for Visual Comparison    ############################

    with open(os.path.join(train_args.output_dir, "test_results.txt"), "w") as f:
        f.write(f"Results: {results}\n")
        for x in zip(prompts, references, predictions):
            f.write("------------------------------------------\n\n")
            f.write(f"Prompt:\n{x[0]}\n")
            f.write(f"Reference:\n{x[1]}\n")
            f.write(f"Prediction:\n{x[2]}\n\n")

        

def train(args, train_args):
    raise NotImplementedError("Training is not yet implemented for gigaword.")

if __name__ == "__main__":
    hf_login()

    ############################    Parse Args and Config    ############################
    
    parser = TrlParser((ScriptArgs, TrainingArguments))
    args, train_args = parser.parse_args_and_config()

    train_args.output_dir = get_output_path(args)
    if train_args.seed == -1:
        train_args.seed = random.randint(0, 100000)
        logger.info(f"Seed set to {train_args.seed}")
    set_seed(train_args.seed)
    if train_args.gradient_checkpointing:
        train_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logger.debug(f"Arguments: \n{pformat(args.__dict__)}\n")
    logger.debug(f"Training Arguments: \n{pformat(train_args.__dict__)}\n")

    ############################    Login to HF and Run Mode    ############################

    logger.info(f"Running Mode: {args.mode}")

    if args.mode == "test":
        test(args, train_args)
    elif args.mode == "train":
        train(args, train_args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        raise ValueError(f"Unknown mode: {args.mode}")

    ############################    Cleanup    ############################

    logger.info("Performing Cleanup ...")
    move_slurm_files(train_args)
    logger.info("Done")