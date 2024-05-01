import os
import random
import torch
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

    dataset = get_dataset(args, split="train")
    sample_idx = random.sample(range(len(dataset)), 3)
    samples = [dataset[i]["messages"][:2] for i in sample_idx]

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

    input_ids = tokenizer.apply_chat_template(
        samples, 
        max_length=args.max_seq_len, 
        padding=True, 
        truncation=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    ############################    Inference    ############################
    
    logger.info("Beginning generation.")

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    responses = tokenizer.batch_decode([x[input_ids.shape[-1]:] for x in outputs], skip_special_tokens=(not args.is_instruct))
    
    logger.info("Finished generation.")

    ############################    Export for Visual Comparison    ############################

    with open(os.path.join(train_args.output_dir, "test_results.txt"), "w") as f:
        for i, idx in enumerate(sample_idx):
            prompt = dataset[idx]['messages'][0]['content']
            question = dataset[idx]['messages'][1]['content']
            answer = dataset[idx]['messages'][2]['content']
            response = responses[i]
            sep = "--------------------------------------------------"

            f.write(f"{sep+sep}\nPrompt:\n{prompt}\n{sep}\nQuestion:\n{question}\n{sep}\nAnswer:\n{answer}\n{sep}\nResponse:\n{response}\n")

def train(args, train_args):
    ############################    Tokenizer    ############################

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, chat_template=get_template(args))
    tokenizer.pad_token = tokenizer.eos_token

    ############################    Dataset    ############################

    train_dataset = get_dataset(args)
    test_dataset = get_dataset(args, split="test")
    
    train_dataset = train_dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)}, remove_columns=["messages"])
    test_dataset = test_dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False)}, remove_columns=["messages"])
    log_samples(train_args, train_dataset)

    ############################    Quantization    ############################

    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    ############################    Model    ############################

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",  # "sdpa"
        torch_dtype=quant_storage_dtype,
        device_map="auto",
        use_cache=False if train_args.gradient_checkpointing else True,
    )

    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ############################    PEFT    ############################

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] if args.is_instruct else None,
    )

    ############################    SFTTrainer    ############################

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ############################    Train    ############################

    logger.info("Starting training ...")

    os.makedirs(os.path.join(train_args.output_dir, "checkpoints"))
    checkpoint = None
    if train_args.resume_from_checkpoint is not None:
        checkpoint = train_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ############################    Save Model    ############################

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

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