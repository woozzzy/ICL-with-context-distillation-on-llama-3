import os
import random
import torch
from datasets import load_dataset
from peft import LoraConfig
from pprint import pprint, pformat
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from trl.commands.cli_utils import TrlParser
from trl import setup_chat_format, SFTTrainer

from src.utils import (
    logger,
    init_logger,
    hf_login,
    get_output_path,
    get_chat_template,
    ScriptArgs,
)


def prep_data(args, training_args):
    logger.info("Preparing data ...")

    prompt = """You are Llama, an AI assistant created by Sunwoo to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

    def _add_prompt(sample):
        if sample["messages"][0]["role"] == "system":
            return sample
        else:
            sample["messages"] = [{"role": "system", "content": prompt}] + sample["messages"]
            return sample

    # Load dataset from the hub
    dataset = load_dataset("HuggingFaceH4/no_robots", num_proc=args.num_workers)

    # Add system prompt to each conversation
    columns_to_remove = list(dataset["train"].features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(_add_prompt, remove_columns=columns_to_remove, batched=False)

    # Filter out conversations which are corrupted with wrong turns, keep which have even number of turns after adding system message
    dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)

    # Save datasets to disk
    dataset["train"].to_json("data/train_dataset.json", orient="records", force_ascii=False)
    dataset["test"].to_json("data/test_dataset.json", orient="records", force_ascii=False)


def test(args, training_args):
    logger.info("Testing ...")


def train(args, training_args):
    logger.info("Training ...")
    ############################
    # Dataset
    ############################
    # Preprocess dataset
    if not args.preprocessed:
        prep_data(args, training_args)

    # Load dataset
    train_dataset = load_dataset(
        "json",
        data_files="data/train_dataset.json",
        split="train",
        num_proc=args.num_workers,
    )
    test_dataset = load_dataset(
        "json",
        data_files="data/test_dataset.json",
        split="train",
        num_proc=args.num_workers,
    )

    ############################
    # Tokenizer
    ############################
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = get_chat_template(args.use_instruct_template)

    # Apply chat template
    def _apply_template(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

    train_dataset = train_dataset.map(_apply_template, remove_columns=["messages"])
    test_dataset = test_dataset.map(_apply_template, remove_columns=["messages"])

    # Log 2 random samples from the processed training set
    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(train_dataset)), 2):
            logger.info(train_dataset[index]["text"])

    ############################
    # Model
    ############################
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",  # "sdpa"
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ############################
    # PEFT
    ############################
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] if args.use_instruct_template else None,
    )

    ############################
    # SFTTrainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
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

    ############################
    # Train
    ############################
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # Save Model
    ##########################
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()


if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArgs, TrainingArguments))
    args, training_args = parser.parse_args_and_config()

    # Initialize logger and output directories
    output_path = get_output_path(args)
    init_logger(output_path)

    # Login to HF
    hf_login()

    # Set seed, output directory, and gradient checkpointing reentrant
    set_seed(training_args.seed)
    training_args.output_dir = output_path
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Logging ...
    logger.debug(f"Arguments: \n{pformat(args.__dict__)}\n")
    logger.debug(f"Training Arguments: \n{pformat(training_args.__dict__)}\n")

    # Run specified mode
    if args.mode == "prep_data":
        prep_data(args, training_args)
    elif args.mode == "test":
        test(args, training_args)
    elif args.mode == "train":
        os.makedirs(os.path.join(output_path, "checkpoints"))
        train(args, training_args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        raise ValueError(f"Unknown mode: {args.mode}")
