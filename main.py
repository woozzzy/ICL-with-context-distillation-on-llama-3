import os
import random
import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from pprint import pformat
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from trl.commands.cli_utils import TrlParser
from trl import SFTTrainer

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
    dataset = load_dataset("HuggingFaceH4/no_robots", num_proc=args.num_workers)

    ############################    Add System Prompt    ############################

    prompt = """You are Llama, an AI assistant created by Sunwoo to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects."""

    def _add_prompt(sample):
        if sample["messages"][0]["role"] == "system":
            return sample
        else:
            sample["messages"] = [{"role": "system", "content": prompt}] + sample["messages"]
            return sample

    columns_to_remove = list(dataset["train"].features)
    columns_to_remove.remove("messages")
    dataset = dataset.map(_add_prompt, remove_columns=columns_to_remove, batched=False)

    ############################    Filter Conversations with Wrong # of Turns    ############################

    dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
    dataset["test"] = dataset["test"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)

    ############################    Distill Context    ############################

    ## TODO: Implement context distillation

    ############################    Save to Disk    ############################

    dataset["train"].to_json("data/train_dataset.json", orient="records", force_ascii=False)
    dataset["test"].to_json("data/test_dataset.json", orient="records", force_ascii=False)


def test(args, training_args):
    logger.info("Testing ...")

    ############################    Dataset    ############################

    test_dataset = load_dataset(
        "json",
        data_files="data/test_dataset.json",
        split="train",
        num_proc=args.num_workers,
    )
    rand_idx = random.randint(0, len(test_dataset) - 1)
    test_samples = test_dataset[rand_idx]["messages"][:2]

    ############################    Model/Tokenizer    ############################

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # torch.float16
        quantization_config={"load_in_4bit": True},
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    ############################    Generate    ############################

    input_ids = tokenizer.apply_chat_template(test_samples, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]

    ############################    Log for Visual Comparison    ############################

    logger.info(f"**Query:**\n{test_dataset[rand_idx]['messages'][1]['content']}\n")
    logger.info(f"**Original Answer:**\n{test_dataset[rand_idx]['messages'][2]['content']}\n")
    logger.info(f"**Generated Answer:**\n{tokenizer.decode(response,skip_special_tokens=True)}")


def train(args, training_args):
    logger.info("Training ...")
    os.makedirs(os.path.join(output_path, "checkpoints"))

    ############################    Dataset    ############################

    if not args.preprocessed:
        prep_data(args, training_args)

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

    ############################    Tokenizer    ############################

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = get_chat_template(args.use_instruct_template)

    def _apply_template(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

    train_dataset = train_dataset.map(_apply_template, remove_columns=["messages"])
    test_dataset = test_dataset.map(_apply_template, remove_columns=["messages"])

    # Log random samples from the processed training set
    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(train_dataset)), 2):
            logger.debug(train_dataset[index]["text"] + "\n")

    ############################    Model    ############################

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

    ############################    PEFT    ############################

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] if args.use_instruct_template else None,
    )

    ############################    SFTTrainer    ############################

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

    ############################    Train    ############################

    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # trainer.train(resume_from_checkpoint=checkpoint)

    ############################    Save Model    ############################

    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((ScriptArgs, TrainingArguments))
    args, training_args = parser.parse_args_and_config()

    output_path = get_output_path(args)
    init_logger(output_path)
    hf_login()

    set_seed(training_args.seed)

    training_args.output_dir = output_path
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logger.debug(f"Arguments: \n{pformat(args.__dict__)}\n")
    logger.debug(f"Training Arguments: \n{pformat(training_args.__dict__)}\n")

    # Run specified mode
    if args.mode == "prep_data":
        prep_data(args, training_args)
    elif args.mode == "test":
        test(args, training_args)
    elif args.mode == "train":
        train(args, training_args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        raise ValueError(f"Unknown mode: {args.mode}")
