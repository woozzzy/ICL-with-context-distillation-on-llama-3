import os
import random
from datasets import load_dataset
from src.utils import (
    logger,
)


def get_dataset(args, train_args, split="train", use_icl=True):
    logger.info("Loading data ...")
    data_path = args.test_path if split == "test" else args.train_path

    if args.use_local_dataset:
        dataset = load_dataset("json", data_files=data_path, num_proc=args.num_workers, split=split)
    else:
        dataset = load_dataset(args.dataset_id, num_proc=args.num_workers, split=split)
        dataset = dataset.shuffle()
        dataset = dataset.select(range(100000))

        ############################    Base System Prompt    ############################
        
        sys_prompt = """In this task, you will help summarize news articles into a single concise sentence. Here are a few examples on how to transform a detailed document into a brief summary:\n"""
        sys_prompt_msg = {'role': 'system', 'content': sys_prompt}

        ############################    In-Context Learning    ############################
        
        logger.info(f"Applying In-Context Learning for Task: {args.icl}")

        ## Filter dataset to match task
        if args.icl != 'summarize':
            raise ValueError("Gigaword only support summarization tasks")
        
        ## Construct and Add Context
        if use_icl:
            ## Sample Examples from Dataset
            example_idx = random.sample(range(len(dataset)), 5)
            examples = dataset.select(example_idx)
            dataset = dataset.select(([i for i in range(len(dataset)) if i not in set(example_idx)]))
            
            ## TODO: Implement context distillation
        
            ## Add Context to Prompt
            instruct_prompt = "" 
            for i, example in enumerate(examples):
                instruct_prompt += f"Document: {example['document']}\nSummary: {example['summary']}\n\n"

            logger.debug(f"Length of prompt with context: {len(prompt)}")

        ############################    Add Prompt to Data    ############################

        prompt += "Now, based on the following document, provide a concise summary."

        def _add_prompt(sample):
            sample["document"] = prompt + sample["document"] + "\n" 
            return sample

        dataset = dataset.map(_add_prompt, batched=False)
        
        ############################    Save to Disk    ############################

        if os.path.exists(data_path):
            os.remove(data_path)
        dataset.to_json(data_path, orient="records", force_ascii=False)

    return dataset

def get_template(args):

    if args.is_instruct:
        return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    else:
        ## Anthropic/Vicuna like template without the need for special tokens
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '\n\nAssistant: ' }}"
            "{% endif %}"
        )