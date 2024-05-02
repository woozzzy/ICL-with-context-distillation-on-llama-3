import os
import json
import random
from datasets import load_dataset
from .utils import (
    logger,
)


def get_dataset(args, train_args, split="train", use_icl=True):
    logger.info("Loading data ...")
    data_path = args.test_path if split == "test" else args.train_path

    ############################    Load Dataset from Local/Remote    ############################
    if args.use_local_dataset:
        dataset = load_dataset("json", data_files=data_path, num_proc=args.num_workers, split=split)
    else:
        dataset = load_dataset(args.dataset_id, num_proc=args.num_workers, split=split)
        dataset = dataset.shuffle()
        dataset = dataset.select(range(1000))

        ############################    Base System Prompt    ############################
        
        base = """You are LLAMA, a powerful AI assistant created by deep learning researchers to provide succinct and meaningful summaries on a variety of texts. Your expansive knowledge base allows you to engage in critical analysis of these texts to identify the most significant portions. In this task, you will help summarize news articles into a single concise headline. Here are a few examples on how to transform a detailed document into a brief summary:\n\n"""
        context = ""
        instruction = "Write a concise summary/headline for the provided document."
        
        ############################    In-Context Learning    ############################
        
        logger.info(f"Applying In-Context Learning for Task: {args.icl}")

        ## Filter dataset to match task
        if args.icl != 'summarize':
            raise ValueError("Gigaword only support summarization tasks")
        
        ## Construct and Add Context
        if use_icl:
            ## Sample Examples from Dataset
            example_idx = random.sample(range(len(dataset)), 4)
            examples = dataset.select(example_idx)
            dataset = dataset.select(([i for i in range(len(dataset)) if i not in set(example_idx)]))
                    
            ## Add Context to Prompt
            for ex in examples:
                context += (f"ExampleHuman: {instruction}\n")
                context += (f"\tDocument - {ex['document']}\n")
                context += (f"ExampleAssistant: Summary - {ex['summary']}\n\n")

        ############################    Convert to Chat Format    ############################

        sys_prompt = base + context

        def _format(sample):
            task = {'role': 'user', 'content': f"{instruction}\n\tDocument - {sample['document']}"}
            reference = {"role": "assistant", "content": f"Summary - {sample['summary']}"}
            sample['document'] = [{'role': 'system', 'content': sys_prompt}, task, reference]
            return sample
        
        columns_to_remove = list(dataset.features)
        columns_to_remove.remove('document')
        dataset = dataset.map(_format, remove_columns=columns_to_remove, batched=False)
        dataset = dataset.rename_column('document', 'messages')

        ############################    Filter Conversations with Odd # of Turns    ############################
        
        dataset = dataset.filter(lambda x: len(x["messages"][1:]) % 2 == 0)

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
            "{{ '\n\nAssistant: Summary - ' }}"
            "{% endif %}"
        )