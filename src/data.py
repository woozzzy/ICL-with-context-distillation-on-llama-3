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
        
        prompt = """As a professional summarizer, create a concise and comprehensive summary for ONLY the provided text denoted by \"Document:\". Be sure to include the main ideas and essential information, while eliminating extraneous language and focusing on critical aspects. Rely strictly on the provided text, without including external information. Provided below are example documents and summmarizations. Format your summary to be at most 2 sentences for easy comprehension.\n"""

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
            for i, example in enumerate(examples):
                prompt += f"Example Document {i}: {example['document']}\nExample Summary {i}: {example['summary']}\n\n"

            logger.debug(f"Length of prompt with context: {len(prompt)}")

        ############################    Add Prompt to Data    ############################

        prompt += "**Document:**: "

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
            "{{ sample }}"
            "{{ sample }}"
            "{% if add_generation_prompt %}"
            "{{ '\n\nSummary: ' }}"
            "{% endif %}"
        )

