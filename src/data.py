import random
from datasets import load_dataset
from src.utils import (
    logger,
)


def get_dataset(args, split="train", use_icl=True):
    logger.info("Loading data ...")
    data_path = args.test_path if split == "test" else args.train_path

    if args.use_local_dataset:
        dataset = load_dataset("json", data_files=data_path, num_proc=args.num_workers, split=split)
    else:
        dataset = load_dataset(args.dataset_id, num_proc=args.num_workers, split=split)

        ############################    Base System Prompt    ############################
        
        prompt = """You are Llama, an AI assistant created by deep learning researchers to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations and provide analysis on complex subjects.\n"""

        ############################    In-Context Learning    ############################
        
        logger.info(f"Applying In-Context Learning for Task: {args.icl}")

        ## Filter dataset to match task
        if args.icl == 'extract':
            dataset = dataset.filter(lambda x: x["category"] == "Extract")
        elif args.icl == 'summarize':
            dataset = dataset.filter(lambda x: x["category"] == "Summarize")
        elif args.icl == 'qa':
            dataset = dataset.filter(lambda x: x["category"] == "Open QA")

        ## Construct and Add Context
        if use_icl:
            ## Sample Examples from Dataset
            example_idx = random.sample(range(len(dataset)), 2)
            examples = [dataset[i]["messages"][:2] for i in example_idx]
            dataset = dataset.select(([i for i in range(len(dataset)) if i not in set(example_idx)]))
            
            ## TODO: Implement context distillation
        
            ## Add Context to Prompt
            for example in examples:
                for message in example:
                    if message["role"] == "system":
                        prompt += f"{message['content']}\n"
                    elif message["role"] == "user":
                        prompt += f"Human: {message['content']}\n"
                    elif message["role"] == "assistant":
                        prompt += f"Assistant: {message['content']}"

            logger.debug(f"Length of prompt with context: {len(prompt)}")
                    
        ############################    Add Prompt to Data    ############################

        def _add_prompt(sample):
            if sample["messages"][0]["role"] == "system":
                return sample
            else:
                sample["messages"] = [{"role": "system", "content": prompt}] + sample["messages"]
                return sample

        columns_to_remove = list(dataset.features)
        columns_to_remove.remove("messages")
        dataset = dataset.map(_add_prompt, remove_columns=columns_to_remove, batched=False)

        ############################    Filter Conversations with Wrong # of Turns    ############################

        dataset = dataset.filter(lambda x: len(x["messages"][1:]) % 2 == 0)
        
        ############################    Save to Disk    ############################

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

