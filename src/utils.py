import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from huggingface_hub import login
from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG
from time import strftime

logger = getLogger(__name__)
logger.setLevel(DEBUG)


@dataclass
class ScriptArgs:
    mode: str = field(default="train", metadata={"help": "Mode to run the script in"})
    model_id: str = field(default="meta-llama/Meta-Llama-3-8b", metadata={"help": "HF Model ID"})
    model_path: str = field(default="./models/", metadata={"help": "Path to Local the model"})
    dataset_id: str = field(default="HuggingFaceH4/no_robots", metadata={"help": "HF Dataset ID"})
    preprocessed: bool = field(default=False, metadata={"help": "Use preprocessed data"})
    num_workers: int = field(default=4, metadata={"help": "Number of workers for DataLoader"})
    max_seq_len: int = field(default=1024, metadata={"help": "Max sequence length"})
    use_local_model: bool = field(default=False, metadata={"help": "Use local model"})
    upload_model: bool = field(default=False, metadata={"help": "Upload model to HF"})
    distill: bool = field(default=False, metadata={"help": "Distill model"})
    use_instruct_template: bool = field(default=False, metadata={"help": "Use instruct template"})


def get_output_path(args):
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./output"):
        os.makedirs("./output")

    model_name = args.model_id.split("/")[-1].lower()
    dataset_name = args.dataset_id.split("/")[-1].lower()
    model_dataset_dir = os.path.join("./output", f"{model_name}_{dataset_name}")

    if not os.path.exists(model_dataset_dir):
        os.makedirs(model_dataset_dir)

    run_no = 1
    while os.path.exists(os.path.join(model_dataset_dir, f"run_{run_no}")):
        run_no += 1

    run_dir = os.path.join(model_dataset_dir, f"run_{run_no}")
    os.makedirs(run_dir)

    return run_dir


def init_logger(output_path="./"):
    file_handler = FileHandler(os.path.join(output_path, f"{strftime('%Y-%m-%d_%H:%M:%S')}.log"))
    console_handler = StreamHandler()
    formatter = Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger Initialized")
    logger.info(f"Output Path: {output_path}")


def hf_login():
    try:
        load_dotenv()
        login(token=os.getenv("HF_ACCESS_TOKEN"))
        logger.info("Logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Error logging in to Hugging Face: {e}")
        raise e


def get_chat_template(use_instruct_template):
    # Llama 3 instruct template -- make sure to add modules_to_save
    if use_instruct_template:
        return "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    # Anthropic/Vicuna like template without the need for special tokens
    else:
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
