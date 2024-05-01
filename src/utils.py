import random
import os
import shutil
from dotenv import load_dotenv
from huggingface_hub import login
from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG
from time import strftime

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def hf_login():
    try:
        load_dotenv()
        login(token=os.getenv("HF_ACCESS_TOKEN"))
    except Exception as e:
        raise e

def get_output_path(args):
    ## Create output directory(s)
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

    output_path = os.path.join(model_dataset_dir, f"run_{run_no}")
    os.makedirs(output_path)

    ## Logger Setup
    file_handler = FileHandler(os.path.join(output_path, f"{strftime('%Y-%m-%d_%H:%M:%S')}.log"))
    console_handler = StreamHandler()
    formatter = Formatter("%(asctime)s %(levelname)s: %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger Initialized")
    logger.info(f"Output Path: {output_path}")

    return output_path

def log_samples(train_args, dataset):
    with train_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(dataset)), 2):
            logger.debug("Processed Sample:\n" + dataset[index]["text"] + "\n")

def move_slurm_files(train_args):
    for file in os.scandir("output"):
        if file.name.endswith(".out"):
            shutil.move(file.path, os.path.join(train_args.output_dir, file.name))
        elif file.name == "slurm_job.sh":
            os.remove(file.path)