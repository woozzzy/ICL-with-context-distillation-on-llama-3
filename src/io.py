import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from json import load
from os import getenv
from pprint import pformat
from src.logs import logger

load_dotenv()
HF_ACCESS_TOKEN = getenv("HF_ACCESS_TOKEN")


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tunes a modle.")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="Path to the json config file.",
        default="config.json",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Mode to run the script in [train, test, valid].",
        default="train",
    )

    args = parser.parse_args().__dict__
    logger.info(f"Args: \n{pformat(args, indent=4, width=1, sort_dicts=False,)}\n")
    return args


def hf_login():
    try:
        login(token=HF_ACCESS_TOKEN)
        logger.info("Logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Error logging in to Hugging Face: {e}")
        raise e


def load_config(file_path: str = "config.json"):
    try:
        with open(file_path) as f:
            config = load(f)
            logger.info(f"Loaded config: \n{pformat(config, indent=4, width=1, sort_dicts=False,)}\n")
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise e
