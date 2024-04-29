from src.io import hf_login, load_config, get_args
from src.logs import logger

if __name__ == "__main__":
    hf_login()
    args = get_args()
    cfg = load_config(args["config_path"])

    logger.info(f"Running script in mode: {args['mode']}")

    if args["mode"] == "train":
        pass
