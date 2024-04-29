import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

from src.io import hf_login, load_config, get_args
from src.logs import logger
from src.data import get_dataset, distill


if __name__ == "__main__":
    hf_login()
    args = get_args()
    cfg = load_config(args["config_path"])

    logger.info(f"Running script in mode: {args['mode']}")

    if args["mode"] == "train":
        dataset = get_dataset(cfg)[args["mode"]]
        if cfg["dataset"]["distill"]:
            dataset = distill(dataset)

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["remote"])
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["remote"])

        if torch.cuda.is_available():
            logger.debug("CUDA is available")
            model.to("cuda")
        else:
            logger.error("CUDA is not available")
            raise Exception("CUDA is not available")

        logger.info(f"Loaded model: {cfg['model']['remote']}")

    elif args["mode"] == "test":
        dataset = get_dataset(cfg)[args["mode"]]
        if cfg["dataset"]["distill"]:
            dataset = distill(dataset)

        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["remote"])
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]["remote"])

        if torch.cuda.is_available():
            logger.debug("CUDA is available")
            model.to("cuda")
        else:
            logger.error("CUDA is not available")
            raise Exception("CUDA is not available")

        logger.info(f"Loaded model: {cfg['model']['remote']}")

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = []

        for example in dataset:
            # Get prompt
            prompt = example["article"]

            # Generate summary
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
            outputs = model.generate(
                **inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Calculate ROUGE score
            score = scorer.score(example["highlights"], summary)["rouge1"].fmeasure
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        logger.info(f"Average ROUGE-1 F1 score: {avg_score}")
