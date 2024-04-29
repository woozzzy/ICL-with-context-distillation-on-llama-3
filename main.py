import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from rouge_score import rouge_scorer

from src.io import hf_login, load_config, get_args
from src.logs import logger
from src.data import get_dataset, distill
from src.rouge import evaluate_summaries

if __name__ == "__main__":
    hf_login()
    args = get_args()
    cfg = load_config(args["config_path"])

    logger.info(f"Running script in mode: {args['mode']}")

    dataset = get_dataset(cfg)
    if cfg["dataset"]["distill"]:
        dataset = distill(dataset)

    tokenizer = LlamaTokenizer.from_pretrained(cfg["model"]["remote"])
    model = LlamaForCausalLM.from_pretrained(cfg["model"]["remote"])

    if torch.cuda.is_available():
        logger.debug("CUDA is available")
        model.to("cuda")
    else:
        logger.error("CUDA is not available")
        raise Exception("CUDA is not available")

    logger.info(f"Loaded model: {cfg['model']['remote']}")

    if args["mode"] == "train":
        tokenized_dataset = dataset.map(lambda x: tokenizer(x["article"]), batched=True)

        training_args = TrainingArguments(
            output_dir="./llama_finetuned",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
        )

        trainer.train()

    elif args["mode"] == "test":
        score = evaluate_summaries(model, tokenizer, dataset["test"])
        logger.info(f"Average ROUGE-1 F1 score: {score}")
