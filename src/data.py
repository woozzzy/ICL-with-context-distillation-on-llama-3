import spacy
from summa import summarizer

from datasets import load_dataset
from src.logs import logger

nlp = spacy.load("en_core_web_sm")


def get_dataset(cfg):
    dataset = load_dataset(cfg["dataset"]["remote"], cfg["dataset"]["ver"])
    logger.info(f"Loaded dataset: {cfg['dataset']['remote']}")
    return dataset


def distill(dataset):
    dataset = dataset.map(distill_dataset, num_proc=4, batched=True)
    logger.info("Distilled the dataset")
    return dataset


def distill_article(article):
    doc = nlp(article)
    key_entities = set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]])

    # Extractive summarization to condense the text
    distilled_context = summarizer.summarize(article, words=250)  # Extract the most relevant sentences

    # Filter distilled context to retain sentences containing key entities
    filtered_context = [
        sentence for sentence in distilled_context.split(".") if any(entity in sentence for entity in key_entities)
    ]

    return ". ".join(filtered_context)


def distill_dataset(dataset):
    try:
        dataset["article"] = [distill_article(article) for article in dataset["article"]]
        return dataset
    except Exception as e:
        logger.error(f"Error in distill_dataset: {e}")
        raise e
