from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def evaluate_summaries(model, tokenizer, dataset):
    model.eval()
    summaries = []
    references = []
    for batch in dataset:
        input_ids = tokenizer(batch["article"], return_tensors="pt", truncation=True, padding=True, max_length=512)
        summary_ids = model.generate(input_ids["input_ids"])
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        references.append(batch["highlights"])

    scores = [scorer.score(ref, summ) for ref, summ in zip(references, summaries)]
    rouge1_scores = [score["rouge1"].fmeasure for score in scores]
    return sum(rouge1_scores) / len(rouge1_scores)
