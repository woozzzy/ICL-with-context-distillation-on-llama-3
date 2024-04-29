# ICL-with-context-distillation-on-llama-3

CS 4644 Final Project

## Dependencies:

```bash
conda create -n llama python=3.11 \
conda activate llama \
pip install torch torchvision torchaudio transformers datasets accelerate huggingface_hub python-dotenv pandas rouge-score sentencepiece spacy summa \
python -m spacy download en_core_web_sm
```

## HuggingFace Authentication:

-   Create `.env` file in root directory using `.env.example` as a template.
-   Copy your HuggingFace Access Token to `.env`.
    -   See [HuggingFace Docs](https://huggingface.co/docs/hub/en/security-tokens) for information on how to get token.

## To-Do:

-   [ ] Refactor Rouge metric calculation to use `evaluate`: [Rouge Metric](https://huggingface.co/spaces/evaluate-metric/rouge)
-   [ ] Implement `accelerate`: [Accelerate](https://huggingface.co/docs/transformers/en/accelerate)
