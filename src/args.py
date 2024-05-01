from dataclasses import dataclass, field

@dataclass
class ScriptArgs:
    mode: str = field(default="train", metadata={"help": "Mode to run the script in"})
    icl: str = field(default='none', metadata={"help": "Options: 'none', 'extract'"})
    distill: bool = field(default=False, metadata={"help": "Distill model"})
    max_seq_len: int = field(default=2048, metadata={"help": "Max sequence length"})
    ## Model Params
    model_id: str = field(default="meta-llama/Meta-Llama-3-8b", metadata={"help": "HF Model ID"})
    model_path: str = field(default="./models/", metadata={"help": "Path to Local the model"})
    use_local_model: bool = field(default=False, metadata={"help": "Use local model"})
    upload_model: bool = field(default=False, metadata={"help": "Upload model to HF"})
    is_peft: bool = field(default=False, metadata={"help": "Use PEFT"})
    is_instruct: bool = field(default=False, metadata={"help": "Using instruct model variant"})
    ## Dataset Params
    dataset_id: str = field(default="HuggingFaceH4/no_robots", metadata={"help": "HF Dataset ID"})
    train_path: str = field(default="data/train_data.json", metadata={"help": "Path to train data"})
    test_path: str = field(default="data/test_data.json", metadata={"help": "Path to test data"})
    use_local_dataset: bool = field(default=False, metadata={"help": "Use preprocessed data"})
    num_workers: int = field(default=0, metadata={"help": "Number of workers for DataLoader"})
