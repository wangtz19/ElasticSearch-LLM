from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import DataCollatorWithPadding
from typing import Dict

from logger import get_logger


logger = get_logger(__name__)

def get_dataset(file_path: str) -> DatasetDict:

    logger.info(f"Loading dataset from {file_path}...")
    dataset = load_dataset("json", data_files=file_path)

    return dataset


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
) -> Dataset:
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], 
                         truncation=True,
                         max_length=512, # 512 is the maximum length of BERT
                         padding="max_length")
    def convert_label(example):
        example["label"] = label2id[example["label"]]
        return example
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    dataset = dataset.map(convert_label) # convert text label into int label
    return dataset


def split_dataset(
    dataset: Dataset, 
    dev_ratio: float, 
    do_train: bool = True
) -> Dict[str, Dataset]:
    # Split the dataset
    if do_train:
        if dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=dev_ratio)
            return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            return {"train_dataset": dataset}
    else: # do_eval or do_predict
        return {"eval_dataset": dataset}


def get_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)