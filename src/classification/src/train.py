from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import compute_metrics
from constants import LABEL2ID, ID2LABEL
import datetime
from dataset import get_dataset, preprocess_dataset, split_dataset, get_collator
from logger import get_logger
from argparse import ArgumentParser
from callbacks import LogCallback

def get_config(clf_level: str):
    assert clf_level in ["first_level", "direct", "policy", "chat", "policyTag", "knowledgeBase"], \
        f"clf_level `{clf_level}` is not supported!"
    if clf_level == "first_level":
        data_path = "data/first_level.json"
        label2id = LABEL2ID[clf_level]
        id2label = ID2LABEL[clf_level]
    elif clf_level == "direct":
        data_path = "data/direct_classification.json"
        label2id = LABEL2ID[clf_level]
        id2label = ID2LABEL[clf_level]
    else:
        data_path = f"data/{clf_level}_second_level.json"
        label2id = LABEL2ID["second_level"][clf_level]
        id2label = ID2LABEL["second_level"][clf_level]
    return data_path, label2id, id2label
    

def train(args):
    logger = get_logger(__name__)

    data_path, label2id, id2label = get_config(args.clf_level)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    date_str = datetime.datetime.now().strftime("%y.%m.%d-%H:%M")
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.clf_level}-{date_str}",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        logging_dir=f"logs/{args.clf_level}-{date_str}",
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    logger.info("Loading dataset...")
    dataset = get_dataset(data_path)["train"]
    dataset = preprocess_dataset(dataset, tokenizer, label2id)
    dataset = split_dataset(dataset, args.dev_ratio)

    logger.info("Getting collator...")
    collator = get_collator(tokenizer)

    callbacks = [LogCallback()]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train_dataset"],
        eval_dataset=dataset["eval_dataset"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()


if __name__ == "__main__":
    
    parser = ArgumentParser()

    # data arguments
    parser.add_argument("--clf_level", type=str, default="first_level", help="first_level or direct or policy")

    # model arguments
    parser.add_argument("--model_name", type=str, default="/root/share/chinese-bert-wwm", help="pretrained classification model name")
    
    #training arguments
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="ratio of dev dataset")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=10.0, help="number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="batch size for evaluation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--logging_steps", type=int, default=100, help="logging steps")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="save strategy")
    
    args = parser.parse_args()
    
    train(args)