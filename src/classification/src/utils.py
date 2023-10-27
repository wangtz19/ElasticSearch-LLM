import evaluate
import numpy as np

LABEL2ID = {
    "first_level": {
        "policy": 0,
        "knowledgeBase": 1,
        "chat": 2,
        "policyTag": 3,
        "others": 4,
    },
    "second_level": {
        "policy": {
            "basicInfo": 0,
            "award": 1,
            "process": 2,
            "materials": 3,
            "condition": 4
        }
    },
    "direct": { # only consider policy and knowledgeBase
        "basicInfo": 0,
        "award": 1,
        "process": 2,
        "materials": 3,
        "condition": 4,
        "knowledgeBase": 5,
        "others": 6,
    }
}

def get_id2label(label2id):
    tmp = {}
    for k, v in label2id.items():
        if not isinstance(v, dict):
            tmp[v] = k
        else:
            tmp[k] = get_id2label(v)
    return tmp

ID2LABEL = get_id2label(LABEL2ID)

# clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
average = "weighted"
accuracy_metric = evaluate.load("accuracy", average=average)
precision_metric = evaluate.load("precision", average=average)
recall_metric = evaluate.load("recall", average=average)
f1_metric = evaluate.load("f1", average=average)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    results = {}
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))
    results.update(precision_metric.compute(predictions=predictions, references=labels, average=average))
    results.update(recall_metric.compute(predictions=predictions, references=labels, average=average))
    results.update(f1_metric.compute(predictions=predictions, references=labels, average=average))
    return results