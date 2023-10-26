import evaluate
import numpy as np

LABEL2ID = {
    "first_level": {
        "policy": 0,
        "knowledgeBase": 1,
        "chat": 2,
        "policyTag": 3,
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

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)