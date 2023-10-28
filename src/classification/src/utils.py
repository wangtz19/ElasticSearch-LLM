import evaluate
import numpy as np

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