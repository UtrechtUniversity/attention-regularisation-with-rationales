import inspect

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from transformers import EvalPrediction


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5, class_names=None):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = predictions
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    cls_report = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True, target_names=class_names
    )

    if class_names is not None:
        class_conf_matrices = {}
        for i, class_name in enumerate(class_names):
            trues = [x[i] for x in y_true]
            preds = [x[i] for x in y_pred]
            conf_matrix = list(confusion_matrix(trues, preds).ravel())
            class_conf_matrices[class_name] = conf_matrix

        cls_report["class_conf_matrices"] = class_conf_matrices
    return cls_report


def binary_label_metrics(predictions, labels):
    predictions = predictions.squeeze().round()
    return classification_report(y_true=labels, y_pred=predictions, output_dict=True)


def auc_scores(predictions, labels, masks=None):
    labels, predictions = np.array(labels), np.array(predictions)
    if masks is not None:
        per_x = [roc_auc_score(lab[mask], preds[mask]) for preds, lab, mask in zip(predictions, labels, masks) if sum(lab[mask]) > 0 and sum(lab[mask]) < len(lab[mask])]
    else:
        per_x = [roc_auc_score(lab, preds) for preds, lab in zip(predictions, labels)]
    return {"AUC": sum(per_x) / len(per_x) if any(per_x) else 1}


def multiclass_label_metrics(predictions, labels):
    predictions = np.argmax(predictions, axis=1)
    return classification_report(
        y_true=labels,
        y_pred=predictions,
        output_dict=True,
    )


def add_prefix(report, prefix=""):
    keys = list(report.keys())
    for key in keys:
        report[prefix + key] = report.pop(key)

    return report


def jaccard_metrics(predictions, labels):
    d = {}
    predictions = predictions.squeeze().round()
    if predictions.shape != labels.shape:
        predictions = predictions[: len(labels[0])]  # fix uneven
    pos_overlap = ((predictions == 1) & (labels == 1)).sum(-1)
    pred_sum = predictions.sum(-1)
    label_sum = labels.sum(-1)
    d["jaccard_precision"] = np.mean(
        np.divide(
            pos_overlap,
            pred_sum,
            out=np.ones_like(pos_overlap).astype(float),
            where=pred_sum != 0,
        )
    )
    d["jaccard_recall"] = np.mean(
        np.divide(
            pos_overlap,
            label_sum,
            out=np.ones_like(pos_overlap).astype(float),
            where=label_sum != 0,
        )
    )
    d["jaccard_f1"] = (
        (2 * d["jaccard_precision"] * d["jaccard_recall"])
        / (d["jaccard_precision"] + d["jaccard_recall"])
        if (d["jaccard_precision"] + d["jaccard_recall"]) > 0
        else 0
    )
    return d


def get_metric_func(p):
    if isinstance(p, tuple):

        def combined_metric_func(predictions, labels, masks=None):
            metrics =  [get_metric_func(metric) for metric in p]
            params =  [(predictions, labels) + ((masks,) if "masks" in inspect.signature(f).parameters else ()) for f in metrics]
            return {
                key: val
                for k in [metric(*params) for (metric, params) in zip(metrics, params)]
                for key, val in k.items()
            }
        return combined_metric_func
    elif p == "single_label_classification":
        return multiclass_label_metrics
    elif p == "regression":
        return binary_label_metrics
    elif p == "jaccard":
        return jaccard_metrics
    elif p == "AUC":
        return auc_scores
    else:
        return multi_label_metrics


class EvaluationMetrics:
    def __init__(
        self,
        num_labels: int,
        metrics_func=None,
        problem_type=None,
        special_ids = []
    ) -> None:
        self.num_labels = num_labels
        self.special_ids = special_ids
        if isinstance(metrics_func, list):
            self.compute_metrics_func = metrics_func
        elif metrics_func is not None:
            self.compute_metrics_func = [metrics_func]
        elif metrics_func is None and problem_type is not None:
            self.compute_metrics_func = []
            if not isinstance(problem_type, list):
                problem_type = [problem_type]
            self.problem_types = problem_type
            for p in self.problem_types:
                func_ = get_metric_func(p)
                self.compute_metrics_func.append(func_)

    def __call__(self, p: EvalPrediction):
        return self.compute_metrics(p)

    def compute_metrics(self, p: EvalPrediction):
        all_labels = p.label_ids
        all_predictions = p.predictions
        class_names = [str(i) for i in range(self.num_labels)]
        cls_report = {"class_names": class_names}

        if isinstance(p.predictions, tuple) and isinstance(p.label_ids, tuple):
            for i, (predictions, labels, metrics_func) in enumerate(
                zip(p.predictions, p.label_ids, self.compute_metrics_func)
            ):
                if "masks" in inspect.signature(metrics_func).parameters and p.inputs is not None:
                    mask = ~np.isin(p.inputs, self.special_ids)
                    report = metrics_func(predictions, labels, mask)
                else:
                    report = metrics_func(predictions, labels)
                cls_report = cls_report | add_prefix(report, str(i) + "_")
            if len(p.predictions) > i + 1:
                losses = p.predictions[i + 1]
                if not isinstance(losses, tuple):
                    losses = (losses.mean().item(),)
                for j in range(i + 1):
                    cls_report[f"loss_{str(j)}"] = losses[j].mean().item()
        else:  # in case of default BERT
            report = self.compute_metrics_func[0](all_predictions[0], all_labels)
            cls_report = cls_report | report
        return cls_report
