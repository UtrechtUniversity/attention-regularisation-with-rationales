

import torch
from datasets.utils.logging import disable_progress_bar
from classification.models.bert_with_attention_regularization import (
    get_attention_rollout, get_mean_cls_attentions)
from classification.models.helpers import activate_predictions
from pretraining.encoder import BertEmbedder

disable_progress_bar()


def print_metrics(metrics):
    print(f"   | P   | R    | f1  ")

    for val in metrics:
        it = metrics[val]
        if isinstance(it, dict) and "precision" in it:
            print(
                f"{val} | {round(it['precision'], 1)} |  {round(it['recall'], 1)} |  {round(it['f1-score'], 1)}"
            )


def _activate_predictions(self, logits, labels=None):
    predictions = logits

    if isinstance(logits, tuple):
        predictions = logits[0]
        secondary_predictions = logits[1]
        secondary_predictions = activate_predictions(
            secondary_predictions, labels[1], problem_type="top_r"
        )
        labels = labels[0]
        primary_loss = logits[2]
        secondary_loss = logits[3]

    predictions = activate_predictions(
        predictions, labels, problem_type=self.config.problem_type
    )
    if isinstance(logits, tuple):
        return (predictions, secondary_predictions, (primary_loss, secondary_loss))
    else:
        return predictions


def get_outputs(
    model_or_path, texts, rationale_mask, num_labels=None, problem_type=None
):
    if isinstance(model_or_path, str):
        BE = BertEmbedder.from_path(
            model_or_path,
            classification=True,
            num_labels=num_labels,
            problem_type=problem_type,
        )
    else:
        BE = model_or_path

    def get_func(x):
        if isinstance(x, tuple):
            logits = x[0].cpu()
            attentions = [y.cpu() for y in x[2]]
        else:
            if "attention_scores" in x:
                attentions = x.attention_scores
            else:
                attentions = x.attentions
            if "logits" in x:
                logits = x.logits.cpu()
            elif "sequence_logits" in x:
                logits = x.sequence_logits.cpu()

        mean_cls_attentions = get_mean_cls_attentions(attentions).cpu()
        rolled_out_attentions = get_attention_rollout(attentions).cpu()
        rolled_out_attentions = get_mean_cls_attentions(
            [rolled_out_attentions], range(0, 1)
        )

        return {
            "logits": logits,
            "mean_cls_attentions": mean_cls_attentions,
            "attention_rollout": rolled_out_attentions,
        }

    output = BE.encode_in_chunks(texts, get_func, output_attentions=True)
    logits = torch.vstack([x["logits"] for x in output])
    mean_cls_attentions = [x["mean_cls_attentions"] for x in output]

    max_n_tokens = max([n.shape[-1]for n in mean_cls_attentions])
    mean_cls_attentions = torch.vstack([torch.nn.functional.pad(x, pad=(0, max_n_tokens - x.shape[-1], )) for x in mean_cls_attentions])
    rolled_out_attentions = [x["attention_rollout"] for x in output]
    rolled_out_attentions = torch.vstack([torch.nn.functional.pad(x, pad=(0, max_n_tokens - x.shape[-1], )) for x in rolled_out_attentions])

    problem_type = BE.inference_model.config.problem_type

    return {
        "logits": logits,
        "probabilities": activate_predictions(logits, problem_type=problem_type),
        "mean_cls_attentions": mean_cls_attentions,
        "attention_rollout": rolled_out_attentions,
        "activated_mean_cls_attentions": activate_predictions(
            mean_cls_attentions, rationale_mask, problem_type="top_r"
        ),
        "activated_attention_rollout": activate_predictions(
            rolled_out_attentions, rationale_mask, problem_type="top_r"
        ),
    }
