import torch
import torch.nn as nn

CLS_TOKEN_INDEX = 0

def activate_predictions(predictions, labels=None, problem_type=None):
    if problem_type == "single_label_classification":
        predictions = torch.softmax(predictions, -1)
    elif problem_type == "regression":
        predictions = torch.sigmoid(predictions)
    elif problem_type == "top_r" and labels is not None:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.clone()
        top_n = labels.sum(-1)
        idx = [
            torch.topk(preds, k=min(n, preds.shape[-1]), axis=-1)[1]  # add 2 for CLS and SEP tokens
            for (preds, n) in zip(predictions, top_n)
        ]
        for i, x in enumerate(idx):
            predictions[i] = 0  # Replace everything with 0
            predictions[i, x] = 1  # replace top r rationales with 1

    else:
        predictions = torch.sigmoid(predictions)
    return predictions


def predictions_to_labels(predictions, problem_type=None):
    if problem_type == "single_label_classification":
        predictions = torch.argmax(predictions, -1)
    else:
        predictions = predictions.round()
    return predictions


def get_attention_rollout(attention_layers):
    """Computes attention rollout from the given list of attention matrices.
    https://arxiv.org/abs/2005.00928
    """
    rollout = attention_layers[0]
    for layer in attention_layers[1:]:
        rollout = torch.matmul(
            0.5 * layer + 0.5 * torch.eye(layer.shape[-2], device=layer.device), rollout
        )  # the computation takes care of skip connections

    return rollout

def min_max_normalize(v:torch.Tensor):
    return (v - v.min()) / (v.max() - v.min())

def masked_softmax(vec: torch.Tensor, keep_mask, dim=1, epsilon=1e-5):
    vec = vec.masked_fill(
        vec.sum(dim, keepdim=True) == 0, 1
    )  # if no positives, make everything positive
    exps = torch.exp(vec)
    masked_exps = exps * keep_mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


def masked_normalize(vec, mask, dim=1):
    vec = vec.masked_fill(
        vec.sum(dim, keepdim=True) == 0, 1
    )  # if no positives, make everything positive
    masked_vec = vec * mask.float()
    masked_sums = masked_vec.sum(dim, keepdim=True) + 1e-5
    return masked_vec / masked_sums

# def min_max_normalize():
    

def masked_sigmoid(vec, mask, dim=1):
    vec = vec.masked_fill(
        vec.sum(dim, keepdim=True) == 0, 1
    )  # if no positives, make everything positive
    vec = 1 / (1 + torch.exp(-vec))
    masked_vec = vec * mask.float()
    return masked_vec


# Taken from https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: initial number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(
                1 + self.params[i] ** 2
            )
        return loss_sum


def get_mean_attention_weights_final_layer(raw_attentions):
    attentions = raw_attentions[-1].mean(1).mean(1)
    return attentions


def get_mean_cls_attentions(
    raw_attentions, attention_layers=range(11, 12), attention_heads=range(0, 12)
):
    # Get the attention weights of the CLS token (averaged)
    attentions = torch.mean(
        torch.stack(
            [
                torch.mean(layer[:, attention_heads], 1)[:, CLS_TOKEN_INDEX]
                for layer in raw_attentions
            ]
        )[attention_layers],
        0,
    )
    return attentions
