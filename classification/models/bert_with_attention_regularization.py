import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sparsemax import Sparsemax
from torch.nn import *
from transformers import (AutoConfig, BertForSequenceClassification,
                          DebertaV2ForSequenceClassification, PreTrainedModel,
                          modeling_outputs)

from classification.models.helpers import (
    CLS_TOKEN_INDEX, activate_predictions,
    get_attention_rollout,
    get_mean_cls_attentions)


class BERTVariantForSequenceClassification(PreTrainedModel):
    @staticmethod
    def from_pretrained(model_path:str, *args, **kwargs):
        if "deberta" in model_path:
            return  DebertaV2ForSequenceClassification.from_pretrained(model_path, *args, **kwargs)
        else:
            return  BertForSequenceClassification.from_pretrained(model_path, *args, **kwargs)


# https://discuss.pytorch.org/t/apply-mask-softmax/14212/12
def masked_softmax(vec, mask, dim=-1, epsilon=1e-8):
    vec = (
        vec.masked_fill(vec.sum(dim, keepdim=True) == 0, 1) * mask
    )  # if no positives, make everything positive
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


def masked_log_softmax(vec, mask, dim=-1, epsilon=1e-8):
    vec = (
        vec.masked_fill(vec.sum(dim, keepdim=True) == 0, 1) * mask
    )  # if no positives, make everything positive
    mt = torch.masked.masked_tensor(vec.float(), mask)
    mt = torch.softmax(mt, dim).get_data() + epsilon
    mt = mt.log()
    return mt


def masked_normalize(vec, mask, dim=1):
    vec = (
        vec.masked_fill(vec.sum(dim, keepdim=True) == 0, 1) * mask
    )  # if no positives, make everything positive
    masked_vec = vec * mask.float()
    masked_sums = masked_vec.sum(dim, keepdim=True) + 1e-8
    return masked_vec / masked_sums


def masked_sigmoid(vec, mask, dim=1):
    vec = (
        vec.masked_fill(vec.sum(dim, keepdim=True) == 0, 1) * mask
    )  # if no positives, make everything positive
    vec = 1 / (1 + torch.exp(-vec))
    masked_vec = vec * mask.float()
    return masked_vec


# https://github.com/hate-alert/HateXplain/blob/master/Models/utils.py
def cross_entropy(input1, target, size_average=True):
    """Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * logsoftmax(input1))


def masked_cross_entropy(input1, target, mask):
    mask = mask.bool()
    cr_ent = 0
    for h in range(0, mask.shape[0]):
        cr_ent += cross_entropy(input1[h][mask[h]], target[h][mask[h]])

    return cr_ent / mask.shape[0]


# Jayaram et al.
def get_mean_attention_weights(raw_attentions):
    attentions = torch.stack([att.mean(1).mean(1) for att in raw_attentions]).mean(0)
    return attentions


def get_max_cls_attentions(
    raw_attentions, attention_layers=range(11, 12), attention_heads=range(0, 12)
):
    # Get the attention weights of the CLS token (averaged)
    attentions = torch.max(
        torch.stack(
            [
                torch.max(layer[:, attention_heads], 1).values[:, CLS_TOKEN_INDEX]
                for layer in raw_attentions
            ]
        )[attention_layers],
        0,
    ).values

    return attentions


def get_sum_cls_attentions(
    raw_attentions, attention_layers=range(11, 12), attention_heads=range(0, 12)
):
    # Get the attention weights of the CLS token (averaged)
    attentions = torch.stack(
        [
            layer[:, attention_heads].sum(1)[:, CLS_TOKEN_INDEX]
            for layer in raw_attentions
        ]
    )[attention_layers].sum(0)

    return attentions


def get_cls_per_head_attentions(
    raw_attentions, attention_layers=range(11, 12), attention_heads=range(0, 12)
):
    # Get the attention weights of the CLS token (averaged)
    attentions = torch.stack(
        [layer[:, attention_heads, CLS_TOKEN_INDEX] for layer in raw_attentions]
    )[attention_layers]
    return attentions


class BertForSequenceClassificationWithAttentionRegularization(
    BERTVariantForSequenceClassification
):
    def __init__(self, config, temperature=None, use_awl=False, **kwargs):
        super().__init__(config, **kwargs)
        self.temperature = temperature if temperature is not None else getattr(config, "temperature", (1, 1))
        start_model_path = getattr(config, "start_model_path")
        self.use_awl = use_awl  # use automatic weighted loss
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        model = super().from_pretrained(
            start_model_path,
        )
        
        missing_attrs = set(dir(model)) - set(dir(self))
        for att in missing_attrs:
            setattr(self, att, getattr(model, att))

        self.config = model.config
        self.config.start_model_path = start_model_path
        self.config.temperature = self.temperature
        self.config.problem_type = config.problem_type
        self.config.use_awl = self.use_awl
        self.config.num_labels = self.num_labels
        if "attention_regularization_method" not in config:
            self.config.attention_regularization_method = self.attention_regularization_method

        self.inner_model = model
        # # Initialize weights and apply final processing
        self.post_init()
            
    @property
    def attention_regularization_method(self):
        return "Unknown"
    
    @property
    def attention_layers(self):
        return range(11, 12)


    def set_temperature(self, temperature):
        self.temperature = temperature
        self.config.temperature = temperature

    def weight_losses(self, primary_loss, secondary_loss):
        return primary_loss * self.temperature[0], secondary_loss * self.temperature[1]

    def get_attentions(self, raw_attentions, attention_mask=None, special_tokens_mask=None):
        return get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )

    def calculate_attention_loss(self, attentions, rationale_mask, **kwargs):
        return self.attention_loss_function(attentions, rationale_mask)

    def combine_loss(self, sequence_loss, attention_loss):
        if self.use_awl:
            return self.awl(sequence_loss, attention_loss)
        else:
            return sequence_loss + attention_loss

    def activate_predictions(self, logits, labels=None):
        predictions = logits

        if isinstance(logits, tuple):
            predictions = logits[0]
            secondary_predictions = logits[1]
            secondary_predictions = activate_predictions(
                secondary_predictions, labels[1], problem_type="top_r"
            )
            labels = labels[0]  # Labels are now the classification labels
            primary_loss = logits[2]
            secondary_loss = logits[3]

        predictions = activate_predictions(
            predictions, labels, problem_type=self.config.problem_type
        )
        if isinstance(logits, tuple):
            return (predictions, secondary_predictions, (primary_loss, secondary_loss))
        else:
            return predictions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=None,
        rationale_mask=None,
        special_tokens_mask=None,
    ):
        inputs = {            
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "output_attentions":output_attentions,
            "output_hidden_states":True}
        model_forward_parameters = inspect.getfullargspec(self.inner_model.forward).args

        for key in list(inputs.keys()):
            if key not in model_forward_parameters:
                inputs.pop(key) 
        
        outputs = self.inner_model.forward(**inputs)

        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.bool()
            
            
            
        if labels is not None:    
            sequence_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            attentions = self.get_attentions(
                outputs["attentions"], attention_mask, special_tokens_mask
            )

            attention_loss = self.calculate_attention_loss(
                attentions=attentions,
                rationale_mask=rationale_mask,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
            )
            sequence_loss, attention_loss = self.weight_losses(
                sequence_loss, attention_loss
            )

            total_loss = self.combine_loss(sequence_loss, attention_loss)
        else:
            total_loss = None
            sequence_loss = None
            attention_loss = None
            attentions = None
            
        return SequenceClassificationAndAttentionRegularizationOutput(
            loss=total_loss,
            sequence_loss=sequence_loss,
            sequence_logits=outputs.logits,
            attention_loss=attention_loss,
            attention_scores=attentions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls, start_model_path, num_labels, problem_type, use_awl=False, *args, **kwargs
    ):
        config = AutoConfig.from_pretrained(start_model_path)
        config.start_model_path = start_model_path
        
        config.temperature = kwargs.get("temperature", getattr(config, "temperature", (1, 1)))
        if isinstance(config.temperature, float):
            config.temperature = (config.temperature, 1)
        config.problem_type = problem_type
        config.use_awl = use_awl
        config.num_labels = num_labels
        return cls(config)


class BertForSequenceClassificationARWithBCE(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "BCE"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.BCELoss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_sigmoid(attentions, ~special_tokens_mask)
        return attentions


class BertForSequenceClassificationARWithMSERollout(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE-rollout"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_normalize(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithMSERolloutTop3(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE-rollout-top3"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss()
        self.attention_heads = range(0, 3)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_normalize(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithBCERollout(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "BCE-rollout"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.BCELoss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_sigmoid(attentions, ~special_tokens_mask)
        return attentions


class BertForSequenceClassificationARWithMAERollout(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MAE-rollout"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.L1Loss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithKLDivRollout(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "KLDiv-rollout"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.KLDivLoss(reduction="batchmean")
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        attentions = torch.log(attentions + 1e-8)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithKLDiv(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "KLDiv"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.KLDivLoss(reduction="batchmean")
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        s = masked_softmax(attentions, ~special_tokens_mask)
        attentions = torch.log(s + 1e-8)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithOrderLoss(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "Order"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = order_loss
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_softmax(attentions, ~special_tokens_mask)  # TODO?
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, special_tokens_mask, attention_mask, **kwargs
    ):
        loss = self.attention_loss_function(attentions, rationale_mask)
        # take sum per instance
        loss = loss.sum(-1)
        return loss.mean()


class BertForSequenceClassificationARWithOrderLossRollout(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "Order-Rollout"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = order_loss
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_softmax(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, special_tokens_mask, attention_mask, **kwargs
    ):
        loss = self.attention_loss_function(attentions, rationale_mask)
        # take sum per instance
        loss = loss.sum(-1)
        return loss.mean()


def summed_masked_attention(attentions, mask):
    if mask.sum() == 0:
        return torch.tensor(0).to(attentions.device)
    a = attentions * mask
    return (1 - a.sum(-1)).mean()


def weighted_MSE_loss(pred, truth, relevance_tensor):
    d = ((pred - truth) ** 2) * relevance_tensor
    wd = d.sum()
    s = relevance_tensor.sum()
    return wd / s


class BertForSequenceClassificationARWithAMr(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "AMr"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = summed_masked_attention
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_softmax(attentions, ~special_tokens_mask)
        return attentions


class BertForSequenceClassificationARWithMSEMAW(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE-MAW"

    @property
    def attention_layers(self):
        return range(0, 12)

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = weighted_MSE_loss
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        attentions = get_mean_attention_weights(raw_attentions)
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask
    ):
        rationales_softmax = masked_normalize(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(
            attentions, rationales_softmax, ~special_tokens_mask
        )


class BertForSequenceClassificationARWithMSE(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_softmax(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithMSETop3(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE-top3"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss()
        self.attention_heads = range(0, 3)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_softmax(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


class BertForSequenceClassificationARWithMAE(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MAE"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.L1Loss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_softmax(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


# Deshpande - 2
class BertForSequenceClassificationARWithMSEWeightedHeads(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE-weighted-heads"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        attentions_per_head = get_cls_per_head_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions_per_head = attentions_per_head[-1]  # take final layer
        return attentions_per_head

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)

        sparsemax = Sparsemax(dim=-1)
        attentions_per_head = attentions

        # TODO write cleaner version
        # loss = self.attention_loss_function(attentions_per_head[:, 0], rationales_softmax)
        attention_loss_per_head = [
            -self.attention_loss_function(
                attentions_per_head[:, head], rationales_softmax
            )
            * self.temperature
            for head in self.attention_heads
        ]
        attention_loss_per_head = torch.stack(attention_loss_per_head)[None, :]
        sparse_masked_attention_head_weights = sparsemax(
            attention_loss_per_head
        ).squeeze(0)
    
        attentions_per_head = (
            attentions_per_head * sparse_masked_attention_head_weights[None, :, None]
        )
        attentions_per_head = attentions_per_head.sum(1)
        attentions = masked_softmax(attentions_per_head, ~special_tokens_mask)

        return self.attention_loss_function(attentions, rationales_softmax)


# Stacey
class BertForSequenceClassificationARWithSSEStacey(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "SSE-top3"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss(reduction="sum")
        self.attention_heads = range(0, 3)
        self.temperature = (1, 1)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        attentions = get_mean_cls_attentions(
            raw_attentions, self.attention_layers, self.attention_heads
        )
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_normalize(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


# HateExplain method
# without temperature
class BertForSequenceClassificationARWithCEMatthew(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "CE-Matthew"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.CrossEntropyLoss()
        self.attention_heads = range(0, 6)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        # Get the attention weights of the CLS token (averaged)
        # attentions = get_mean_cls_attentions(raw_attentions, self.attention_layers, self.attention_heads)
        # attentions = masked_softmax(attentions, ~special_tokens_mask)
        return raw_attentions

    def activate_predictions(self, logits, labels=None):
        predictions = logits[0]

        if isinstance(labels, tuple):
            predictions = logits[0]
            secondary_predictions = logits[1]
            secondary_predictions = torch.sigmoid(
                secondary_predictions[-1].mean(1)[:, CLS_TOKEN_INDEX]
            )
            primary_loss = logits[2]
            secondary_loss = logits[3]

        predictions = activate_predictions(
            predictions, problem_type=self.config.problem_type
        )
        if isinstance(logits, tuple):
            return (predictions, secondary_predictions, (primary_loss, secondary_loss))
        else:
            return predictions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        # rationale_values = masked_softmax(rationale_mask, ~special_tokens_mask)
        rationale_values = rationale_mask

        loss_att = 0
        for head in self.attention_heads:
            attention_weights = attentions[-1][:, head, CLS_TOKEN_INDEX, :]
            loss_att += self.temperature[1] * masked_cross_entropy(
                attention_weights, rationale_values, ~special_tokens_mask
            )

        return loss_att

    def weight_losses(self, primary_loss, secondary_loss):
        return primary_loss, secondary_loss


class BertForSequenceClassificationARWithCERollout(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "CE-rollout"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.CrossEntropyLoss()
        self.attention_heads = range(0, 12)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        attentions = get_attention_rollout(raw_attentions)
        attentions = get_mean_cls_attentions(
            [attentions], range(0, 1), self.attention_heads
        )
        attentions = masked_normalize(attentions, ~special_tokens_mask)
        attentions = torch.log(attentions + 1e-8)
        return attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, special_tokens_mask, **kwargs
    ):
        rationales_softmax = masked_softmax(rationale_mask, ~special_tokens_mask)
        return self.attention_loss_function(attentions, rationales_softmax)


# Deshpande
class BertForSequenceClassificationARWithMSEDeshpande(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "MSE-DESHP"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.attention_loss_function = torch.nn.MSELoss()
        self.attention_heads = range(0, 6)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        return raw_attentions

    def activate_predictions(self, logits, labels=None):
        predictions = logits

        if isinstance(logits, tuple):
            predictions = logits[0]
            secondary_predictions = logits[1]
            secondary_predictions = torch.sigmoid(
                secondary_predictions[-1].mean(1)[:, CLS_TOKEN_INDEX]
            )
            primary_loss = logits[2]
            secondary_loss = logits[3]

        predictions = activate_predictions(
            predictions, problem_type=self.config.problem_type
        )
        if isinstance(logits, tuple):
            return (predictions, secondary_predictions, (primary_loss, secondary_loss))
        else:
            return predictions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        rationale_values = masked_softmax(rationale_mask, ~special_tokens_mask)
        loss_att = 0
        for layer in self.attention_layers:
            for head in self.attention_heads:
                attention_weights = attentions[layer][:, head, CLS_TOKEN_INDEX, :]
                loss_att += self.attention_loss_function(
                    attention_weights, rationale_values
                )

        return loss_att


class BertForSequenceClassificationARWithEAR(
    BertForSequenceClassificationWithAttentionRegularization
):
    @property
    def attention_regularization_method(self):
        return "EAR"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def get_attentions(
        self, raw_attentions, attention_mask=None, special_tokens_mask=None
    ):
        return raw_attentions

    def calculate_attention_loss(
        self, attentions, rationale_mask, attention_mask, special_tokens_mask, **kwargs
    ):
        # Make sure non-tokens are ignored when calculating mask
        # attentions = attentions.masked_fill(~attention_mask.bool(), -1e9)
        return compute_negative_entropy(attentions, attention_mask)

    def activate_predictions(self, logits, labels=None):
        predictions = logits

        if isinstance(logits, tuple):
            predictions = logits[0]
            secondary_predictions = logits[1]
            secondary_predictions = secondary_predictions[-1].mean(1)[
                :, CLS_TOKEN_INDEX
            ]
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


# from https://github.com/INK-USC/ER-Test/blob/main/src/utils/losses.py
def order_loss(attention_values, rationale_mask: torch.Tensor):
    # Fix for when an instance has only rationales
    max_non_rationale_attr = torch.max(
        (1 - rationale_mask) * attention_values, dim=-1
    ).values
    max_non_rationale_attr += 1e-8
    max_non_rationale_attr = max_non_rationale_attr.unsqueeze(1).expand(
        -1, attention_values.shape[1]
    )

    ordered_attr = torch.where(
        rationale_mask == 1,
        torch.sub(
            torch.div((rationale_mask * attention_values), max_non_rationale_attr),
            torch.tensor(1.0).to(rationale_mask.device),
        ),
        torch.tensor(0.0).to(rationale_mask.device),
    )
    loss = torch.square(
        torch.minimum(
            ordered_attr, torch.zeros(size=ordered_attr.shape).to(rationale_mask.device)
        )
    )
    return loss


# https://aclanthology.org/2022.findings-acl.88/
def compute_negative_entropy(
    inputs: tuple, attention_mask: torch.Tensor, return_values=False
):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS
    """
    inputs = torch.stack(inputs)  # Â LayersBatchHeadsSeqlenSeqlen
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    # Â average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        # Â get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]

        # Â get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        # Â get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        # Â store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy


@dataclass
class SequenceClassificationAndAttentionRegularizationOutput(
    modeling_outputs.ModelOutput
):
    loss: torch.FloatTensor = None
    sequence_logits: torch.FloatTensor = None
    attention_scores: torch.FloatTensor = None
    sequence_loss: Optional[torch.FloatTensor] = None
    attention_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


MODEL_CLASSES = {}

for k, v in list(globals().items()):
    if isinstance(v, type):
        if hasattr(v, "attention_regularization_method"):
            k = k.split("BertForSequenceClassificationARWith")[-1]
            MODEL_CLASSES[k] = v
