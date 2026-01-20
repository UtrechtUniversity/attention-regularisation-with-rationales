import inspect
import typing
from dataclasses import dataclass, field
import torch.nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from transformers import (AutoModelForSequenceClassification, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

import classification.models.bert_with_attention_regularization as bert_with_attention_regularization
from classification.models.helpers import activate_predictions

MODEL_CLASSES = (
    bert_with_attention_regularization.MODEL_CLASSES 
)


    
def get_top_value_mask(arr: np.ndarray, bins: int = 10):
    # For every item in the array, find the values in the 10th bin
    return np.apply_along_axis(
        lambda x: np.where(x >= np.histogram(x, bins=bins)[1][-1], 1.0, 0.0), 1, arr
    )


def rationale_token_metrics(attention_predictions, rationale_labels):
    attention_predictions = get_top_value_mask(attention_predictions)
    return classification_report(
        y_true=rationale_labels,
        y_pred=attention_predictions,
        output_dict=True,
    )["macro avg"]


class MetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


def default_activation(
    predictions, labels=None, task_type="single_label_classification"
):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return (activate_predictions(predictions, labels, task_type),)


def setup_temperature(eval_metrics, strategy="initial", regularization_bias=1):
    if isinstance(strategy, int):
        return (1, strategy)
    if strategy == "initial" and regularization_bias > 0:
        # Initial
        primary_weight = 1 / eval_metrics["eval_loss_0"]
        secondary_weight = (
            1 / (abs(eval_metrics["eval_loss_1"]) + 2e-5)
        ) * regularization_bias
    elif strategy == "balance":
        primary_weight = 1
        secondary_weight = (
            eval_metrics["eval_loss_0"] / eval_metrics["eval_loss_1"]
        ) * regularization_bias
    else:
        primary_weight = 1
        secondary_weight = 1 * regularization_bias

    return (primary_weight, secondary_weight)


# TODO takes up quite some memory / is slow
class TrainerWithRationales(Trainer):
    def __init__(
        self,
        args,
        attention_layers=11,
        save_path=None,
        settings=None,
        **kwargs,
    ):
        if "model" not in kwargs:
            model = self.init_model(
                start_model_path=args.start_model_path,
                num_labels=args.num_labels,
                task_type=args.task_type,
                attention_regularization_method=args.attention_regularization_method,
                use_awl=args.weighting_strategy == "awl",
                settings=settings,
                num_token_labels=args.num_token_labels,
                seed=args.seed
            )
            kwargs["model"] = model

        super().__init__(
            args=args,
            **kwargs,
        )
        self.attention_layers = (
            range(attention_layers, attention_layers + 1)
            if isinstance(attention_layers, int)
            else attention_layers
        )
        self.save_path = save_path
        classifier_func = (
            self.model.activate_predictions
            if hasattr(self.model, "activate_predictions")
            else default_activation
        )

        self.preprocess_logits_for_metrics = classifier_func

        model_forward_parameters = inspect.getfullargspec(self.model.forward).args

        label_names = [
            x
            for x in [
                "labels",
                "rationale_mask",
                "rationale_attention",
                "rationale_mask_per_class",
            ]
            if x in self.train_dataset.features and x in model_forward_parameters
        ]

        self.label_names = label_names

        # setup temperature
        self._setup_temperature()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inner_model = (
            model.module
            if isinstance(model, DataParallel)
            or isinstance(model, DistributedDataParallel)
            else model
        )
        if "labels" in inputs:
            if inner_model.config.problem_type == "single_label_classification":
                inputs["labels"] = inputs["labels"].long()
            else:
                inputs["labels"] = inputs["labels"].float()

        model_forward_parameters = inspect.getfullargspec(inner_model.forward).args

        for key in list(inputs.keys()):
            if key not in model_forward_parameters:
                inputs.pop(key)

        if "rationale_mask" in inputs:
            inputs["rationale_mask"] = inputs["rationale_mask"].float()
        if "rationale_mask_per_class" in inputs:
            inputs["rationale_mask_per_class"] = inputs[
                "rationale_mask_per_class"
            ].float()

        outputs = model.forward(**inputs.to(inner_model.device), output_attentions=True)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def re_init(self, **kwargs):
        return TrainerWithRationales(
            model=kwargs.get("model", self.model),
            args=kwargs.get("args", self.args),
            train_dataset=kwargs.get("train_dataset", self.train_dataset),
            eval_dataset=kwargs.get("eval_dataset", self.eval_dataset),
            tokenizer=kwargs.get("tokenizer", self.tokenizer),
            data_collator=kwargs.get("data_collator", self.data_collator),
            compute_metrics=kwargs.get("compute_metrics", self.compute_metrics),
            preprocess_logits_for_metrics=kwargs.get(
                "preprocess_logits_for_metrics", self.preprocess_logits_for_metrics
            ),
            save_path=kwargs.get("save_path", self.save_path),
            callbacks=kwargs.get("callbacks", self.callback_handler.callbacks),
        )

    def init_model(
        self,
        start_model_path=None,
        num_labels=None,
        task_type=None,
        attention_regularization_method=None,
        seed=42,
        settings=None,
        **kwargs,
    ):
        if attention_regularization_method is None:
            attention_regularization_method = self.args.attention_regularization_method

        start_model_path = start_model_path or self.args.start_model_path
        num_labels = num_labels or self.args.num_labels
        task_type = task_type or self.args.task_type
        if settings is not None and "seed" in settings:
            seed = settings["seed"]
        set_seed(seed)
        
        model_class = MODEL_CLASSES.get(
            attention_regularization_method, AutoModelForSequenceClassification
        )

        classifier_type = model_class
        # remove parameters that are not in the model from_pretrained param list
        model_parameters = inspect.getfullargspec(classifier_type.from_pretrained).args
        for k in list(kwargs):
            if k not in model_parameters:
                print(f"Removing param {k} as it is not used by {model_class.__name__}")
                kwargs.pop(k)

        self.model = classifier_type.from_pretrained(
            start_model_path, num_labels=num_labels, problem_type=task_type, **kwargs
        )
        if (
            settings
            and "temperature" in settings
            and hasattr(self.model, "set_temperature")
        ):
            # Use pre-defined temperature
            self.model.set_temperature(settings["temperature"])
        else:
            if hasattr(self, "eval_dataset"):
                self._setup_temperature()
        return self.model.cuda()

    def save_model(self, output_dir: typing.Optional[str] = None, _internal_call: bool = False):
        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir, safe_serialization=False)
        self.model.config.save_pretrained(output_dir)

    def _setup_temperature(self):
        # calculate the initial loss (only for balance)
        if self.args.weighting_strategy in ["balance"]:
            eval_metrics = self.evaluate()

            if "eval_loss_0" in eval_metrics:
                print(
                    f"initial losses: {eval_metrics['eval_loss_0']} and {eval_metrics['eval_loss_1']}"
                )

            if "eval_loss_1" in eval_metrics and self.model.temperature == (1, 1):
                self.model.set_temperature(
                    setup_temperature(
                        eval_metrics,
                        strategy=self.args.weighting_strategy,
                        regularization_bias=self.args.regularization_bias,
                    )
                )
                print(f"Temperature set to {self.model.temperature}")


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


def compute_stacey_loss(
    desired_att_weights: torch.tensor,
    attention_output: torch.tensor,
    size_in_batch: int,
    seq_len: int,
    att_mask: torch.tensor,
) -> torch.tensor:
    """
    We find the difference between the current and the desired attention

    Args:
        desired_att_weights: desired attention values for the batch (0s,1s)
        attention_output: attention output for layer being supervised
        size_in_batch: observations in minibatch
        seq_len: sequence length of input_ids (including padding)
        att_mask: attention masks for minibatch

    Returns:
        all_attention_loss: attention loss for the minibatch

    """

    device = "cuda"
    loss_type = "sse"
    attention_header_numbers = range(12)
    # Create loss tensor
    all_attention_loss = torch.tensor(0).to(device)

    # First we calculate the attention weight that we need
    for i_val in range(size_in_batch):
        attention_mask = att_mask[i_val]
        # We scale the desired weight tensor to 1
        weight_mult = torch.sum(desired_att_weights[i_val, :])

        # We only scale the weights if the first value is not -1
        if desired_att_weights[i_val, :][0] >= 0:
            desired_att_weights[i_val, :] = desired_att_weights[i_val, :] / weight_mult

            assert torch.allclose(
                torch.sum(desired_att_weights[i_val, :]),
                torch.tensor([1.0]).to(device),
            )

        if loss_type == "sse":
            # We loop over every head
            for head_no in attention_header_numbers:
                single_head_CLS_att = attention_output[i_val, head_no, 0, :]
                # Calculate loss for specific head
                loss_atten_head = single_head_CLS_att - desired_att_weights[i_val, :]

                # Include loss if first element not -1
                if desired_att_weights[i_val, :][0] >= 0:
                    all_attention_loss = all_attention_loss + (
                        torch.sum(torch.square(loss_atten_head))
                    )

        elif loss_type == "kl":
            single_head_CLS_att = torch.mean(attention_output[i_val, :, 0, :], 0)

            epsilon = torch.tensor([0.0001]).to(device)

            a = desired_att_weights[i_val, :]
            a = torch.where(a == 0, epsilon, a)
            a = torch.true_divide(a, torch.sum(a))
            b = single_head_CLS_att
            b = torch.true_divide(b, torch.sum(b))

            # Â Find the KL divergence:
            kld = torch.tensor([0]).to(device)

            if 0 in attention_mask:
                max_len = (attention_mask == 0).int().nonzero().min()
            else:
                max_len = attention_mask.shape[0]

            for i in range(max_len):
                if a[i] != torch.tensor([0.0]).to(device):
                    kld = kld + (-1) * a[i] * torch.log(b[i] / a[i]).to(device)
            all_attention_loss = all_attention_loss + kld

    # This counteracts dividing by the number of attention heads
    if loss_type == "kl":
        all_attention_loss = all_attention_loss * float(len(attention_header_numbers))

    return all_attention_loss


@dataclass
class RationaleTrainingArguments(TrainingArguments):
    regularization_bias: float = field(
        default=1.0,
        metadata={
            "help": "This values is used to weight the regularization loss against the classification loss. Defaults to 1."
        },
    )
    weighting_strategy: str = field(
        default=None,
        metadata={
            "help": "This values indicates the strategy to use when weighting the regularization loss against the classification loss. Defaults to None."
        },
    )
    attention_regularization_method: str = field(
        default=None,
        metadata={
            "help": "Value to specifiy how the attention patterns should be regularized. Defaults to None."
        },
    )
    start_model_path: str = field(
        default=None,
        metadata={"help": "Path of the base model to finetune. Defaults to None."},
    )
    num_labels: int = field(
        default=2,
        metadata={
            "help": "Value to specifiy the number of labels in the task. Defaults to 2."
        },
    )
    task_type: str = field(
        default="single_label_classification",
        metadata={
            "help": 'The type of classification task. Defaults to "single_label_classification".'
        },
    )

    num_token_labels: str = field(
        default=None,
        metadata={
            "help": "Number of token labels. Used in BertForSequenceClassificationAndTokenClassification. Defaults to None."
        },
    )
    
    seed: int = field(
        default=None,
        metadata={
            "help": "Seed to train model on. Defaults to None"
        },
    )
    
    
