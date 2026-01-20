import json
from collections import defaultdict
import numpy as np
import shap
import torch

from classification.models.helpers import (activate_predictions,
                                           masked_softmax, min_max_normalize)
from classification.models.metrics import auc_scores, jaccard_metrics
from pretraining.tokenizer import tokenize_dataset


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_outputs(path, texts, num_labels=None, problem_type=None):
    def get_func(x):
        if isinstance(x, tuple):
            return {"logits": x[0].cpu()}
        else:
            return {"logits": x.logits.cpu()}

    output = BE.encode_in_chunks(texts, get_func, output_attentions=False)

    return {
        "logits": torch.vstack([x["logits"] for x in output]),
        "predictions": torch.sigmoid(
            torch.vstack([x["logits"] for x in output])
        ).round(),
    }


def preprocess_dataset(data, tokenizer, dataset:str): 
    data = tokenize_dataset(
        data,
        tokenizer,
        "words",
        prefix="",
        rationale_mask_column="rationale_mask",
    )
    
    data = data.filter(
        lambda x: sum(x["rationale_mask"][:512]) > 0
    ) # remove instance where rats are truncated

    # Throw out all words that are not included in the model prediction
    data = data.map(
        lambda s: {"truncated_text": " ".join(s["words"][: s["word_ids"][-2]])}
    )
    if "Hate" not in dataset:
        data = data.filter(
            lambda x: sum(x["rationale_mask"]) > 0
        )

    if "SST" in dataset:
        data = data.filter(
            lambda x: sum(x["rationale_mask"])
            != sum([1 for y in x["special_tokens_mask"] if y == 0])
        )


    return data 

def get_shap_explanation_jaccard_scores(test_set, BE, do_preprocess=True):

    if do_preprocess:
        test_set = preprocess_dataset(test_set, BE.tokenizer)

    def l(x):
        x = list(x)  # to list as only strings are accepted
        return BE.predict(x)["probabilities"].numpy()

    explainer = shap.Explainer(l, BE.tokenizer)
    shap_values = explainer(test_set["truncated_text"], batch_size=64)
    predictions = l(test_set["truncated_text"])
    test_set = test_set.add_column("prediction", predictions.tolist())
    shap_scores = defaultdict(list)
    raw_scores = defaultdict(list)
    for t, s in zip(test_set, shap_values):
        label = torch.tensor(t["labels"])
        prediction =  t["prediction"]
        prediction_label = np.argmax(prediction)
        rationale_mask = torch.tensor(t["rationale_mask"]).unsqueeze(0)
        token_mask = ~torch.tensor(t["special_tokens_mask"]).unsqueeze(0).bool()
        # values per token, only take into account rationales passed to the model during training time
        
        values = torch.from_numpy(s.values[: len(rationale_mask[0])]).unsqueeze(0)
        n_tokens = values.shape[1]
        top_activate_values = activate_predictions(
            values[:, :, label], rationale_mask, "top_r"
        )
        metrics = jaccard_metrics(
            top_activate_values, rationale_mask[:, : n_tokens]
        )
        min_max_shap_scores = min_max_normalize(values[:,:, prediction_label])
        normalized_shap_scores = masked_softmax(min_max_shap_scores, torch.ones_like(min_max_shap_scores) )
        metrics = metrics | auc_scores(normalized_shap_scores, rationale_mask[:, : n_tokens], token_mask[:, :n_tokens])
        for metric in metrics:
            shap_scores[metric].append(metrics[metric])
        raw_scores["raw"].append(values[:, :, ].numpy().tolist())
        raw_scores["top_r"].append(top_activate_values.numpy().tolist())
        raw_scores["normalize"].append(normalized_shap_scores.numpy().tolist())
        raw_scores["rationale_mask"].append(rationale_mask.numpy().tolist())

        raw_scores["prediction_label"].append(prediction_label)
        raw_scores["prediction"].append(prediction)
        raw_scores["label"].append(int(label))

    for metric in shap_scores:
        shap_scores[metric] = sum(np.array(shap_scores[metric])) / len(
            shap_scores[metric] 
        )
    shap_scores["raw"] = raw_scores
    
    return shap_scores