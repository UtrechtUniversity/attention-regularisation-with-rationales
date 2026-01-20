import copy
import importlib
import itertools
import json
import os
import os.path
import random
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker
from torch import optim
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertForSequenceClassification,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          TrainerCallback, set_seed)

import classification.models.bert_with_attention_regularization as bert_with_attention_regularization
from calibration.calculate_calibration import calculate_calibration_error
from classification.models.helpers import activate_predictions
from classification.models.metrics import EvaluationMetrics
from classification.trainer_with_rationales import (RationaleTrainingArguments,
                                                    TrainerWithRationales,
                                                    setup_temperature)
from pretraining.tokenizer import (
                                   shuffle_mask, shuffle_mask_non_rationale,
                                   switch_mask, tokenize_dataset)

MODEL_CLASSES = (
    bert_with_attention_regularization.MODEL_CLASSES
)

tracker = OfflineEmissionsTracker(country_iso_code="NLD", allow_multiple_runs=True)


class OnTrainEndCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        pass


def default_activation(
    predictions, labels=None, task_type="single_label_classification"
):
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return (activate_predictions(predictions, labels, task_type),)


def load_sota_from_hs(sota, start_path, rationale_type, dataset_selection, dataset, method, problem_type, start_model):
    start_path_ = start_path
    rationale_type = rationale_type.replace("None", "baseline").replace("random_nr", "normal")
    search_path = f"{start_path_}{rationale_type}/{dataset_selection.replace("-shuffled", "")}/HS/"
    settings_path = [
        x
        for x in os.listdir(search_path)
        if x.startswith(
            f"temp_{dataset}_{method}_{problem_type}_{start_model}"
        )
    ]
    assert (
        len(settings_path) > 0
    ), f"{search_path}/temp_{dataset}_{method}_{problem_type}_"
    hs_sota = json.load(
        open(
            search_path
            + settings_path[0]
            + "/final/results/results.json",
            "r",
        )
    )
    sota.update(hs_sota)
    return sota

def do_hyperparameter_search(
    start_model_path,
    task_type=None,
    attention_regularization_method="",
    num_labels=2,
    output_dir=None,
    num_token_labels=None,
    trainer: TrainerWithRationales = None,
    seed=42,
    search_lr=True,
    n_trials=20,
):
    def optuna_hp_space(trial):
        space = {
            "regularization_bias": trial.suggest_float(
                "regularization_bias", 0.1, 150, log=search_lr
            )
        }
        if search_lr:
            space["learning_rate"] = trial.suggest_float(
                "learning_rate", 1e-8, 1e-4, log=True
            )

        return space

    def my_objective(metrics):
        if "eval_0_macro avg" in metrics:
            return metrics["eval_0_macro avg"]["f1-score"]
        else:
            return metrics["eval_macro avg"]["f1-score"]

    def model_init(trial=None):
        model = trainer.init_model(
            start_model_path=start_model_path,
            num_labels=num_labels,
            task_type=task_type,
            attention_regularization_method=attention_regularization_method,
            seed=seed,
            num_token_labels=num_token_labels,
        )
        # Weighting strategy is HS
        if hasattr(model, "set_temperature"):
            if trial is not None:
                eval_metrics = trainer.evaluate()
                if hasattr(trial, "params"):
                    regularization_bias = trial.params["regularization_bias"]
                elif hasattr(trial, "hyperparameters"):
                    regularization_bias = trial.hyperparameters["regularization_bias"]

                temperature = setup_temperature(
                    eval_metrics,
                    strategy="balance",
                    regularization_bias=regularization_bias,
                )
            else:
                temperature = (1, 1)

            model.set_temperature(temperature)

        return model.cuda()

    if task_type is None:  # used by model_init
        task_type = trainer.model.config.problem_type

    if output_dir is None:
        output_dir = f"temp_hyp_{dataset}_{attention_regularization_method}"

    trainer.model_init = model_init
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=n_trials,
        compute_objective=my_objective,
    )
    if isinstance(best_trial, list):
        best_trial = best_trial[0]
    if "learning_rate" in best_trial.hyperparameters:
        trainer.args.learning_rate = best_trial.hyperparameters["learning_rate"]

    trainer.model = model_init(best_trial)
    trainer.model_init = None
    return trainer, best_trial


def save_final_model_and_results(
    trainer,
    test_data,
    save_dir,
    initial_eval_metrics,
    settings,
    remove_temp_results=True,
    emissions=-1,
):
    classifier = trainer.model

    eval_preds, eval_labels, eval_metrics = trainer.predict(
        test_data, metric_key_prefix="eval"
    )

    cls_labels = eval_labels[0] if isinstance(eval_labels, tuple) else eval_labels
    calibration_error = calculate_calibration_error(eval_preds[0], cls_labels)

    final_path = f"{save_dir}/final"
    Path(final_path).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_path)
    # Remove all checkpoints to save space
    if remove_temp_results:
        for d in os.listdir(save_dir):
            if d.startswith("checkpoint") or d.startswith("run"):
                shutil.rmtree(save_dir + "/" + d)

    results = {
        "train_emissions": emissions,
        "calibration_error": calibration_error,
        "regularization_bias": trainer.args.regularization_bias,
        "learning_rate": trainer.args.learning_rate,
        "initial_eval_metrics": initial_eval_metrics,
        "seed": seed,
        "epochs": settings["epochs"],
        "batch_size": settings["batch_size"],
        "eval_results": eval_metrics,
    }
    if hasattr(classifier, "temperature"):
        results["temperature"] = classifier.temperature
    if hasattr(classifier, "use_awl") and classifier.use_awl:
        results["temperature"] = list((x.item() for x in classifier.awl.params))
    results["train_results"] = [
        x for x in trainer.state.log_history if "eval_loss" in x
    ]
    
    results["training_loss"] = [
        x for x in trainer.state.log_history if "loss" in x
    ]

    final_result_path = final_path + "/results"
    Path(final_result_path).mkdir(parents=True, exist_ok=True)
    json.dump(results, open(f"{final_result_path}/results.json", "w"))
    id_predictions = {
        "predictions": eval_preds[0].tolist(),
        "labels": cls_labels.tolist(),
    }
    json.dump(id_predictions, open(f"{final_result_path}/id_predictions.json", "w"))

    if ood_script is not None:
        ood_script.run_model(final_path)

    return eval_metrics


def train_bert_classifier(
    data,
    tokenizer,
    start_model_path,
    task_type=None,
    attention_regularization_method="",
    num_labels=2,
    num_token_labels=None,
    settings=None,
    save_path=None,
    rationale_type="normal",
    weighting_strategy=None,
    dataset_selection="full",
    override=True,
    regularization_bias=1,
    seed=42,
    
    early_stopping=True
):
    if settings is None:
        settings = {
            "batch_size": 32,
            "learning_rate": 0.00002,
            "optimizer": "adam2_torch",
            "epochs": 3,
        }
        print(f"Using default settings: {settings}")
        
    start_model = settings["start_model"]

    if weighting_strategy != "HS":
        settings = load_sota_from_hs(settings, start_path, rationale_type, dataset_selection, dataset, attention_regularization_method, problem_type, start_model)
        
    # if "seed" not in settings:
    settings["seed"] = seed
        

    start_model = settings["start_model"]

    output_dir = f"{start_path}{rationale_type}/{dataset_selection}/{weighting_strategy}/temp_{dataset}_{attention_regularization_method}_{task_type}_{start_model}_{dataset_selection}_{rationale_type}_{weighting_strategy}"
    
    if not early_stopping:
        output_dir = output_dir.replace("temp_", "noe_")
    
    print(output_dir)
    if weighting_strategy != "HS":
        output_dir += f"/{seed}"
    print(
        f"Training model with {attention_regularization_method} on {dataset_selection} examples with weighting strategy '{weighting_strategy}' on {rationale_type} rationales with seed {seed}."
    )
    if not override and os.path.exists(output_dir):
        print(f"Skipping training as override == {override} and {output_dir} exists")
        return

    torch.cuda.empty_cache()

    compute_metrics = EvaluationMetrics(
        num_labels, problem_type=[task_type, ("jaccard", "AUC")], special_ids=tokenizer.all_special_ids
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    eval_strategy = "steps"
    training_args = RationaleTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=settings["epochs"],
        eval_strategy=eval_strategy,
        logging_strategy=eval_strategy,
        load_best_model_at_end=True,
        save_strategy=eval_strategy,
        save_total_limit=1,
        eval_accumulation_steps=4,
        learning_rate=settings["learning_rate"],
        optim=settings["optimizer"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=8,
        metric_for_best_model="loss",
        warmup_ratio=0.10,
        weight_decay=0.01,
        regularization_bias=regularization_bias,
        weighting_strategy=weighting_strategy,
        attention_regularization_method=attention_regularization_method,
        start_model_path=start_model_path,
        num_labels=num_labels,
        logging_steps = 0.1,
        save_steps=0.1,
        task_type=task_type,
        num_token_labels=num_token_labels,
        include_for_metrics = ["inputs"],
        seed=settings["seed"],

    )

    trainer = TrainerWithRationales(
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        save_path=save_path,
        settings=settings,
        callbacks=([EarlyStoppingCallback(early_stopping_patience=2)] if early_stopping else []),
    )

    initial_eval_metrics = trainer.evaluate(data["dev"], metric_key_prefix="eval")

    if weighting_strategy == "HS":
        trainer, best_trial = do_hyperparameter_search(
            start_model_path,
            attention_regularization_method=attention_regularization_method,
            task_type=problem_type,
            num_labels=num_labels,
            num_token_labels=num_labels,
            output_dir=output_dir,
            trainer=trainer,
        )
        new_args = trainer.args
        for hp, v in best_trial.hyperparameters.items():
            setattr(new_args, hp, v)

        trainer = trainer.re_init(args=new_args)  # reset trainer
    elif weighting_strategy == "temperature_search":
        trainer, best_trial = do_hyperparameter_search(
            start_model_path,
            attention_regularization_method=attention_regularization_method,
            task_type=problem_type,
            num_labels=num_labels,
            num_token_labels=num_labels,
            output_dir=output_dir,
            trainer=trainer,
            search_lr=False,
            n_trials=20,
        )

        # Save per run
        for root, dirs, files in os.walk(output_dir):
            if "trainer_state.json" in files:
                state = json.load(open(root + "/trainer_state.json", "r"))
                run_path = root
                trainer.args.regularization_bias = state["trial_params"][
                    "regularization_bias"
                ]
                # Overrides model
                trainer.init_model(
                    start_model_path=run_path,
                    num_labels=num_labels,
                    task_type=task_type,
                    attention_regularization_method=attention_regularization_method,
                    seed=seed,
                    num_token_labels=num_token_labels,
                )

                # Save model results with the test set
                eval_metrics = save_final_model_and_results(
                    trainer,
                    data["test"],
                    os.path.dirname(run_path),  # Save one up
                    initial_eval_metrics,
                    settings,
                    emissions=-1,
                )
        return None

    tracker.start()
    trainer.train()

    eval_metrics = save_final_model_and_results(
        trainer,
        data["test"],
        training_args.output_dir,
        initial_eval_metrics,
        settings,
        emissions=-1,
    )

    return trainer.model_wrapped, None, None, eval_metrics, trainer.state.log_history

if __name__ == "__main__":
    
    datasets_ = [ "HateExplain-multilabel","IMDB","SST", "VAST"]
    for dataset in datasets_:
        print(dataset)
        start_path = f"data/{dataset}/"
        try:
            ood_script = importlib.import_module(f"data.{dataset}.OOD.ood_performance")
        except:
            ood_script = None

        if os.path.exists(start_path + "sota.json"):
            sota = json.load(open(start_path + "sota.json", "r"))
        else:
            sota = {
                "batch_size": 8,
                "learning_rate": 0.0001,
                "optimizer": "adamw_torch",
                "epochs": 5,
                "start_model": "bert-base-uncased",
            }
            print("Using default configuration.")

        start_model = sota["start_model"]
        start_model_path = start_model

        df = pd.read_pickle(start_path + "all_data.pickle")

        if "set" in df.columns:
            dictionary = {
                key: datasets.Dataset.from_pandas(df[df.set == key])
                for key in df.set.unique()
            }

            classification_data = datasets.DatasetDict(dictionary)
        else:
            classification_data = datasets.Dataset.from_pandas(df)
            train_test = classification_data.train_test_split(test_size=0.3, seed=42)
            test_val = train_test["test"].train_test_split(test_size=0.4, seed=42)
            classification_data = datasets.DatasetDict(
                {
                    "train": train_test["train"],
                    "test": test_val["test"],
                    "dev": test_val["train"],
                }
            )

        tokenizer = AutoTokenizer.from_pretrained(
            start_model_path, truncate=True, add_prefix_space=True, local_files_only=True
        )
        tokenizer.model_max_length = sota.get("model_max_length", 512)

        if isinstance(classification_data["train"]["labels"][0], int):
            num_labels = len(set(classification_data["train"]["labels"]))
            if num_labels == 1:
                problem_type = "regression"
            else:
                problem_type = "single_label_classification"

        else:
            num_labels = len(classification_data["train"]["labels"][0])
            problem_type = "multi_label_classification"

        classification_data = tokenize_dataset(
            classification_data,
            tokenizer,
            "words",
            prefix="",
            rationale_mask_column="rationale_mask",
            rationale_attention_column="rationales_softmax",
            rationale_mask_per_class_column="rationale_mask_per_class",
        )

        train_len = len(classification_data["train"])

        # Take out instances where rats are at indices > 512
        if "Hate" not in dataset:
            old_len = train_len
            classification_data = classification_data.filter(
                lambda x: sum(x["rationale_mask"]) > 0
            )

            # Print classification_data characteristics
            train_len = len(classification_data["train"])
            if train_len < old_len:
                print(f"Removed {old_len - train_len} examples without rationales.")
            if "SST" in dataset:
                old_len = train_len
                classification_data = classification_data.filter(
                    lambda x: sum(x["rationale_mask"])
                    != sum([1 for y in x["special_tokens_mask"] if y == 0])
                )

                # Print classification_data characteristics
                train_len = len(classification_data["train"])
                if train_len < old_len:
                    print(
                        f"Removed {old_len - train_len} examples where all tokens are rationales."
                    )

        words_len = [len(x) for x in classification_data["train"]["words"]]
        token_len = [
            len(x) - sum(x) for x in classification_data["train"]["special_tokens_mask"]
        ]
        print(f"Average tokens:  {sum(token_len) / train_len}")
        print(f"Min tokens: {min(token_len)}")
        print(f"Max tokens: {max(token_len)}")
        rationale_len = [sum(x) for x in classification_data["train"]["rationale_mask"]]
        print(f"Average rationale tokens:  {sum(rationale_len) / train_len}")
        print(f"Min rationale tokens: {min(rationale_len)}")
        print(f"Max rational tokens: {max(rationale_len)}")
        rat_word_ratio = [x / y for x, y in zip(rationale_len, token_len)]
        print(f"Average rationale ration:  {sum(rat_word_ratio) / train_len}")
        print(f"Min rationale ratio: {min(rat_word_ratio)}")
        print(f"Max rationale ratio: {max(rat_word_ratio)}")

        print(f"Test sample size: {len(classification_data['test'])}")

        methods = [
            "KLDiv",
            "MAE",
            "MSE",
            "OrderLoss",
            "AMr",
        ]

        seeds = [42,24,16]
        train_baselines = False
        weighting_strategies = ["from-HS"]
        rationale_types = ["random_nr", "normal"] 
        dataset_selections = ["100", "200", "500", "100%"]
        spurious_correlations_frac = 0
        shuffle_class_labels = False
        early_stopping = True
        training_combinations = list(
            itertools.product(rationale_types, weighting_strategies)
        )
        
        for seed in seeds:
            print(f"Training with seed {seed}")

            for dataset_selection in dataset_selections:
                selected_data = copy.deepcopy(classification_data)
                print(f"Dataset selection of {dataset_selection}")
                set_seed(seed)

                if dataset_selection.isnumeric():
                    n = min(int(dataset_selection), len(selected_data["train"]))
                    print(f"Taking sample of {n}")

                    selected_data["train"] = (
                        selected_data["train"].shuffle(seed=42).select(range(n))
                    )
                elif dataset_selection[-1:] == "%" and dataset_selection[:-1].isnumeric():
                    dataset_selection = int(dataset_selection[:-1])
                    print(f"Taking sample of {dataset_selection} %")
                    dataset_selection = str(int(
                        (len(selected_data["train"]) * dataset_selection) / 100
                    ))
                    selected_data["train"] = (
                        selected_data["train"]
                        .shuffle(seed=42)
                        .select(range(int(dataset_selection)))
                    )
                    print(len(selected_data["train"]))

                ignore_tokens_random_selection = []
                if spurious_correlations_frac > 0:
                    replace_tokens = tokenizer.encode(".?!")[1:-1]

                    def add_spurious_correlations(row, replace_tokens):
                        if random.randint(0,100) < spurious_correlations_frac:
                            label = row["labels"]
                            if label == 0:
                                row["input_ids"] = [id if id not in replace_tokens else tokenizer.vocab["!"] for id in row["input_ids"]]
                            elif label == 1:
                                row["input_ids"] = [id if id not in replace_tokens else tokenizer.vocab["?"] for id in row["input_ids"]]
                        return row
                    dataset_selection = f"{dataset_selection}-{spurious_correlations_frac}"
                    selected_data = selected_data.map(lambda row: add_spurious_correlations(row, replace_tokens))
                    ignore_tokens_random_selection = replace_tokens
                    try:
                        ood_script = importlib.import_module(f"data.{dataset}-spurious.OOD.ood_performance")
                    except:
                        ood_script = None
                        
                if shuffle_class_labels:
                    print("Shuffling labels...")
                    dataset_selection = f"{dataset_selection}-shuffled"
                    def shuffle_labels(row, possible_values):
                        row["labels"] = np.random.choice(possible_values)
                        return row
                    train_labels = selected_data["train"]["labels"]
                    selected_data = selected_data.map(lambda row: shuffle_labels(row, train_labels))
                    ood_script = None #Disable OOD


                if train_baselines:
                    train_results = train_bert_classifier(
                        selected_data,
                        tokenizer,
                        start_model_path,
                        task_type=problem_type,
                        num_labels=num_labels,
                        settings=sota,
                        save_path=start_path,
                        rationale_type="baseline",
                        weighting_strategy=weighting_strategies[0],
                        dataset_selection=dataset_selection,
                        num_token_labels=num_labels,
                        seed=seed,
                        early_stopping=early_stopping
                    )

                    train_results = train_bert_classifier(
                        selected_data,
                        tokenizer,
                        start_model_path,
                        task_type=problem_type,
                        num_labels=num_labels,
                        settings=sota,
                        save_path=start_path,
                        rationale_type="baseline",
                        attention_regularization_method="EAR",
                        weighting_strategy=weighting_strategies[0],
                        dataset_selection=dataset_selection,
                        num_token_labels=num_labels,
                        seed=seed,
                        early_stopping=early_stopping
                    )

                for rationale_type, weighting_strategy in training_combinations:
                    modified_data = copy.deepcopy(selected_data)

                    for method in methods:

                        if rationale_type == "swapped":
                            modified_data = modified_data.map(switch_mask)
                        elif rationale_type == "random":
                            modified_data = modified_data.map(lambda row: shuffle_mask(row, ignore_tokens_random_selection))
                        elif rationale_type == "random_nr":
                            modified_data = modified_data.map(lambda row: shuffle_mask_non_rationale(row, ignore_tokens_random_selection))

                        train_results = train_bert_classifier(
                            modified_data,
                            tokenizer,
                            start_model_path,
                            task_type=problem_type,
                            num_labels=num_labels,
                            settings=sota,
                            save_path=start_path,
                            rationale_type=rationale_type,
                            weighting_strategy=weighting_strategy,
                            dataset_selection=dataset_selection,
                            attention_regularization_method=method,
                            num_token_labels=num_labels,
                            seed=seed,
                            early_stopping=early_stopping
                        )
                        torch.cuda.empty_cache()

                    modified_data.cleanup_cache_files()
