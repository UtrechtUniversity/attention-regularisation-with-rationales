import gc
from itertools import islice
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForPreTraining,
                          AutoModelForSequenceClassification, AutoTokenizer)

from classification.models.bert_with_attention_regularization import (BertForSequenceClassificationWithAttentionRegularization)
from classification.models.helpers import (activate_predictions,
                                           predictions_to_labels)


def pad_chunks(chunks_list:List):
    first = next(iter(chunks_list))
    if isinstance(first, dict):
        for key in first.keys():
            last_dims =  [c[key].shape[-1] for c in chunks_list]
            if len(set(last_dims)) > 1:
                max_last_dims = max(last_dims)
                for c in chunks_list:
                    c[key] = torch.vstack([torch.nn.functional.pad(x, pad=(0, max_last_dims - x.shape[-1])) for x in c[key]])
    return chunks_list

def find_indices_of_sublist(x, sublist):
    return [
        i for i in range(0, len(x)) if list(x[i : i + len(sublist)]) == list(sublist)
    ]


def chunks(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())



class BertEmbedder:
    def __init__(
        self,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def from_path(
        path,
        truncation_limit=512,
        new_model_name: str = None,
        classification=False,
        num_labels=None,
        problem_type=None,
    ):
        model = BertEmbedder()
        model.tokenizer = AutoTokenizer.from_pretrained(
            path, model_max_length=truncation_limit, add_prefix_space=True
        )
        config = AutoConfig.from_pretrained(path)
        config.num_labels = num_labels or config.num_labels
        config.problem_type = problem_type or config.problem_type

        if classification:
            if isinstance(temperature := getattr(config, "temperature", None), list):
                model.inference_model = (
                    BertForSequenceClassificationWithAttentionRegularization.from_pretrained(
                        path,
                        num_labels=config.num_labels,
                        problem_type=config.problem_type,
                        temperature=temperature
                    )
                    .to(model.device)
                    .eval()
                )
            
            else:
                model.inference_model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        path, config=config
                    )
                    .to(model.device)
                    .eval()
                )
        else:
            model.inference_model = (
                AutoModel.from_pretrained(path, config=config).to(model.device).eval()
            )


        model.pretraining_model = AutoModelForPreTraining.from_pretrained(path)
        model.setup_chunk_size_formula()
        model._name = path.split("/")[-1]
        if new_model_name is not None:
            model.set_model_names(new_model_name)
        return model

    @staticmethod
    def from_encoder(tokenizer, encoder):
        model = BertEmbedder()
        model.tokenizer = tokenizer
        model.inference_model = encoder.to(model.device).eval()
        model.pretraining_model = encoder.to(model.device)
        model.setup_chunk_size_formula()
        model._name = encoder.name_or_path.split("/")[-1]
        return model

    def setup_chunk_size_formula(self):
        trainable_parameters = self.inference_model.num_parameters()

        def chunk_size_formula(max_len):
            free_memory, total_memory = torch.cuda.mem_get_info()
            if free_memory < (total_memory / 2):
                gc.collect()
                torch.cuda.empty_cache()
                free_memory = torch.cuda.mem_get_info()[0]

            batch_size = free_memory / 4 / ((max_len) + trainable_parameters)
            batch_size = int(batch_size)
            return batch_size

        self.chunk_size_formula = chunk_size_formula

    def set_model_names(self, new_model_name: str):
        self.tokenizer.name_or_path = new_model_name
        self.inference_model.name_or_path = new_model_name
        self.pretraining_model.name_or_path = new_model_name

    def tokenization_loss(self, texts: List[str]):
        tokens = [self.tokenizer.tokenize(text) for text in texts]
        truncation_limit = self.tokenizer.model_max_length
        losses = [
            max(1, len(token[truncation_limit:]))
            / max(1, len(text.split(" ")[truncation_limit:]))
            for text, token in zip(texts, tokens)
        ]
        return sum(losses) / len(losses)

    def tokenize(
        self, texts: List[str], return_tensors="pt", return_special_tokens_mask=False
    ):
        is_split_into_words = isinstance(texts[0], list)
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
            is_split_into_words=is_split_into_words,
            return_special_tokens_mask=return_special_tokens_mask,
        )

    def encode_tokens(
        self, tokens: List, output_hidden_states=False, output_attentions=False
    ):
        with torch.no_grad():
            return self.inference_model(
                **tokens,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )

    def tokenize_and_embed(
        self,
        texts,
        output_hidden_states=False,
        output_attentions=False,
        return_special_tokens_mask=False,
    ):
        tokens = self.tokenize(
            texts, return_special_tokens_mask=return_special_tokens_mask
        ).to(self.device)
        torch.cuda.empty_cache()
        return self.encode_tokens(tokens, output_hidden_states, output_attentions)

    def get_max_token_len_of_input(self, texts):
        is_split_into_words = isinstance(texts[0], list)
        token_lengths = self.tokenizer(
            texts,
            truncation=True,
            return_length=True,
            is_split_into_words=is_split_into_words,
        )["length"]
        return max(token_lengths)

    def encode_in_chunks(
        self,
        texts: List = None,
        get_func=lambda x: x,
        output_hidden_states=False,
        output_attentions=False,
        return_special_tokens_mask=False,
        tokens=None,
    ):
        if tokens is not None:
            max_num_tokens = max(len(t) for t in tokens)
            chunk_func = lambda x: self.encode_tokens(
                x, output_hidden_states, output_attentions
            )
            texts = tokens
        else:
            max_num_tokens = self.get_max_token_len_of_input(texts)

            chunk_func = lambda x: self.tokenize_and_embed(
                x,
                output_hidden_states,
                output_attentions,
                return_special_tokens_mask,
            )

        chunk_size = self.chunk_size_formula(max_num_tokens)
        all_chunks = chunks(texts, chunk_size)
        all_chunks = [get_func(chunk_func(chunk)) for chunk in all_chunks]
        padded_chunks = pad_chunks(all_chunks)
        return padded_chunks

    def encode_texts(self, texts: List):
        return self.encode_in_chunks(texts)

    def get_last_hidden_state_embedding(self, texts: List, return_cuda=False):
        def get_func(x):
            output = torch.mean(x.last_hidden_state, dim=1)
            return output if return_cuda else output.detach().cpu()

        output = self.encode_in_chunks(texts, get_func)
        output = torch.vstack(output)
        return output

    def get_hidden_state_embedding(self, texts: List, return_cuda=False):
        def get_func(x):
            return x.hidden_states

        output = self.encode_in_chunks(texts, get_func, True)
        states = [
            [torch.mean(chunk[layer], dim=1) for chunk in output]
            for layer in range(len(output[0]))
        ]
        states = [torch.vstack(layer) for layer in states]
        if not return_cuda:
            return [layer.detach().cpu() for layer in states]
        return states

    def get_perplexity_of_word(self, word, sent):
        tokenized_word = self.tokenizer.tokenize(word)
        word_ids = self.tokenizer.convert_tokens_to_ids(tokenized_word)
        n_tokens = len(tokenized_word)
        masked_sent = sent.replace(word, self.tokenizer.mask_token * n_tokens, 1)
        tokenized_inputs = self.tokenizer(masked_sent, return_tensors="pt")["input_ids"]
        mask_index = torch.where(tokenized_inputs == self.tokenizer.mask_token_id)
        outputs_probs = torch.softmax(
            self.pretraining_model(tokenized_inputs).logits, -1
        )
        inverse_perplexity = outputs_probs[mask_index][
            range(outputs_probs[mask_index].shape[0]), tuple(word_ids)
        ].mean
        return 1 - inverse_perplexity

    def get_word_embedding(self, texts, words):
        def get_func(x):
            return x.last_hidden_state

        output = self.encode_in_chunks(texts, get_func, True)
        embeddings = torch.vstack(output)
        tokens = self.tokenize(texts)
        word_embeddings = []
        for word, sent_token, embedding in zip(words, tokens["input_ids"], embeddings):
            token_sequence = self.tokenize(word)["input_ids"][0][1:-1]
            indices = find_indices_of_sublist(sent_token, token_sequence)
            word_embedding = [
                torch.mean(embedding[i : i + len(token_sequence)], dim=0)
                for i in indices
            ]  # first take average of all tokens
            word_embedding = torch.mean(
                torch.stack(word_embedding), dim=0
            )  # then take average of all words in sentence
            word_embeddings.append(word_embedding)
        return torch.stack(word_embeddings)

    @property
    def name(self):
        return self._name

    def encode_dataset(self, dataset, text_column, encode_func, prefix=None):
        name = prefix or self.name

        def encode(batch):
            embeddings = flatten_and_encode_texts(batch, encode_func)
            return {f"{name}_vectors": embeddings}

        return dataset.map(encode, input_columns=[text_column], batched=True)

    def split_string(self, text):
        tokens = self.tokenizer.tokenize(text)
        return [t.replace("##", "") for t in tokens if t not in "\"',.?!"]

    def predict(self, texts):
        problem_type = self.inference_model.config.problem_type

        def get_func(x):
            if isinstance(x, tuple):
                return x[0].cpu()
            elif hasattr(x, "sequence_logits"):
                return x.sequence_logits.cpu()
            else:
                return x.logits.cpu()

        outputs = self.encode_in_chunks(texts, get_func, output_attentions=False)
       
        probabilities = torch.vstack(
            [
                activate_predictions(output, problem_type=problem_type)
                for output in outputs
            ]
        )
        prediction_labels = predictions_to_labels(probabilities, problem_type)

        return {
            "logits": torch.vstack(outputs),
            "probabilities": probabilities,
            "predictions": prediction_labels,
        }


def flatten_and_encode_texts(texts, encode_func):
    if isinstance(texts[0], str):
        return encode_func(texts)
    elif isinstance(texts[0][0], str):
        nums = [len(text) for text in texts]
        flattened_texts = [t for text in texts for t in text]  # Flatten list
        encodings = iter(encode_func(flattened_texts))
        return [
            np.vstack(tuple(islice(encodings, n))) for n in tqdm(nums)
        ]  # Unflatten list


def encode_list_of_texts(texts_list, model_names):
    all_embeddings = dict()
    for model_name in model_names:
        embedder = BertEmbedder(model_name)
        embeddings = flatten_and_encode_texts(
            texts_list, embedder.get_last_hidden_state_embedding
        )

        name = model_name.split("/")[-1]
        all_embeddings[name] = embeddings
    return all_embeddings