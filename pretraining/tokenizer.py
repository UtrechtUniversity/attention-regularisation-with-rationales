import difflib
import itertools
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import AddedToken
from transformers import AutoTokenizer


def get_word_counts(texts: List[str] | str):
    texts = [texts] if isinstance(texts, str) else texts

    corpus = [word for text in texts for word in text.split(" ")]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    freqs = X.sum(axis=0).tolist()[0]
    return list(zip(vectorizer.get_feature_names_out(), freqs))


def get_top_n_percent_threshold(freqs: List[int], n=1):
    b_bins = int(100 / n)
    hist = np.histogram(freqs, bins=b_bins)
    threshold = int(hist[1][1])
    return hist[1][np.where(hist[1] >= threshold)[0][0]]


def find_common_words(texts: List[str]):
    words, freqs = zip(*get_word_counts(texts))
    n = get_top_n_percent_threshold(freqs, 1)
    counts_n = [x for x, y in zip(words, freqs) if y > n]
    return counts_n


def extend_tokenizer(tokenizer: AutoTokenizer, tokens: List[str]):
    tokens = [AddedToken(token, single_word=True) for token in tokens]
    old_vocab = tokenizer.vocab
    added_tokens = tokenizer.add_tokens(tokens)
    new_vocab = tokenizer.vocab
    new_tokens = set(new_vocab) - set(old_vocab)
    print(f"Added {added_tokens} / {len(new_tokens)} new tokens")
    return tokenizer, new_tokens


# Get a list of words with how often they occur in the dataset, multiplied by the number of tokens required.
def get_high_impact_words(
    texts: List[str], tokenizer: AutoTokenizer, with_impacts=False, top_n=1
):
    words, freqs = zip(*get_word_counts(texts))
    word_tokens = [tokenizer.tokenize(word) for word in words]
    word_impacts = [freq * len(tokens) for freq, tokens in zip(freqs, word_tokens)]
    n = get_top_n_percent_threshold(word_impacts, top_n)
    words_with_impact = sorted(list(zip(words, word_impacts)), key=lambda x: x[1])
    n_impact_words = [
        word if not with_impacts else (word, y)
        for word, y in words_with_impact
        if y > n
    ]
    return n_impact_words


def tokenization_loss(texts: List[str], tokenizer):
    tokens = [tokenizer.tokenize(text) for text in texts]
    truncation_limit = tokenizer.model_max_length
    losses = [
        max(1, len(token[truncation_limit:]))
        / max(1, len(text.split(" ")[truncation_limit:]))
        for text, token in zip(texts, tokens)
    ]
    return sum(losses) / len(losses)


def set_mask_for_overlap(snippet, sequence, mask=None):
    if mask is None:
        mask = np.repeat(0, len(sequence))
    s = difflib.SequenceMatcher(None, snippet, sequence)
    sn, se, size = s.find_longest_match()
    mask[se : se + size] = 1
    return mask


def get_zipf_formula_for_corpus(texts: str | List[str]):
    """https://pubmed.ncbi.nlm.nih.gov/24417251/"""
    words, word_counts = zip(*get_word_counts(texts))
    n_unique_words = len(word_counts) / 1000000
    n_words = sum(word_counts) / 1000000

    def zipf_formula(frequency):
        return np.log10((frequency + 1) / (n_words + n_unique_words)) + 3

    return (
        zipf_formula,
        dict(zip(words, word_counts)),
        dict(zip(words, [zipf_formula(x) for x in word_counts])),
    )


def get_rationale_mask(mask, word_ids, prefix=""):
    rationale_mask = [
        mask[word_id] if word_id is not None else 0 for word_id in word_ids
    ]
    return {prefix + "rationale_mask": rationale_mask}


def get_rationale_mask_per_class(masks, word_ids, prefix=""):
    default = [0] * len(masks[0])
    rationale_masks = [
        masks[word_id] if word_id is not None else default for word_id in word_ids
    ]
    return {prefix + "rationale_mask_per_class": rationale_masks}


def get_rationale_mask_from_string(
    rationales, original_input_ids, tokenizer=None, prefix=""
):
    if is_singular := isinstance(original_input_ids[0], int):
        original_input_ids = [original_input_ids]
        rationales = [rationales]
    masks = [np.zeros(len(input_ids)) for input_ids in original_input_ids]

    for mask, rationale, input_ids in zip(masks, rationales, original_input_ids):
        if len(rationale) > 0:
            rationale_input_ids = tokenizer(
                rationale,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if isinstance(rationale_input_ids[0], int):
                set_mask_for_overlap(rationale_input_ids, input_ids, mask)
            elif isinstance(rationale_input_ids[0][0], int):
                for rationale_encoding in rationale_input_ids:
                    set_mask_for_overlap(rationale_encoding, input_ids, mask)
    return {prefix + "rationale_mask": masks[0] if is_singular else masks}


def get_words_above_zipf_value(texts: str | List[str], zipf_value):
    _, _, zipf_dict = get_zipf_formula_for_corpus(texts)
    return [k for (k, v) in zipf_dict.items() if v > zipf_value]


def split_list(lst, val):
    return [
        list(group) for k, group in itertools.groupby(lst, lambda x: x == val) if not k
    ]


def combine_masks(base_mask, additional_mask, fill_val=2):
    new_mask = np.array(base_mask)
    new_mask[np.array(additional_mask) != 0] = fill_val
    return new_mask


def tokenize_dataset(
    dataset,
    tokenizer,
    column="sentences",
    prefix="",
    rationale_column=None,
    max_length=None,
    rationale_mask_column=None,
    rationale_mask_per_class_column=None,
    rationale_attention_column=None,
):
    prefix = prefix + "_" if prefix else prefix
    max_length = max_length or tokenizer.model_max_length

    def _tokenize_(text):
        words = text.split(" ") if isinstance(text, str) else text

        encodings = tokenizer(
            words,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
            is_split_into_words=True,
        )
        return {
            prefix + "input_ids": encodings["input_ids"],
            prefix + "attention_mask": encodings["attention_mask"],
            prefix + "special_tokens_mask": encodings["special_tokens_mask"],
            prefix + "word_ids": encodings.word_ids(),
        }

    dataset = dataset.map(_tokenize_, input_columns=[column], keep_in_memory=True)

    if rationale_column:
        dataset = dataset.map(
            get_rationale_mask_from_string,
            input_columns=[rationale_column, prefix + "input_ids"],
            fn_kwargs={"prefix": prefix},
        )

    if rationale_mask_column:
        dataset = dataset.map(
            get_rationale_mask,
            input_columns=[rationale_mask_column, prefix + "word_ids"],
            fn_kwargs={"prefix": prefix},
        )

    if rationale_mask_per_class_column:
        dataset = dataset.map(
            get_rationale_mask_per_class,
            input_columns=[rationale_mask_per_class_column, prefix + "word_ids"],
            fn_kwargs={"prefix": prefix},
        )

    if rationale_attention_column:

        def get_rationale_values(values, word_ids):
            rationale_attention = [
                values[word_id] if word_id is not None else 0 for word_id in word_ids
            ]

            return {
                prefix + "rationale_attention": rationale_attention,
            }

        dataset = dataset.map(
            get_rationale_values,
            input_columns=[rationale_attention_column, prefix + "word_ids"],
        )

    return dataset


def switch_mask(row):
    row["rationale_mask"] = [
        ((x - 1) * -1) * int(not y)
        for x, y in zip(row["rationale_mask"], row["special_tokens_mask"])
    ]
    return row


def mask_non_rationales(row, mask_token_id):
    row["input_ids"] = [
        y if x else mask_token_id
        for x, y in zip(row["rationale_mask"], row["input_ids"])
    ]
    return row


def mask_rationales(row, mask_token_id):
    row["input_ids"] = [
        y if not x else mask_token_id
        for x, y in zip(row["rationale_mask"], row["input_ids"])
    ]
    return row


def shuffle_mask(row, exclude=[]):
    
    rationale_mask = np.stack(row["rationale_mask"])
    rationale_mask_per_class = np.stack(row["rationale_mask_per_class"])
    not_special_tokens_mask = np.stack(row["special_tokens_mask"]) != 1
    n_rationales = sum(rationale_mask)
    #exclude rationales
    include_mask = [id not in exclude and sp == 1 for id, sp in zip(row["input_ids"], not_special_tokens_mask)]
    rationale_indexes = np.random.choice(np.where(include_mask)[0], min(n_rationales, sum(include_mask)))
    rationale_mask = np.zeros_like(rationale_mask)
    rationale_mask[rationale_indexes] = 1
    rationale_mask_per_class = np.zeros_like(rationale_mask_per_class)
    rationale_mask_per_class[rationale_indexes] = 1
    row["rationale_mask"] = rationale_mask
    row["rationale_mask_per_class"] = rationale_mask_per_class
    return row

def shuffle_mask_non_rationale(row, exclude=[]):
    rationale_mask = np.stack(row["rationale_mask"])
    rationale_mask_per_class = np.stack(row["rationale_mask_per_class"])
    not_special_tokens_mask = np.stack(row["special_tokens_mask"]) != 1
    n_rationales = sum(rationale_mask)
    # exclude rationales
    include_mask = [id not in exclude and r == 0 and sp for id, r, sp in zip(row["input_ids"], rationale_mask, not_special_tokens_mask)]
    # Check if any non-rationales left after mask
    non_rationale_options = len(rationale_mask[include_mask])
    if non_rationale_options == 0:        
        # If no non-rationale candidates are available, allow rationale selection
        non_rationale_options = max(int(n_rationales / 2), 1) # but keep at least one rationale
        include_mask = [id not in exclude and sp for id, sp in zip(row["input_ids"], not_special_tokens_mask)]

    rationale_indexes = np.random.choice(np.where(include_mask)[0], min(n_rationales, non_rationale_options))
    rationale_mask = np.zeros_like(rationale_mask)
    rationale_mask[rationale_indexes] = 1
    rationale_mask_per_class = np.zeros_like(rationale_mask_per_class)
    rationale_mask_per_class[rationale_indexes] = 1
    row["rationale_mask"] = rationale_mask
    row["rationale_mask_per_class"] = rationale_mask_per_class
    return row