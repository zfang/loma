from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset


def build_tokenized_datasets(raw_datasets, tokenizer, dataset_text_field, seqlen):
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            truncation=True,
            max_length=seqlen,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == seqlen:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    return tokenized_datasets


def get_wikitext2(tokenizer, train_nsamples, test_nsamples, seqlen):
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    joined_train = Dataset.from_list([{"text": " ".join(train["text"].select(range(train_nsamples * seqlen)))}])
    joined_test = Dataset.from_list([{"text": "\n\n".join(test["text"].select(range(test_nsamples * seqlen)))}])

    raw_datasets = DatasetDict(
        {
            "train": joined_train,
            "test": joined_test,
        }
    )

    return build_tokenized_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        dataset_text_field="text",
        seqlen=seqlen,
    )


def get_ptb(tokenizer, train_nsamples, test_nsamples, seqlen):
    train = load_dataset("ptb_text_only", "penn_treebank", split="train")
    test = load_dataset("ptb_text_only", "penn_treebank", split="test")
    joined_train = Dataset.from_list([{"text": " ".join(train["sentence"].select(range(train_nsamples * seqlen)))}])
    joined_test = Dataset.from_list([{"text": " ".join(test["sentence"].select(range(test_nsamples * seqlen)))}])

    raw_datasets = DatasetDict(
        {
            "train": joined_train,
            "test": joined_test,
        }
    )

    return build_tokenized_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        dataset_text_field="sentence",
        seqlen=seqlen
    )


def get_c4(tokenizer, train_nsamples, test_nsamples, seqlen):
    raw_datasets = DatasetDict(
        {
            "train": load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train"
            ).shuffle().select(range(train_nsamples)),
            "test": load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation"
            ).select(range(test_nsamples)),
        }
    )

    return build_tokenized_datasets(
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        dataset_text_field="text",
        seqlen=seqlen
    )


def get_datasets(name, tokenizer, train_nsamples, test_nsamples, seqlen=512):
    if "wikitext2" in name:
        return get_wikitext2(tokenizer, train_nsamples, test_nsamples, seqlen)
    if "ptb" in name:
        return get_ptb(tokenizer, train_nsamples, test_nsamples, seqlen)
    if "c4" in name:
        return get_c4(tokenizer, train_nsamples, test_nsamples, seqlen)
