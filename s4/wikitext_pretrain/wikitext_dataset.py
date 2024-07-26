import os

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"])


class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer):
        self.dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split).select(range(100))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"]


def yield_texts(data_iter):
    for text in data_iter:
        yield text


def get_tokenizer_and_vocab():
    tokenizer_file = "tokenizer.json"

    # Check if the tokenizer file already exists
    if os.path.exists(tokenizer_file):
        print("Tokenizer loaded from file.")
        return Tokenizer.from_file(tokenizer_file)

    train_iter = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    tokenizer.train_from_iterator(yield_texts(train_iter), trainer)

    # Save the tokenizer to a file
    tokenizer.save(tokenizer_file)
    print("Tokenizer trained and saved to file.")

    return tokenizer


def text_pipeline(text, tokenizer):
    return tokenizer.encode(text).ids


def collate_batch(batch, tokenizer):
    text_list = []
    for _text in batch:
        processed_text = torch.tensor(text_pipeline(_text, tokenizer), dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=tokenizer.token_to_id("<pad>"), batch_first=True)


def get_dataloaders(batch_size, tokenizer):
    train_dataset = WikiTextDataset(split="train", tokenizer=tokenizer)
    test_dataset = WikiTextDataset(split="test", tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, tokenizer),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, tokenizer),
    )
    return train_dataloader, test_dataloader
