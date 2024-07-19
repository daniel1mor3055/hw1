from collections import Counter

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer("basic_english")


class WikiTextDataset(Dataset):
    def __init__(self, split, vocab):
        self.dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split).select(range(100))
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"]


def build_vocab(data_iter):
    counter = Counter()
    for text in data_iter:
        counter.update(tokenizer(text))
    return counter


def get_vocab():
    train_iter = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")["text"]
    vocab_counter = build_vocab(train_iter)
    vocab = {token: idx for idx, (token, _) in enumerate(vocab_counter.items(), start=2)}
    vocab["<unk>"] = 0
    vocab["<pad>"] = 1
    return vocab


def text_pipeline(text, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenizer(text)]


def collate_batch(batch, vocab):
    text_list = []
    for _text in batch:
        processed_text = torch.tensor(text_pipeline(_text, vocab), dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=vocab["<pad>"], batch_first=True)


def get_dataloaders(batch_size, vocab):
    train_dataset = WikiTextDataset(split="train", vocab=vocab)
    test_dataset = WikiTextDataset(split="test", vocab=vocab)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, vocab),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, vocab),
    )
    return train_dataloader, test_dataloader
