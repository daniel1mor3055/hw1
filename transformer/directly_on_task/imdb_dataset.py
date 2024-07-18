import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def get_vocab():
    train_iter = IMDB(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def text_pipeline(text, vocab):
    return vocab(tokenizer(text))


def label_pipeline(label):
    return label - 1


def collate_batch(batch, vocab):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text, vocab), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list, dtype=torch.int64), pad_sequence(
        text_list, padding_value=vocab["<unk>"], batch_first=True
    )


def get_dataloaders(batch_size, vocab):
    train_iter, test_iter = IMDB(split="train"), IMDB(split="test")
    train_dataloader = DataLoader(
        list(train_iter),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, vocab),
    )
    test_dataloader = DataLoader(
        list(test_iter),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, vocab),
    )
    return train_dataloader, test_dataloader
