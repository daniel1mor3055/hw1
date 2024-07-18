import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def get_vocab():
    train_iter = WikiText2(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def text_pipeline(text, vocab):
    return vocab(tokenizer(text))


def collate_batch(batch, vocab):
    text_list = []
    for _text in batch:
        processed_text = torch.tensor(text_pipeline(_text, vocab), dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=vocab["<unk>"], batch_first=True)


def get_dataloaders(batch_size, vocab):
    train_iter, test_iter = WikiText2(split="train"), WikiText2(split="test")
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