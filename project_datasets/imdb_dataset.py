import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"])


# Prepare your dataset (list of sentences)
def yield_texts(data_iter):
    for _, text in data_iter:
        yield text


def get_tokenizer_and_vocab():
    train_iter = IMDB(split="train")
    tokenizer.train_from_iterator(yield_texts(train_iter), trainer)
    return tokenizer


def text_pipeline(text, tokenizer):
    return tokenizer.encode(text).ids


def label_pipeline(label):
    return label - 1


def collate_batch(batch, tokenizer):
    label_list, text_list = [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text, tokenizer), dtype=torch.int64)
        text_list.append(processed_text)
    return torch.tensor(label_list, dtype=torch.int64), pad_sequence(
        text_list, padding_value=tokenizer.token_to_id("<pad>"), batch_first=True
    )


def get_dataloaders(batch_size, tokenizer):
    train_iter, test_iter = IMDB(split="train"), IMDB(split="test")
    train_dataloader = DataLoader(
        list(train_iter),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, tokenizer),
    )
    test_dataloader = DataLoader(
        list(test_iter),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_batch(x, tokenizer),
    )
    return train_dataloader, test_dataloader
