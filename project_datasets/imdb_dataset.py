import os
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB


class IMDBDataset:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, "../saved_datasets")

    def __init__(self, tokenizer=None, tokenizer_file="imdb_tokenizer.json"):
        if tokenizer is None:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            self.trainer = trainers.BpeTrainer(
                vocab_size=10000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
            )
            self.get_tokenizer_and_vocab()
        else:
            self.tokenizer = tokenizer
        self.tokenizer_file = tokenizer_file

    def yield_texts(self, data_iter):
        for _, text in data_iter:
            yield text

    def get_tokenizer_and_vocab(self):
        train_iter = IMDB(split="train", root=IMDBDataset.datasets_dir)
        self.tokenizer.train_from_iterator(self.yield_texts(train_iter), self.trainer)
        return self.tokenizer

    def text_pipeline(self, text):
        return self.tokenizer.encode(text).ids

    def label_pipeline(self, label):
        return label - 1

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
        return torch.tensor(label_list, dtype=torch.int64), pad_sequence(
            text_list,
            padding_value=self.tokenizer.token_to_id("<pad>"),
            batch_first=True,
        )

    def get_dataloaders(self, batch_size):
        train_iter, test_iter = IMDB(split="train", root=IMDBDataset.datasets_dir), IMDB(split="test",root=IMDBDataset.datasets_dir)
        train_dataloader = DataLoader(
            list(train_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: self.collate_batch(x),
        )
        test_dataloader = DataLoader(
            list(test_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: self.collate_batch(x),
        )
        return train_dataloader, test_dataloader
