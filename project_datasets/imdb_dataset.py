import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from transformers import BertTokenizer


class IMDBDataset:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(script_dir, "../saved")

    def __init__(self, tokenizer=None, tokenizer_name="bert-base-uncased"):
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(
                tokenizer_name, cache_dir=IMDBDataset.cache_dir
            )
        else:
            self.tokenizer = tokenizer

    def text_pipeline(self, text):
        # Encode the text using the BERT tokenizer
        return self.tokenizer.encode(
            text,
            truncation=True,
            padding="max_length",  # Optional: You can also handle padding in `collate_batch`
            max_length=512,  # Typical max length for BERT
            return_tensors="pt",
        ).squeeze(
            0
        )  # Remove batch dimension

    def label_pipeline(self, label):
        # In this dataset the labels are 1 for negative and 2 for positive
        return label - 1

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = self.text_pipeline(_text)
            text_list.append(processed_text)
        return torch.tensor(label_list, dtype=torch.int64), pad_sequence(
            text_list, padding_value=self.tokenizer.pad_token_id, batch_first=True
        )

    def get_dataloaders(self, batch_size):
        train_iter, test_iter = IMDB(split="train", root=IMDBDataset.cache_dir), IMDB(
            split="test", root=IMDBDataset.cache_dir
        )
        train_dataloader = DataLoader(
            list(train_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        test_dataloader = DataLoader(
            list(test_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        return train_dataloader, test_dataloader
