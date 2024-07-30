import os
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from logger import setup_logger

# Initialize BPE tokenizer

logger = setup_logger(__name__)


class WikiTextDataset(Dataset):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    datasets_dir = os.path.join(script_dir, "../saved_datasets")

    def __init__(self, split, tokenizer=None, tokenizer_file="wikitext_tokenizer.json"):
        self.dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split, cache_dir=WikiTextDataset.datasets_dir
        ).filter(lambda x: x["text"].strip() != "")

        self.tokenizer_file = tokenizer_file
        if tokenizer is None:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            self.trainer = trainers.BpeTrainer(
                vocab_size=10000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
            )
            self.get_tokenizer_and_vocab()
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"]

    def yield_texts(self, data_iter):
        for text in data_iter:
            yield text

    def get_tokenizer_and_vocab(self):
        # Check if the tokenizer file already exists
        if os.path.exists(self.tokenizer_file):
            logger.info("Tokenizer loaded from file.")
            self.tokenizer = Tokenizer.from_file(self.tokenizer_file)
        else:
            train_iter = load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", cache_dir=WikiTextDataset.datasets_dir
            ).filter(lambda x: x["text"].strip() != "")["text"]
            self.tokenizer.train_from_iterator(
                self.yield_texts(train_iter), self.trainer
            )

            # Save the tokenizer to a file
            self.tokenizer.save(self.tokenizer_file)
            logger.info("Tokenizer trained and saved to file.")

        return self.tokenizer

    def text_pipeline(self, text):
        return self.tokenizer.encode(text).ids

    def collate_batch(self, batch):
        text_list = []
        for _text in batch:
            processed_text = torch.tensor(
                self.text_pipeline(_text),
                dtype=torch.int64,
            )
            text_list.append(processed_text)
        # Dummy labels for wikitext
        return torch.zeros(len(text_list), dtype=torch.int64), pad_sequence(
            text_list,
            padding_value=self.tokenizer.token_to_id("<pad>"),
            batch_first=True,
        )

    def get_dataloaders(self, batch_size):
        train_dataset = WikiTextDataset(
            split="train", tokenizer=self.tokenizer, tokenizer_file=self.tokenizer_file
        )
        test_dataset = WikiTextDataset(
            split="test", tokenizer=self.tokenizer, tokenizer_file=self.tokenizer_file
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: self.collate_batch(x),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: self.collate_batch(x),
        )
        return train_dataloader, test_dataloader
