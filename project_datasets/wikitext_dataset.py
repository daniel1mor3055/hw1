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
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=10000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
    )

    def __init__(self, split, tokenizer):
        self.dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        ).filter(lambda x: x["text"].strip() != "")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"]

    @staticmethod
    def yield_texts(data_iter):
        for text in data_iter:
            yield text

    @staticmethod
    def get_tokenizer_and_vocab():
        tokenizer_file = "wikitext_tokenizer.json"

        # Check if the tokenizer file already exists
        if os.path.exists(tokenizer_file):
            logger.info("Tokenizer loaded from file.")
            WikiTextDataset.tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            train_iter = load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split="train"
            ).filter(lambda x: x["text"].strip() != "")["text"]
            WikiTextDataset.tokenizer.train_from_iterator(
                WikiTextDataset.yield_texts(train_iter), WikiTextDataset.trainer
            )

            # Save the tokenizer to a file
            WikiTextDataset.tokenizer.save(tokenizer_file)
            logger.info("Tokenizer trained and saved to file.")

        return WikiTextDataset.tokenizer

    @staticmethod
    def text_pipeline(text):
        return WikiTextDataset.tokenizer.encode(text).ids

    @staticmethod
    def collate_batch(batch):
        text_list = []
        for _text in batch:
            processed_text = torch.tensor(
                WikiTextDataset.text_pipeline(_text),
                dtype=torch.int64,
            )
            text_list.append(processed_text)
        # Dummy labels for wikitext
        return torch.zeros(len(text_list), dtype=torch.int64), pad_sequence(
            text_list,
            padding_value=WikiTextDataset.tokenizer.token_to_id("<pad>"),
            batch_first=True,
        )

    @staticmethod
    def get_dataloaders(batch_size):
        tokenizer = WikiTextDataset.tokenizer
        train_dataset = WikiTextDataset(split="train", tokenizer=tokenizer)
        test_dataset = WikiTextDataset(split="test", tokenizer=tokenizer)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: WikiTextDataset.collate_batch(x),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: WikiTextDataset.collate_batch(x),
        )
        return train_dataloader, test_dataloader
