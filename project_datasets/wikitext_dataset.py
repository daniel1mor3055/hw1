import os
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class WikiTextDataset(Dataset):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(script_dir, "../saved")

    def __init__(self, split, tokenizer=None, tokenizer_name="bert-base-uncased"):
        self.dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split, cache_dir=WikiTextDataset.cache_dir
        ).filter(lambda x: x["text"].strip() != "")

        if tokenizer is None:
            # Load pre-trained BERT tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name,cache_dir=WikiTextDataset.cache_dir)
            logger.info(f"Pre-trained BERT tokenizer loaded: {tokenizer_name}")
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]["text"]

    def text_pipeline(self, text):
        # Encode text and return token ids
        return self.tokenizer.encode(
            text, truncation=True, padding=False, return_tensors="pt"
        ).squeeze(
            0
        )  # Remove the batch dimension

    def collate_batch(self, batch):
        text_list = [self.text_pipeline(text) for text in batch]
        # Padding is handled by pad_sequence
        padded_text_list = pad_sequence(
            text_list, padding_value=self.tokenizer.pad_token_id, batch_first=True
        )
        # Dummy labels for wikitext, as it's often used for language modeling
        return torch.zeros(len(text_list), dtype=torch.int64), padded_text_list

    def get_dataloaders(self, batch_size):
        train_dataset = WikiTextDataset(split="train", tokenizer=self.tokenizer)
        test_dataset = WikiTextDataset(split="test", tokenizer=self.tokenizer)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        return train_dataloader, test_dataloader
