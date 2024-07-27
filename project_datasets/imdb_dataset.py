import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB


class IMDBDataset:
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=10000, special_tokens=["<unk>", "<pad>", "<s>", "</s>"]
    )

    @staticmethod
    def yield_texts(data_iter):
        for _, text in data_iter:
            yield text

    @staticmethod
    def get_tokenizer_and_vocab():
        train_iter = IMDB(split="train")
        IMDBDataset.tokenizer.train_from_iterator(
            IMDBDataset.yield_texts(train_iter), IMDBDataset.trainer
        )
        return IMDBDataset.tokenizer

    @staticmethod
    def text_pipeline(text, tokenizer):
        return tokenizer.encode(text).ids

    @staticmethod
    def label_pipeline(label):
        return label - 1

    @staticmethod
    def collate_batch(batch, tokenizer):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(IMDBDataset.label_pipeline(_label))
            processed_text = torch.tensor(
                IMDBDataset.text_pipeline(_text, tokenizer), dtype=torch.int64
            )
            text_list.append(processed_text)
        return torch.tensor(label_list, dtype=torch.int64), pad_sequence(
            text_list, padding_value=tokenizer.token_to_id("<pad>"), batch_first=True
        )

    @staticmethod
    def get_dataloaders(batch_size):
        tokenizer = IMDBDataset.tokenizer
        train_iter, test_iter = IMDB(split="train"), IMDB(split="test")
        train_dataloader = DataLoader(
            list(train_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: IMDBDataset.collate_batch(x, tokenizer),
        )
        test_dataloader = DataLoader(
            list(test_iter),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: IMDBDataset.collate_batch(x, tokenizer),
        )
        return train_dataloader, test_dataloader
