import datetime

import torch
from torch import nn, optim

import wandb
from logger import setup_logger
from transformer.wikitext_pretrain.imdb_dataset import get_vocab as get_imdb_vocab, \
    get_dataloaders as get_imdb_dataloaders
from transformer.wikitext_pretrain.transformer_finetune_train_evaluate import train, evaluate
from transformer.wikitext_pretrain.transformer_model import CustomTransformerModel
from transformer.wikitext_pretrain.wikitext_dataset import get_vocab

run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_transformer_finetune_imdb"

# Toggle WandB
use_wandb = False

if use_wandb:
    # Initialize WandB
    wandb.login(key="5fda0926085bc8963be5e43c4e501d992e35abe8")
    wandb.init(project="model-comparison", name=run_name)

# Setup logging
logger = setup_logger(__name__)

# Hyperparameters
batch_size = 8
embed_dim = 64
num_heads = 1
num_layers = 2
ff_hidden_dim = 128
n_epochs = 1
learning_rate = 0.001

# Load vocab and data loaders for IMDB
vocab = get_vocab()
imdb_vocab = get_imdb_vocab()
train_dataloader, test_dataloader = get_imdb_dataloaders(batch_size, imdb_vocab)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
model = CustomTransformerModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_hidden_dim=ff_hidden_dim,
    finetune=True
).to(device)

# Load the pretrained weights
checkpoint_path = "transformer_wikitext_pretrained.pth"
model.load_state_dict(torch.load(checkpoint_path))
logger.info(f"Checkpoint loaded from {checkpoint_path}")

# Replace the final layer
model.fc_out = nn.Linear(embed_dim, 1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

logger.info(f"parameter_cnt: {model.count_parameters}")

if use_wandb:
    # Log hyperparameters and model
    wandb.config.update(
        {
            "run_name": run_name,
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_hidden_dim": ff_hidden_dim,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "model": "CustomTransformerModel",
            "parameter_cnt": model.count_parameters,
        }
    )

# Fine-tuning loop
logger.info("Starting fine-tuning...")
for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch, logger, use_wandb)
    test_loss = evaluate(model, test_dataloader, criterion, device, logger, use_wandb)
    logger.info(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    if use_wandb:
        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })

logger.info("Fine-tuning completed.")
if use_wandb:
    wandb.finish()