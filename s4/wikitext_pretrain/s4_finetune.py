import datetime

import torch
import wandb
from torch import nn, optim

from logger import setup_logger
from s4.wikitext_pretrain.imdb_dataset import get_dataloaders as get_imdb_dataloaders
from s4.wikitext_pretrain.s4_finetune_train_evaluate import train, evaluate
from s4.wikitext_pretrain.s4_model import S4Model
from s4.wikitext_pretrain.wikitext_dataset import get_tokenizer_and_vocab

run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_s4_wikitext_pretrain_finetune_imdb"

# Toggle WandB
use_wandb = False

if use_wandb:
    # Initialize WandB
    wandb.login(key="5fda0926085bc8963be5e43c4e501d992e35abe8")
    wandb.init(project="model-comparison", name=run_name)

# Setup logging
logger = setup_logger(__name__)

# Hyperparameters
batch_size = 32
embed_dim = 24
hidden_dim = 128
num_layers = 2
n_epochs = 2
learning_rate = 0.001

# Load vocab and data loaders for IMDB
wikitext_tokenizer = get_tokenizer_and_vocab()
train_dataloader, test_dataloader = get_imdb_dataloaders(batch_size, wikitext_tokenizer)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(wikitext_tokenizer.get_vocab())
model = S4Model(
    d_input=embed_dim,
    vocab_size=vocab_size,
    d_model=hidden_dim,
    n_layers=num_layers,
    dropout=0.1,
    finetune=True
).to(device)

# Load the pretrained weights
checkpoint_path = "s4_wikitext_pretrained.pth"
model.load_state_dict(torch.load(checkpoint_path))
logger.info(f"Checkpoint loaded from {checkpoint_path}")

# Replace the final layer
model.decoder = nn.Linear(hidden_dim, 1).to(device)

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
            "hidden_dim": hidden_dim,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "model": "S4Model",
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
