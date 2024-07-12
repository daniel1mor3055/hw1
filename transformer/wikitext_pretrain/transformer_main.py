import datetime

import torch
import wandb
from torch import nn, optim

from dataset import get_vocab, get_dataloaders
from logger import setup_logger
from transformer.directly_on_task.transformer_train_evaluate import train, evaluate
from transformer.directly_on_task.transformer_model import CustomTransformerModel

run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_transformer_lra_pretrain"

# Toggle WandB
use_wandb = True

if use_wandb:
    # Initialize WandB
    wandb.login(key="5fda0926085bc8963be5e43c4e501d992e35abe8")
    wandb.init(project="model-comparison", name=run_name)

# Setup logging
logger = setup_logger(__name__)

# Hyperparameters
batch_size = 8
embed_dim = 64
num_heads = 2
num_layers = 16
ff_hidden_dim = 768
output_dim = 1
n_epochs = 2
learning_rate = 0.001

# Load vocab and data loaders
vocab = get_vocab()
train_dataloader, test_dataloader = get_dataloaders(batch_size, vocab)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
model = CustomTransformerModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_hidden_dim=ff_hidden_dim,
    output_dim=output_dim
).to(device)

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
            "output_dim": output_dim,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "model": "CustomTransformerModel",
            "parameter_cnt": model.count_parameters,
        }
    )

# Training loop
logger.info("Starting training...")
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

logger.info("Training completed.")
if use_wandb:
    wandb.finish()