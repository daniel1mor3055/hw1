import datetime

import torch
import wandb
from torch import nn, optim

from logger import setup_logger
from project_datasets.imdb_dataset import get_tokenizer_and_vocab, get_dataloaders
from train_evaluate.train_evaluate import train, evaluate
from lstm.lra_pretrain.lstm_model import CustomLSTMModel

run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_lstm_lra_pretrain_finetune_imdb"

# Toggle WandB
use_wandb = False

if use_wandb:
    # Initialize WandB
    wandb.login(key="5fda0926085bc8963be5e43c4e501d992e35abe8")
    wandb.init(project="model-comparison", name=run_name)

# Setup logging
logger = setup_logger(__name__)

# Hyperparameters
batch_size = 1
embed_dim = 16
hidden_dim = 128
num_layers = 2
n_epochs = 2
learning_rate = 0.001

# Load tokenizer and data loaders
tokenizer = get_tokenizer_and_vocab()
train_dataloader, test_dataloader = get_dataloaders(batch_size, tokenizer)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(tokenizer.get_vocab())

model = CustomLSTMModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    finetune=True
).to(device)

# Load the pretrained weights
checkpoint_path = "lstm_lra_pretrained.pth"
model.load_state_dict(torch.load(checkpoint_path))
logger.info(f"Checkpoint loaded from {checkpoint_path}")

# Replace the final layer
output_dim = 1
model.Wy = nn.Parameter(torch.empty(output_dim, hidden_dim).to(device))
model.by = nn.Parameter(torch.zeros(output_dim, 1).to(device))
nn.init.xavier_uniform_(model.Wy)

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
            "output_dim": output_dim,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "model": "CustomLSTMModel",
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
