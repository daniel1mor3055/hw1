import torch
import wandb
from torch import nn, optim

from dataset import get_vocab, get_dataloaders
from logger import setup_logger
from lstm_model import CustomLSTMModel
from train_evaluate import train, evaluate

# Initialize WandB
wandb.init(project="model-comparison")

# Setup logging
logger = setup_logger(__name__)

# Hyperparameters
batch_size = 8
embed_dim = 64
hidden_dim = 128
output_dim = 1
n_epochs = 5
learning_rate = 0.001

# Load vocab and data loaders
vocab = get_vocab()
train_dataloader, test_dataloader = get_dataloaders(batch_size, vocab)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
model = CustomLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Log hyperparameters and model
wandb.config.update({
    "batch_size": batch_size,
    "embed_dim": embed_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
    "n_epochs": n_epochs,
    "learning_rate": learning_rate,
    "model": "CustomLSTMModel"
})

# Training loop
logger.info("Starting training...")
for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch, logger)
    test_loss = evaluate(model, test_dataloader, criterion, device, logger)
    logger.info(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss
    })

logger.info("Training completed.")
wandb.finish()
