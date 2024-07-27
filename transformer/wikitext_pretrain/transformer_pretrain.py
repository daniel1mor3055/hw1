import datetime

import torch
import wandb
from torch import nn, optim

from logger import setup_logger
from transformer.wikitext_pretrain.transformer_model import CustomTransformerModel
from transformer.wikitext_pretrain.transformer_pretrain_train_evaluate import train, evaluate
from project_datasets.wikitext_dataset import get_tokenizer_and_vocab, get_dataloaders

run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_transformer_wikitext_pretrain"

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
embed_dim = 32
num_heads = 2
num_layers = 4
ff_hidden_dim = 128
n_epochs = 1
learning_rate = 0.001

# Load vocab and data loaders
tokenizer = get_tokenizer_and_vocab()
train_dataloader, test_dataloader = get_dataloaders(batch_size, tokenizer)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(tokenizer.get_vocab())
model = CustomTransformerModel(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_hidden_dim=ff_hidden_dim
).to(device)

# TODO - creates dummy lstm_wikitext_pretrained.pth, remove in real training
# Save the model state dict
checkpoint_path = "transformer_wikitext_pretrained.pthu"
torch.save(model.state_dict(), checkpoint_path)
print(f"Dummy checkpoint saved at {checkpoint_path}")
# TODO - dummy ends here

criterion = nn.CrossEntropyLoss()
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

# Save the model checkpoint
checkpoint_path = "transformer_wikitext_pretrained.pth"
torch.save(model.state_dict(), checkpoint_path)
logger.info(f"Checkpoint saved at {checkpoint_path}")

logger.info("Training completed.")
if use_wandb:
    wandb.finish()
