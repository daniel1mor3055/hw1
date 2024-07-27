import argparse
import datetime
import json

import torch
import wandb

from logger import setup_logger
from models.lstm_model import CustomLSTMModel
from models.s4_model import S4Model
from models.transformer_model import CustomTransformerModel
from utils import replace_final_layer

# Command line arguments
parser = argparse.ArgumentParser(description="Run different models")

# Load config
with open('config.json') as f:
    config = json.load(f)

parser.add_argument('--model', type=str, required=True, choices=list(config["models"].keys()), help="Model to use")
parser.add_argument('--run_type', type=str, required=True,
                    choices=["task", "lra_pretrain", "wikitext_pretrain", "task_finetune_lra_pretrain",
                             "task_finetune_wikitext_pretrain"], help="Run type")
parser.add_argument('--use_wandb', action='store_true')
args = parser.parse_args()

model_config = config["models"][args.model]
run_parameters = config["run_parameters"]

# Setup logging
logger = setup_logger(__name__)

run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}_{args.run_type}"

batch_size, n_epochs, learning_rate = run_parameters["batch_size"], run_parameters["n_epochs"], run_parameters[
    "learning_rate"]

# Load tokenizer and data loaders
if "wikitext" in args.run_type and "finetune" in args.run_type:
    from project_datasets.wikitext_dataset import get_tokenizer_and_vocab
    from project_datasets.imdb_dataset import get_dataloaders
elif "wikitext" in args.run_type:
    from project_datasets.wikitext_dataset import get_tokenizer_and_vocab, get_dataloaders
else:
    from project_datasets.imdb_dataset import get_tokenizer_and_vocab, get_dataloaders
tokenizer = get_tokenizer_and_vocab()
train_dataloader, test_dataloader = get_dataloaders(batch_size, tokenizer)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(tokenizer.get_vocab())
finetune = args.run_type in ["task", "task_finetune_lra_pretrain", "task_finetune_wikitext_pretrain"]

model_dic = {
    "lstm": CustomLSTMModel,
    "s4": S4Model,
    "transformer": CustomTransformerModel
}

if args.run_type == "task":
    config["models"][args.model]["output_dim"] = 1
else:
    config["models"][args.model]["output_dim"] = vocab_size

model = model_dic[args.model](vocab_size=vocab_size, **config["models"][args.model], finetune=finetune)

pretrained = "lra_pretrained" if "lra" in args.run_type else "wikitext_pretrained"
checkpoint_path = f"{args.model}_{pretrained}.pth"
if not finetune:
    # TODO - creates dummy lstm_wikitext_pretrained.pth, remove in real training
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Dummy checkpoint saved at {checkpoint_path}")
    # TODO - dummy ends here
elif args.run_type != "task":
    model.load_state_dict(torch.load(checkpoint_path))
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    replace_final_layer(model, config, args.model, device)

criterion = torch.nn.BCEWithLogitsLoss() if finetune else torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

logger.info(f"parameter_cnt: {model.count_parameters}")

if args.use_wandb:
    # Initialize WandB
    wandb.login(key="5fda0926085bc8963be5e43c4e501d992e35abe8")
    wandb.init(project="model-comparison", name=run_name)

    # Log hyperparameters and model
    wandb.config.update(
        {
            **{
                "run_name": run_name,
                "model": args.model,
                "parameter_cnt": model.count_parameters,
            },
            **config["models"][args.model],
            **config["run_parameters"]}
    )

# Training loop
logger.info("Starting training...")

if finetune:
    from train_evaluate.train_evaluate import train, evaluate
else:
    from train_evaluate.pretrain_train_evaluate import train, evaluate

for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device, epoch, logger, args.use_wandb)
    test_loss = evaluate(model, test_dataloader, criterion, device, logger, args.use_wandb)
    logger.info(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    if args.use_wandb:
        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss
        })

# Save the model checkpoint
checkpoint_path = f"{args.model}_{args.run_type}_checkpoint.pth"
torch.save(model.state_dict(), checkpoint_path)
logger.info(f"Checkpoint saved at {checkpoint_path}")

logger.info("Training completed.")

if args.use_wandb:
    wandb.finish()
