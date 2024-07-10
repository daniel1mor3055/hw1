import torch

import wandb


def train(model, dataloader, criterion, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0
    for batch_idx, (labels, texts) in enumerate(dataloader):
        labels, texts = labels.to(device), texts.to(device)
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(
                f"Train Epoch: {epoch + 1} [{batch_idx * len(labels)}/{len(dataloader.dataset)} "
                f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            # wandb.log({"train_batch_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    logger.info(f"====> Epoch: {epoch + 1} Average loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, criterion, device, logger):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (labels, texts) in enumerate(dataloader):
            labels, texts = labels.to(device), texts.to(device)
            output = model(texts)
            loss = criterion(output, labels.float())
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Eval Batch: {batch_idx + 1}/{len(dataloader)}\tLoss: {loss.item():.6f}"
                )
                # wandb.log({"eval_batch_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    logger.info(f"====> Test set loss: {avg_loss:.4f}")
    return avg_loss
