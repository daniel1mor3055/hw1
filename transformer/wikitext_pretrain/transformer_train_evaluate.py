import torch

import wandb


def train(model, dataloader, criterion, optimizer, device, epoch, logger, use_wandb):
    model.train()
    total_loss = 0
    for batch_idx, texts in enumerate(dataloader):
        texts = texts.to(device)
        optimizer.zero_grad()

        input_texts = texts[:, :-1]
        target_texts = texts[:, 1:]

        output = model(input_texts)
        loss = criterion(output.reshape(-1, output.size(-1)), target_texts.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(
                f"Train Epoch: {epoch + 1} [{batch_idx * len(texts)}/{len(dataloader.dataset)} "
                f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            if use_wandb:
                wandb.log({"train_batch_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    logger.info(f"====> Epoch: {epoch + 1} Average loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, criterion, device, logger, use_wandb):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, texts in enumerate(dataloader):
            texts = texts.to(device)

            # Shift the target texts by one time step for the decoder input
            input_texts = texts[:, :-1]
            target_texts = texts[:, 1:]

            output = model(input_texts)
            loss = criterion(output.view(-1, output.size(-1)), target_texts.view(-1))
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Eval Batch: {batch_idx + 1}/{len(dataloader)}\tLoss: {loss.item():.6f}"
                )
            if use_wandb:
                wandb.log({"eval_batch_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    logger.info(f"====> Test set loss: {avg_loss:.4f}")
    return avg_loss
