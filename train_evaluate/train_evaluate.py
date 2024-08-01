import torch


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

        if batch_idx % 250 == 0:
            logger.info(
                f"Train Epoch: {epoch} [{batch_idx * len(labels)}/{len(dataloader.dataset)} "
                f"({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, criterion, device, logger):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (labels, texts) in enumerate(dataloader):
            labels, texts = labels.to(device), texts.to(device)
            output = model(texts)
            loss = criterion(output, labels.float())
            total_loss += loss.item()

            preds = (output > 0).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 250 == 0:
                logger.info(
                    f"Eval Batch: {batch_idx + 1}/{len(dataloader)}\tLoss: {loss.item():.6f}"
                )
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    logger.info(f"====> Test set loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
