import torch


def train_one_epoch(model, dataloader, optimizer, criterion, epoch, model_name, logger, device):
    model.train()
    total_loss = 0
    dataloader.sampler.set_epoch(epoch)  # Move this line outside the loop
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the correct device
        if model_name == 'Transformer':
            outputs = model(inputs, targets)
        elif model_name == 'LSTM':
            outputs = model(inputs)
        else:
            logger.error(f"Undefined model: {model_name}")
            raise BaseException
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    logger.info(f'|TRAIN|: epoch: {epoch + 1}, total_loss: {total_loss}')
    return total_loss 


@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, epoch, model_name, logger, device):
    model.eval()
    total_loss = 0
    dataloader.sampler.set_epoch(epoch)  # Move this line outside the loop
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the correct device
        if model_name == 'Transformer':
            outputs = model(inputs, targets)
        elif model_name == 'LSTM':
            outputs = model(inputs)
        else:
            logger.error(f"Undefined model: {model_name}")
            exit()
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    logger.info(f'|VALID|: epoch: {epoch + 1}, total_loss: {total_loss}')
    return total_loss
