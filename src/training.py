import torch
import torch.nn as nn

def train_epoch(model, loss_function, train_loader, optimizer, device):
    model.train()
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs, _ = model(inputs, model.init_hidden(batch_size))
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

def test_epoch(model, loss_function, test_loader, device):
    model.eval()
    losses = []
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs, _ = model(inputs, model.init_hidden(batch_size))
        loss = loss_function(outputs, labels)
        losses.append(loss.item())
    return sum(losses)/len(losses)