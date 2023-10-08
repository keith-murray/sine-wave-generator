import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.task import PatternDataset

def train_epoch(model, loss_function, train_loader, optimizer):
    model.train()
    for data in train_loader:
        inputs, labels = data[0], data[1]
        outputs, _ = model(inputs, model.init_hidden(batch_size))
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

def test_epoch(model, loss_function, test_loader):
    model.eval()
    losses = []
    for data in test_loader:
        inputs, labels = data[0], data[1]
        outputs, _ = model(inputs, model.init_hidden(batch_size))
        loss = loss_function(outputs, labels)
        losses.append(loss.item())
    return sum(losses)/len(losses)

def train_model(model, loss_function, train_loader, test_loader, optimizer, epochs, save_dir):
    train_losses = []
    test_losses = []
    best_test_loss = float('inf') # Initialize with a large number

    for epoch in range(epochs):
        train_epoch(model, loss_function, train_loader, optimizer)
        avg_train_loss = test_epoch(model, loss_function, train_loader)
        avg_test_loss = test_epoch(model, loss_function, test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Save model if it has the lowest test loss so far
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = os.path.join(save_dir, f'best_model_freqs_{len(train_loader.dataset)}.pth')
            torch.save(model.state_dict(), save_path)

    return train_losses, test_losses

def curriculum_train_model(model, freqs, time, epochs, save_dir):
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = nn.MSELoss()

    test_data = PatternDataset(freqs, time)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    all_train_losses = []
    all_test_losses = []

    for i in range(len(freqs)):
        print(f"Training with frequencies up to: {freqs[:i+1]}")
        
        # Adjust the training dataset to include more frequencies
        train_data = PatternDataset(freqs[:i+1], time)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        # Train the model
        train_losses, test_losses = train_model(
            model, 
            loss_function, 
            train_loader, 
            test_loader, 
            optimizer, 
            epochs,
            save_dir
        )
        
        all_train_losses.extend(train_losses)
        all_test_losses.extend(test_losses)

    return all_train_losses, all_test_losses