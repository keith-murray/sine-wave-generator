import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sinusoid.task import PatternDataset

def plot_loss(task_folder):
    # Load results from the task folder
    with open(os.path.join(task_folder, "results.json"), "r") as f:
        results = json.load(f)
    
    train_losses = results["train_losses"]
    test_losses = results["test_losses"]
    
    # Find the epoch with the lowest test loss
    min_loss_epoch = test_losses.index(min(test_losses))
    
    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(test_losses, label="Test Loss", color='green')
    plt.axvline(min_loss_epoch, color='red', linestyle='--')
    plt.scatter(min_loss_epoch, test_losses[min_loss_epoch], color='red') # highlight the point
    plt.text(
        min_loss_epoch, 
        test_losses[min_loss_epoch], 
        f'Min Loss at Epoch {min_loss_epoch}', 
        verticalalignment='bottom'
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(task_folder, "loss_plot.png"))
    plt.show()

def return_params_times(task_folder):
    json_path = os.path.join(task_folder, "params.json")
    with open(json_path, 'r') as f:
        json_params = json.load(f)
    time = json_params.get("time", 100)
    return time
    
def create_summary_graph(task_folder, model):
    # Load frequencies from the task's JSON file
    json_path = os.path.join(task_folder, "params.json")
    with open(json_path, 'r') as f:
        params = json.load(f)
    freqs = params.get("freqs")
    time = params.get("time", 100)

    # Create a 6x6 grid of subplots
    fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(15, 15), sharex=True, sharey=True)

    for model_indx in range(6):
        # Load model parameters
        model_path = os.path.join(task_folder, f"best_model_freqs_{model_indx+1}.pth")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        for freq_indx, freq in enumerate(freqs):
            # Prepare DataLoader for the specific frequency
            test_data = PatternDataset([freq], time)
            test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
            
            inputs, labels = next(iter(test_loader))
            
            outputs, _ = model(inputs, model.init_hidden(1))

            ax = axs[model_indx][freq_indx]
            ax.plot(labels[0].numpy(), label="Target")
            ax.plot(outputs[0].detach().numpy(), label="Output")
            
            # Set titles and legends for the first row and first column
            if freq_indx == 0:
                ax.set_ylabel(f"Model {model_indx+1}")
            if model_indx == 0:
                ax.set_title(f"Freq {freq}")

    plt.tight_layout()
    plt.savefig(os.path.join(task_folder, "summary_plot.png"))
    plt.show()

def determine_freqs_index_from_lowest_loss(task_folder):
    # Load results and params from their respective JSON files
    with open(os.path.join(task_folder, "results.json"), "r") as f:
        results = json.load(f)
    with open(os.path.join(task_folder, "params.json"), "r") as f:
        params = json.load(f)

    test_losses = results["test_losses"]
    epochs_per_phase = params["epochs"]

    # Find the epoch with the minimum test loss
    min_loss_epoch = test_losses.index(min(test_losses))
    
    # Determine the frequency set index
    freqs_index = (min_loss_epoch // epochs_per_phase) + 1
    
    return freqs_index

def pca_plot(task_folder, model):
    # Load the model parameters with the lowest test loss
    freqs_index = determine_freqs_index_from_lowest_loss(task_folder)
    model_path = os.path.join(task_folder, f"best_model_freqs_{freqs_index}.pth")  # Adjust this if necessary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Set up the dataset
    frequencies = np.linspace(0.1, 0.6, 51)
    time = return_params_times(task_folder)
    dataset = PatternDataset(frequencies, time)
    dataloader = DataLoader(dataset, batch_size=len(frequencies))
    assert len(list(dataloader)) == 1, "Length of dataset is not 1"

    # Get the RNN outputs for each example in the dataset
    with torch.no_grad():
        X, y = list(dataloader)[0]
        _, outputs = model(X, model.init_hidden(len(frequencies)))
        outputs = outputs.numpy()

    # Reshape the outputs and apply PCA
    outputs_3d = np.reshape(outputs, (len(frequencies), time, -1))
    pca = PCA(n_components=3)
    outputs_3d_pca = pca.fit_transform(outputs_3d.reshape(-1, outputs_3d.shape[-1]))
    outputs_3d_pca = np.reshape(outputs_3d_pca, (len(frequencies), time, 3))

    # Plot the 3D PCA projection
    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=min(frequencies), vmax=max(frequencies))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(frequencies)):
        ax.plot(
            outputs_3d_pca[i, :, 0],
            outputs_3d_pca[i, :, 1],
            outputs_3d_pca[i, :, 2],
            color=cmap(norm(frequencies[i]))
        )
    ax.set_xlabel('PCA dimension 1')
    ax.set_ylabel('PCA dimension 2')
    ax.set_zlabel('PCA dimension 3')
    ax.set_title('PCA of the hidden neuron activations')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=frequencies.min(), vmax=frequencies.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.ax.set_title('Frequency')
    plt.savefig(os.path.join(task_folder, "pca_plot.png"))
    plt.show()
    
def plot_analyses(task_folder, model):
    plot_loss(task_folder)
    create_summary_graph(task_folder, model)
    pca_plot(task_folder, model)
    
    print("Analysis plots complete.")
