import json
import os
import sys
import torch
import numpy as np

from sinusoid.model import VanillaRNN
from sinusoid.training import curriculum_train_model
from sinusoid.analysis import plot_analyses

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Check for the necessary command-line arguments
if len(sys.argv) < 2:
    print("Usage: python train.py <path_to_params.json>")
    sys.exit(1)

task_id = sys.argv[1] # Here is the $LLSUB_RANK slurm argument
experiment_folder = sys.argv[2] # Here is a non-slurm argument, this folder is the same across the entire job
task_folder = os.path.join(experiment_folder, f"task_{task_id}")

json_path = os.path.join(task_folder, "params.json")
with open(json_path, 'r') as f:
    json_params = json.load(f)

# Extract the parameters
epochs = json_params.get("epochs", 25000)
freqs = json_params.get("freqs", [0.6,0.5,0.4,0.3,0.2,0.1])
time = json_params.get("time", 100)
seed = json_params.get("seed", 0)

# Set the seed for reproducibility
set_seed(seed)

# Initialize model
model = VanillaRNN(1, 1, 5)

# Train model
train_losses, test_losses = curriculum_train_model(model, freqs, time, epochs, task_folder)

# Save the results and model parameters
results = {
    "train_losses": train_losses,
    "test_losses": test_losses
}

# Save results
with open(os.path.join(task_folder, "results.json"), "w") as f:
    json.dump(results, f)
    
plot_analyses(task_folder, model)

print("Training complete. Results and model parameters saved.")
