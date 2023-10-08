import json
import os
import sys
import torch
import numpy as np

from src.model import VanillaRNN
from src.training import curriculum_train_model

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

# Read in the JSON parameters
json_path = sys.argv[1]
with open(json_path, 'r') as f:
    params = json.load(f)

# Extract the parameters
epochs = params.get("epochs")
freqs = params.get("freqs")
time = params.get("time")
seed = params.get("seed", None)

# Set the seed for reproducibility
if seed is not None:
    set_seed(seed)

# Initialize model
model = VanillaRNN(1, 1, 5)

# Train model
train_losses, test_losses = curriculum_train_model(model, freqs, time, epochs)

# Save the results and model parameters
results = {
    "train_losses": train_losses,
    "test_losses": test_losses
}

# Create a directory to save results if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save results
with open(os.path.join(output_dir, "results.json"), "w") as f:
    json.dump(results, f)

# Save model parameters
torch.save(model.state_dict(), os.path.join(output_dir, "model_params.pth"))

print("Training complete. Results and model parameters saved.")