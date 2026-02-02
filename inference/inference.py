import torch
import numpy as np
from model_definition import DermCNN

def evaluate_new_dataset(npz_path, model_path):
    """
    Loads a trained model and evaluates accuracy on a new dataset.
    Used during live on-campus evaluation.
    """
    model = DermCNN(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data = np.load(npz_path)
    images = torch.tensor(data["images"]).float().unsqueeze(1) / 255.0
    labels = torch.tensor(data["labels"]).long()

    with torch.no_grad():
        preds = model(images).argmax(dim=1)

    accuracy = (preds == labels).float().mean().item()
    return accuracy
