import torch
import numpy as np
from model_definition import DermCNN

def evaluate_new_dataset(npz_path, model_path):
    """
    Loads a dataset and trained model, returns accuracy.
    Used during live on-campus evaluation.
    """

    model = DermCNN(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data = np.load(npz_path)
    images = torch.tensor(data["images"]).float().unsqueeze(1) / 255.0
    labels = torch.tensor(data["labels"]).long()

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    accuracy = (predictions == labels).float().mean().item()
    return accuracy


