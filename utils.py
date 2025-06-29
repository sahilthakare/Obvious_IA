# utils.py
import torch

def calculate_accuracy(outputs, labels):
    return (outputs.argmax(1) == labels).float().mean().item()

def save_model(model, path='best_model.pth'):
    torch.save(model.state_dict(), path)
    print("Model saved.")

def load_model(model, path='best_model.pth'):
    model.load_state_dict(torch.load(path))
    print("Model loaded.")
    return model
