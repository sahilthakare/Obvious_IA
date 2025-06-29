# train.py

import torch
import torch.nn as nn
from tqdm import tqdm
from utils import calculate_accuracy, save_model

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, callback):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = 100 * correct / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        print(f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}")

        callback.check(epoch, val_loss, model, train_acc)
        if callback.stop_training:
            break

    print("Training complete.")
    save_model(model)
