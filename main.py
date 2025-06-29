# main.py
import torch
from data_loader import get_dataloaders
from model_architecture import get_model
from train import train_model
from evaluate import evaluate
from callbacks import CustomLRA


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    model = get_model(freeze=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    callback = CustomLRA(
        optimizer=optimizer,
        patience=2,
        stop_patience=3,
        factor=0.5,
        dwell=True,
        ask_epoch=5,
        threshold=80
    )

    train_model(model, train_loader, val_loader, optimizer, device, num_epochs=20, callback=callback)
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
