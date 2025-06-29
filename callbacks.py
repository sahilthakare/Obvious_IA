# callbacks.py

import torch
import copy

class CustomLRA:
    def __init__(self, optimizer, patience, stop_patience, factor, dwell, ask_epoch, threshold):
        self.optimizer = optimizer
        self.patience = patience
        self.stop_patience = stop_patience
        self.factor = factor
        self.dwell = dwell
        self.ask_epoch = ask_epoch
        self.threshold = threshold

        self.best_loss = float('inf')
        self.best_model_wts = None
        self.bad_epoch_count = 0
        self.lr_reduce_count = 0
        self.stop_training = False

    def check(self, epoch, val_loss, model, train_acc):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.bad_epoch_count = 0
        else:
            self.bad_epoch_count += 1

        if self.bad_epoch_count >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= self.factor
                print(f"\nReducing LR from {old_lr:.6f} to {param_group['lr']:.6f}")
            self.bad_epoch_count = 0
            self.lr_reduce_count += 1

            if self.dwell:
                model.load_state_dict(self.best_model_wts)
                print("Restored best model weights.")

        if self.lr_reduce_count >= self.stop_patience:
            print("Early stopping triggered.")
            self.stop_training = True

        if epoch + 1 == self.ask_epoch:
            ans = input(f"Continue training after {epoch+1} epochs? (y/n): ")
            if ans.lower() != 'y':
                self.stop_training = True
