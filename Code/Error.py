import torch
import torch.optim as optim
import torch.nn as nn
import optuna
import numpy as np
from sklearn.model_selection import train_test_split

class HyperparameterTuner:
    def __init__(self, model_class, train_dataset, val_dataset, device, num_trials=50):
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.num_trials = num_trials
        
    def objective(self, trial):
        # Define search space
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"]) 
        num_epochs = trial.suggest_int("epochs", 5, 50)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = self.model_class().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def run_optimization(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.num_trials)
        return study.best_params

