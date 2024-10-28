import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models.MLP import MLPModel
from models.UNet1D import UNet1D


# Preprocessing is done

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def trainAndEval(features_df, labels):
    # KFold Cross Validation
    kf = KFold(n_splits=8, shuffle=True)
    fold = 1 # To count the fold number

    # Store metrics for each fold
    fold_metrics = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'val_auroc': []}
    all_preds = []
    all_true = []

    for train_index, val_index in kf.split(features_df):
        # Split the dataset into train and validation sets
        X_train, X_val = features_df.values[train_index], features_df.values[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        # Rescale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Convert it into tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        print("Len check: ", len(y_train_tensor), len(y_val_tensor))
        # Create dataloaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

        # Get input and output dimensions
        input_dim = X_train.shape[1]
        output_dim = 2

        # # Instantiate model
        # model = MLPModel(input_dim, output_dim).to(device)
        # use_unet = False
        # # a simple feedforward neural network


        mapping_MLP = MLPModel(input_dim, 1024)
        uNet = UNet1D(1, output_dim).to(device)
        classifyHead_MLP = MLPModel(2048, 1)
        model = nn.Sequential(mapping_MLP, uNet, classifyHead_MLP).to(device)
        use_unet = True
        # Use UNet1D as the model

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 50
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1) if use_unet else inputs
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[1])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                _, tar_labels = torch.max(targets.data, 1)
                total_train += tar_labels.size(0)
                correct_train += (predicted == tar_labels).sum().item()
            train_accuracy = 100 * correct_train / total_train

        fold_metrics['train_loss'].append(running_loss / len(train_loader))
        fold_metrics['train_accuracy'].append(train_accuracy)
        print(
            f'Fold [{fold}], Epoch [{epochs}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1) if use_unet else inputs
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[1])
                val_loss += loss.item()

                # Calculate validation accuracy
                # print("Shape check; ", outputs.shape, targets.shape)
                _, predicted = torch.max(outputs.data, 1)
                _, tar_eval_labels = torch.max(targets.data, 1)
                total_val += tar_eval_labels.size(0)
                correct_val += (predicted == tar_eval_labels).sum().item()

                # Collect predictions and true labels for AUROC and confusion matrix
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(tar_eval_labels.cpu().numpy())

        val_accuracy = 100 * correct_val / total_val
        all_targets = np.concatenate(all_targets).ravel()
        all_outputs = np.concatenate(all_outputs).ravel()
        # print("All targets shape: ", all_targets.shape, "All outputs shape: ", all_outputs.shape)
        # print("Their values:" , all_targets, all_outputs)
        val_auroc = roc_auc_score(all_targets, all_outputs)

        fold_metrics['val_loss'].append(val_loss / len(val_loader))
        fold_metrics['val_accuracy'].append(val_accuracy)
        fold_metrics['val_auroc'].append(val_auroc)

        print(
            f'Fold [{fold}], Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation AUROC: {val_auroc:.4f}')

        fold += 1

    # Calculate average metrics
    avg_train_loss = np.mean(fold_metrics['train_loss'])
    avg_train_accuracy = np.mean(fold_metrics['train_accuracy'])
    avg_val_loss = np.mean(fold_metrics['val_loss'])
    avg_val_accuracy = np.mean(fold_metrics['val_accuracy'])
    avg_val_auroc = np.mean(fold_metrics['val_auroc'])

    print("K-Fold Cross Validation complete.")
    print(f'Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_accuracy:.2f}%')
    print(
        f'Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_accuracy:.2f}%, Average Validation AUROC: {avg_val_auroc:.4f}')

    # Compute and visualize confusion matrix
    conf_matrix = confusion_matrix(all_true, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # return key socres
    return avg_val_accuracy, avg_val_auroc

if __name__ == '__main__':
    # run the training and evaluation for n times
    n = 1
    # record the key scores and then calculate the average and std
    avg_val_accuracy_list = []
    avg_val_auroc_list = []
    for i in range(n):
        print(f"Training and evaluating model for run {i+1}")
        avg_val_accuracy, avg_val_auroc = trainAndEval(features_df, labels)
        avg_val_accuracy_list.append(avg_val_accuracy)
        avg_val_auroc_list.append(avg_val_auroc)
        print(f"Training and evaluation for run {i+1} complete.\n")

    # calculate the average and std of the key scores
    avg_val_accuracy = np.mean(avg_val_accuracy_list)
    avg_val_auroc = np.mean(avg_val_auroc_list)
    std_val_accuracy = np.std(avg_val_accuracy_list)
    std_val_auroc = np.std(avg_val_auroc_list)
    print(f"Training and evaluation for {n} runs complete.")
    print("\n\n\nKey scores list: ", avg_val_accuracy_list, avg_val_auroc_list)
    print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%, Average Validation AUROC: {avg_val_auroc:.4f}")
    print(f"Standard Deviation of Validation Accuracy: {std_val_accuracy:.4f}, Standard Deviation of Validation AUROC: {std_val_auroc:.4f}")
