import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

from models.MLP import MLPModel, MLPMapper
from models.UNet1D import UNet1D

"""
To be executed in terminal with the following command:
python ramanClassification.py 
with --file_path (file path to Mammobot Raman PCA_13-12.xlsx) --num_runs (How many runs)
"""

# Use argsphrase to record configs
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", "-fp", default="data/Mammobot Raman PCA_13-12.xlsx", type=str)
parser.add_argument("--k_fold", "-kf", default=8, type=int)
parser.add_argument("--num_runs", "-nr", default=5, type=int)
parser.add_argument("--use_unet", "-u", default=False, type=bool)
args = parser.parse_args()


file_path = args.file_path
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist.")
df = pd.read_excel(file_path, engine='openpyxl', index_col=None, header=None)

# Drop the cloumns without any values and Transpose the dataframe, make samples in rows
df = df.dropna(axis=1).T
# Set the first row as the column names
df.columns = df.iloc[0]
# Delete the first row since it's used as column names
df_index = df[1:]

# df_index.set_index('Specimen label', inplace=True) Not using this, considering them as different scans
df_index = df_index.iloc[:, [0, 1] + list(range(3, df_index.shape[1]))]  # Remove the index column
df_filtered = df_index[(df_index['Group'] == 'Normal') | (df_index['Group'] == 'Tumour')]

# select unstained samples
# df_filtered = df_filtered[df_filtered['Stained/unstained'] == 'unstained']
# Drop the 'Stained/unstained' column
# df_filtered = df_filtered.drop(columns=['Stained/unstained'])

# or, select stained samples
# df_filtered = df_filtered[df_filtered['Stained/unstained'] == 'stained']
# df_filtered = df_filtered.drop(columns=['Stained/unstained'])

# Not filtering samples but turning the 'Stained/unstained' column into a binary column (OneHotEncoding)
encoder = OneHotEncoder(sparse=False)
stained_encoded = encoder.fit_transform(df_filtered[['Stained/unstained']])
stained_encoded_df = pd.DataFrame(stained_encoded, columns=encoder.get_feature_names_out(['Stained/unstained']))
stained_encoded_df.reset_index(drop=True, inplace=True)

group_encoded = encoder.fit_transform(df_filtered[['Group']])
group_encoded_df = pd.DataFrame(group_encoded, columns=encoder.get_feature_names_out(['Group']))
group_encoded_df.reset_index(drop=True, inplace=True)

df_filtered.reset_index(drop=True, inplace=True)

# Maybe not filtering the 'Spectrum number' column because it's just a repeated scanning with different staining settings
# spectrum_encoded = encoder.fit_transform(df_filtered[['Spectrum number']])
# spectrum_encoded_df = pd.DataFrame(spectrum_encoded, columns=encoder.get_feature_names_out(['Spectrum number']))
# features_df = pd.concat([stained_encoded_df, spectrum_encoded_df, df_filtered.iloc[:, 4:]], axis=1) if use spectrum number
# Else, just use the first 4 columns

features_df = pd.concat([stained_encoded_df, df_filtered.iloc[:, 4:]], axis=1)
labels = group_encoded_df.values  # define the labels
# Preprocessing is done

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')


def trainAndEval(features_df, labels):
    # KFold Cross Validation
    kf = KFold(n_splits=args.k_fold, shuffle=True)
    fold = 1 # To count the fold number

    # Store metrics for each fold
    fold_metrics = {'train_loss': [], 'train_accuracy': [], 'val_loss': [],
                    'val_accuracy': [], 'val_auroc': [], 'val_recall': [],
                    'val_f1': []}
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
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)

        # Get input and output dimensions
        input_dim = X_train.shape[1]
        output_dim = 2

        use_unet = False

        if use_unet:
            mapping_MLP = MLPMapper(input_dim, 256, 128)
            uNet = UNet1D(1, output_dim).to(device)
            classifyHead_MLP = MLPMapper(256, 128, output_dim)
            model = nn.Sequential(mapping_MLP, uNet, classifyHead_MLP).to(device)
            # Use UNet1D as the model

        else:
            # Instantiate model
            model = MLPModel(input_dim, output_dim).to(device)
            use_unet = False
            # Use a simple feedforward neural network
        print("Using U-Net for classification" if use_unet else "Using MLP for classification")

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 100 if use_unet else 50
        # print("Configs: ",
        #       '\nepochs: ', epochs,
        #       '\nuse UNet: ', use_unet,
        #       '\ndevice: ', device)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # print("Input shape: ", inputs.shape, "Target shape: ", targets.shape)
                inputs = inputs.unsqueeze(1) if use_unet else inputs
                optimizer.zero_grad()
                outputs = model(inputs)
                # print("Output shape: ", outputs.shape, "torch.max targets: ", torch.max(targets, 1)[1].shape)
                loss = criterion(outputs, torch.max(targets, 1)[1])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                _, tar_labels = torch.max(targets.data, 1)
                total_train += tar_labels.size(0)
                correct_train += (predicted == tar_labels).sum().item()
                if epoch % 20 == 0:
                    print('\r', "Epoch: ", epoch, "Loss: ", loss.item(), end='', flush=True)
            train_accuracy = correct_train / total_train

        fold_metrics['train_loss'].append(running_loss / len(train_loader))
        fold_metrics['train_accuracy'].append(train_accuracy)
        print(f'Fold [{fold}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}')

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
                print("VAL Input shape: ", inputs.shape,
                      "Output shape: ", outputs.shape)
                loss = criterion(outputs, torch.max(targets, 1)[1])
                val_loss += loss.item()

                # Calculate validation accuracy
                # print("Shape check; ", outputs.shape, targets.shape)
                _, predicted = torch.max(outputs.data, 1)
                _, tar_eval_labels = torch.max(targets.data, 1)
                print("Predicted shape: ", predicted.shape,
                      "labels shape: ", tar_eval_labels.shape)
                total_val += tar_eval_labels.size(0)
                correct_val += (predicted == tar_eval_labels).sum().item()

                # Collect predictions and true labels for AUROC and confusion matrix
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(tar_eval_labels.cpu().numpy())

        val_accuracy = correct_val / total_val
        all_targets = np.concatenate(all_targets).ravel()
        all_outputs = np.concatenate(all_outputs).ravel()
        # print("All targets shape: ", all_targets.shape, "All outputs shape: ", all_outputs.shape)
        val_auroc = roc_auc_score(all_targets, all_outputs)
        val_recall = recall_score(all_true, all_preds)
        val_f1 = f1_score(all_true, all_preds)

        fold_metrics['val_loss'].append(val_loss / len(val_loader))
        fold_metrics['val_accuracy'].append(val_accuracy)
        fold_metrics['val_auroc'].append(val_auroc)
        fold_metrics['val_recall'].append(val_recall)
        fold_metrics['val_f1'].append(val_f1)

        print(
            f'Fold [{fold}], Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}, Validation AUROC: {val_auroc:.4f}')

        fold += 1

    # Calculate average metrics
    avg_train_loss = np.mean(fold_metrics['train_loss'])
    avg_train_accuracy = np.mean(fold_metrics['train_accuracy'])
    avg_val_loss = np.mean(fold_metrics['val_loss'])
    avg_val_accuracy = np.mean(fold_metrics['val_accuracy'])
    avg_val_auroc = np.mean(fold_metrics['val_auroc'])
    avg_val_recall = np.mean(fold_metrics['val_recall'])
    avg_val_f1 = np.mean(fold_metrics['val_f1'])

    print("K-Fold Cross Validation complete.")
    print(f'Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_accuracy:.2f}')
    print(
        f'Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_accuracy:.2f}, Average Validation AUROC: {avg_val_auroc:.4f}')

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
    return avg_val_accuracy, avg_val_auroc, avg_val_recall, avg_val_f1

if __name__ == '__main__':

    # run the training and evaluation for n times
    n = args.num_runs
    # record the key scores and then calculate the average and std
    avg_val_accuracy_list = []
    avg_val_auroc_list = []
    avg_val_recall_list = []
    avg_val_f1_list = []
    for i in range(n):
        print(f"Training and evaluating model for run {i+1}")
        avg_val_accuracy, avg_val_auroc, avg_val_recall, avg_val_f1 = trainAndEval(features_df, labels)
        avg_val_accuracy_list.append(avg_val_accuracy)
        avg_val_auroc_list.append(avg_val_auroc)
        avg_val_recall_list.append(avg_val_recall)
        avg_val_f1_list.append(avg_val_f1)
        print(f"Training and evaluation for run {i+1} complete.\n")

    # calculate the average and std of the key scores
    avg_val_accuracy = np.mean(avg_val_accuracy_list)
    avg_val_auroc = np.mean(avg_val_auroc_list)
    avg_val_recall = np.mean(avg_val_recall_list)
    avg_val_f1 = np.mean(avg_val_f1_list)

    std_val_accuracy = np.std(avg_val_accuracy_list)
    std_val_auroc = np.std(avg_val_auroc_list)
    std_val_recall = np.std(avg_val_recall_list)
    std_val_f1 = np.std(avg_val_f1_list)
    print(f"Training and evaluation for {n} runs complete.")
    print("\n\n\nKey scores list: ", avg_val_accuracy_list, avg_val_auroc_list)
    print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}, "
          f"\nAverage Validation AUROC: {avg_val_auroc:.4f}",
          f"\nAverage Validation Recall: {avg_val_recall:.4f},"
          f"\nAverage Validation F1: {avg_val_f1:.4f}")

    print(f"Standard Deviation of Validation Accuracy: {std_val_accuracy:.4f}, "
          f"\nStandard Deviation of Validation AUROC: {std_val_auroc:.4f}"
          f"\nStandard Deviation of Validation Recall: {std_val_recall:.4f}"
          f"\nStandard Deviation of Validation F1: {std_val_f1:.4f}")
