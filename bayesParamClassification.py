"""
Use bayesian parameter search to find the best parameters for a classifier
This applies to all classifiers that have parameters that can be tuned
pCLE / RS / Multi-modal
"""
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
# Import the MLPMapper class
from models.MLP import MLPMapper

# Load the dataset
file_path = "data/Mammobot Raman PCA_13-12.xlsx"
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist.")
df = pd.read_excel(file_path, engine='openpyxl', index_col=None, header=None)

# Drop the columns without any values and transpose the dataframe, make samples in rows
df = df.dropna(axis=1).T
# Set the first row as the column names
df.columns = df.iloc[0]
# Delete the first row since it's used as column names
df_index = df[1:]
df_index = df_index.iloc[:, [0, 1] + list(range(3, df_index.shape[1]))]  # Remove the index column
df_filtered = df_index[(df_index['Group'] == 'Normal') | (df_index['Group'] == 'Tumour')]

# Not filtering samples but turning the 'Stained/unstained' column into a binary column (OneHotEncoding)
encoder = OneHotEncoder(sparse=False)
stained_encoded = encoder.fit_transform(df_filtered[['Stained/unstained']])
stained_encoded_df = pd.DataFrame(stained_encoded, columns=encoder.get_feature_names_out(['Stained/unstained']))
stained_encoded_df.reset_index(drop=True, inplace=True)

group_encoded = encoder.fit_transform(df_filtered[['Group']])
group_encoded_df = pd.DataFrame(group_encoded, columns=encoder.get_feature_names_out(['Group']))
group_encoded_df.reset_index(drop=True, inplace=True)

df_filtered.reset_index(drop=True, inplace=True)
features_df = pd.concat([stained_encoded_df, df_filtered.iloc[:, 4:]], axis=1)
labels = group_encoded_df.values  # define the labels

# Define the objective function
def objective(params):
    interim_dim = int(params['interim_dim'])
    kf = KFold(n_splits=8, shuffle=True)
    fold_accuracies = []

    for train_index, val_index in kf.split(features_df):
        X_train, X_val = features_df[train_index], features_df[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = MLPMapper(input_dim, interim_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 50
        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[1])
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.numpy())
                val_true.extend(torch.max(targets, 1)[1].numpy())

        fold_accuracies.append(accuracy_score(val_true, val_preds))

    return {'loss': -np.mean(fold_accuracies), 'status': STATUS_OK}

# Define the search space
space = {
    'interim_dim': hp.choice('interim_dim', range(50, 1000))
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best parameters found: ", best)