import numpy as np
import pandas as pd
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataset_for_single_feature_prediction(data, input_steps, output_steps, target_feature_idx):
    """
    Convert a series of data into a dataset where the input is multivariate time series data
    and the output is a single feature across multiple future time steps.

    :param data: A 2D numpy array or a pandas DataFrame containing the time series data.
    :param input_steps: The number of time steps to use for the input features.
    :param output_steps: The number of future time steps of the target feature to predict.
    :param target_feature_idx: The column index of the target feature to be predicted.
    :return: A tuple of input-output pairs for training the model.
    """
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:(i + input_steps), :])
        y.append(data[(i + input_steps):(i + input_steps + output_steps), :])
    return np.array(X), np.array(y)


def split_dataset(X, y, train_ratio, val_ratio):
    """
    Split the dataset into training, validation, and testing sets.

    :param X: The input features.
    :param y: The target labels.
    :param train_ratio: The proportion of the dataset to include in the train split.
    :param val_ratio: The proportion of the dataset to include in the validation split.
    :return: A tuple containing train, validation, and test datasets.
    """
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_dataloader(train_data_path, test_data_path, val_data_path, input_steps=96, output_steps=96, target_feature_idx=9):
    # Assuming `data` is your DataFrame or 2D numpy array
    """
    input_steps  # Number of past time steps for input
    output_steps # Number of future time steps to predict for the target feature
    target_feature_idx # Index of the target feature in the data array
    """
    
    # Convert the DataFrame to a numpy array if necessary
    train_data = read_csv(train_data_path, header=0)
    test_data = read_csv(test_data_path, header=0)
    val_data = read_csv(val_data_path, header=0)
    
    train_data_array = train_data.values if isinstance(train_data, pd.DataFrame) else train_data
    test_data_array = test_data.values if isinstance(test_data, pd.DataFrame) else test_data
    val_data_array = val_data.values if isinstance(val_data, pd.DataFrame) else val_data
    X_train, y_train = create_dataset_for_single_feature_prediction(train_data_array, input_steps, output_steps, target_feature_idx)
    X_test, y_test = create_dataset_for_single_feature_prediction(test_data_array, input_steps, output_steps, target_feature_idx)
    X_val, y_val = create_dataset_for_single_feature_prediction(val_data_array, input_steps, output_steps, target_feature_idx)
    # Split the dataset into training (60%), validation (20%), and testing (20%) sets
    # X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, train_ratio=0.6, val_ratio=0.2)
    
    print("Input shape:", X_train.shape)
    print("Output shape:", y_train.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test