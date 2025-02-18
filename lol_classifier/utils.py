import os, shutil, glob
from typing import List
import pandas as pd
import numpy as np
from tabulate import tabulate
from constants import *

def select_channels(df: pd.DataFrame, channels: List[int] = [1,2,3,4,5,6,7,8]):
    selected_columns = []
    for c in channels:
        selected_columns.append(df.iloc[:, c-1::8])
    if len(selected_columns) == 0:
        raise ValueError('No columns selected')
    return pd.concat(selected_columns, axis=1)

def select_bands(df: pd.DataFrame, bands: slice = slice(16,56)):
    return df.iloc[:, bands]

def read_eeg_data(path) -> pd.DataFrame:
  df = pd.read_csv(path, low_memory=False, header=None)
  markers = df.iloc[:, -1].astype(str)
  df = select_bands(df)
  df = select_channels(df)
  df = pd.concat([df, markers], axis=1) # take only alpha, beta, gamma
  return df

def propagate_events(x, propagation: list):
  """"
  Propagate events over 3 seconds since event occurrence
  propagation: [marker_to_be_set, countdown]
  """
  if x != '0':
    propagation[1] = SAMPLING_RATE * 3 -1 # 3 seconds
    propagation[0] = x
  
  if x == '0' and propagation[1] > 0:
    propagation[1] -= 1
    x = propagation[0]
  return x

def clear_directories(dirs: list[str]):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

def split_and_save(df, filename):
  markers = df.iloc[:, -1]
  training = []
  testing = []
  # validation = []

  for idx, state in enumerate(STATES):
    df_subset = df[markers == state]
    if len(df_subset) == 0:
      continue

    df_subset.iloc[:, -1] = str(idx)
    training_size = int(TRAINING_SIZE * len(df_subset))
    # testing_size = int(TESTING_SIZE * len(df_subset))

    training.append(df_subset.iloc[:training_size])
    testing.append(df_subset.iloc[training_size:])
    # validation.append(df_subset.iloc[training_size + testing_size:])

  training = pd.concat(training, axis=0)  
  testing = pd.concat(testing, axis=0)
  # validation = pd.concat(validation, axis=0)

  training.to_csv(os.path.join(TRAINING_DATA_PATH, filename), index=False, header=False)
  testing.to_csv(os.path.join(TESTING_DATA_PATH, filename), index=False, header=False)
  # validation.to_csv(os.path.join(VALIDATION_DATA_PATH, filename), index=False, header=False)
      
def load_data(path):
   files = glob.glob(os.path.join(path + '/*.csv'))
   return np.concatenate([np.genfromtxt(file, delimiter=',') for file in files])

def load_datasets():
    training_data = load_data(TRAINING_DATA_PATH)
    testing_data = load_data(TESTING_DATA_PATH)
   
    markers = training_data[:, -1]
    kills = len(markers[markers == 0])
    deaths = len(markers[markers == 1])
    neutral = len(markers[markers == 2])
    print(f"Number of kills={kills}; deaths={deaths}; neutral={neutral}")
   
    return training_data, testing_data


def present_confusion_matrix(matrix, labels=STATES):
    """
    Present the confusion matrix in a readable table format.

    Parameters:
    - matrix: list of lists or 2D array, the confusion matrix
    - labels: list, the class labels
    """
    print("Confusion Matrix:")
    df = pd.DataFrame(matrix, index=[f'Actual: {label}' for label in labels], columns=[f'Predicted: {label}' for label in labels])
    table = tabulate(df, headers='keys', tablefmt='pretty', stralign='center') # type: ignore
    print(table)


def present_metrics(metrics_dict):
    """
    Present the metrics dictionary in a readable table format.

    Parameters:
    - metrics_dict: dict, the metrics dictionary
    """
    print("\nClassification Report:")
    df = pd.DataFrame(metrics_dict).T  # Transpose to have labels as rows
    df = df.round(3)
    table = tabulate(df, headers='keys', tablefmt='pretty', stralign='center') # type: ignore
    print(table)

def apply_notch_filter(df, sampling_rate, freqs=50.0):
    """
    Apply a notch filter to remove specified frequencies from EEG data.

    Parameters:
    df (pd.DataFrame): The EEG data with channels as columns.
    sampling_rate (float): The sampling rate of the EEG data.
    freqs (float or list): The frequency or frequencies to remove (default is 50 Hz).

    Returns:
    pd.DataFrame: The filtered EEG data.
    """
    ch_names = [str(ch) for ch in df.columns]
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types="eeg")
    raw = mne.io.RawArray(df.T.values, info)
    raw.notch_filter(freqs=freqs)
    return pd.DataFrame(raw.get_data().T, columns=ch_names)

def apply_bandpass_filter(df, sampling_rate, l_freq, h_freq):
    """
    Apply a band-pass filter to EEG data.

    Parameters:
    df (pd.DataFrame): The EEG data with channels as columns.
    sampling_rate (float): The sampling rate of the EEG data.
    l_freq (float): The lower bound of the filter.
    h_freq (float): The upper bound of the filter.

    Returns:
    pd.DataFrame: The filtered EEG data.
    """
    ch_names = [str(ch) for ch in df.columns]
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types="eeg")
    raw = mne.io.RawArray(df.T.values, info)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return pd.DataFrame(raw.get_data().T, columns=ch_names)