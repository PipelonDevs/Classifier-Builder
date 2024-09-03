import os, shutil, glob
import pandas as pd
import numpy as np
from constants import *

def read_eeg_data(path) -> pd.DataFrame:
  df = pd.read_csv(path, low_memory=False, header=None)
  df = pd.concat([df.iloc[:, 16:56], df.iloc[:, -1].astype(str)], axis=1) # take only alpha, beta, gamma
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
    print(f"Number of kills={kills}; deaths={deaths}")
   
    return training_data, testing_data
