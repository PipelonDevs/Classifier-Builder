from typing import List, Tuple
import pandas as pd
import numpy as np
import os
import shutil


from constants import DATA_COLUMNS, STATES, SAMPLING_RATE

def list_experiment_files(directory: str, prefix: str, extension:str='.csv') -> List[str]:
    files = []
    for experiment in os.listdir(directory):
        path = os.path.join(directory, experiment)

        for file in os.listdir(path):
            if file.startswith(prefix) and file.endswith(extension):
                files.append(os.path.join(path, file))
    return files

def list_files(directory: str, prefix: str, extension:str='.csv') -> List[str]:
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.startswith(prefix) and file.endswith(extension)]

def load_data(filename) -> pd.DataFrame:
  return pd.read_csv(filename, sep=',')

def select_bands(df: pd.DataFrame, bands: slice = slice(16,56)):
    return df.iloc[:, bands]

def select_channels(df: pd.DataFrame, channels: List[int] = [1,2,3,4]):
    selected_columns = []
    for c in channels:
        selected_columns.append(df.iloc[:, c-1::8])
    if len(selected_columns) == 0:
        raise ValueError('No columns selected')
    return pd.concat(selected_columns, axis=1)

class Experiment:
    def __init__(self, path: str) -> None:
        self. filename = path.split('/')[-1]
        self.data = select_channels(select_bands(load_data(path)))
        #self.annotations

        self.trash = pd.DataFrame()
        self.boredom = pd.DataFrame()
        self.flow = pd.DataFrame()
        self.frustration = pd.DataFrame()
        self.survey = pd.DataFrame()
        self.neutral = pd.DataFrame()

    def __getitem__(self, key: str) -> pd.DataFrame:
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        setattr(self, key, value)

    def __iter__(self):
        return iter(['trash', *STATES])

def getStates(experiment: Experiment, experiment_cycles: List[Tuple[int, str]]):
    state_begin = 0
    for state_end, state_name in experiment_cycles:
        state_end += state_begin
        # TODO: regard annotations 
        experiment[state_name] = pd.concat([experiment[state_name], experiment.data.iloc[state_begin:state_end]], axis=0)
        state_begin = state_end

def splitDataAndSave(experiment: Experiment, training_size: float, testing_size: float, validation_size: float, training_path: str, testing_path: str, validation_path: str):
    for state in experiment: 
        if state == 'trash':
            continue

        chunk_number = SAMPLING_RATE * 4 # 4 seconds
        chunk_size = len(experiment[state]) // chunk_number
        chunks = [experiment[state][i: i+ chunk_size] for i in range(0, len(experiment[state]), chunk_size)]
        np.random.shuffle(chunks)

        training = pd.concat(chunks[:int(training_size * chunk_number)], axis=0)
        testing = pd.concat(chunks[int(training_size * chunk_number):int((training_size + testing_size) * chunk_number)], axis=0)
    
        training.to_csv(os.path.join(training_path, state + '_' + experiment.filename), index=False)
        testing.to_csv(os.path.join(testing_path, state + '_' + experiment.filename), index=False)

        if validation_size != 0:
            validation = pd.concat(chunks[int((training_size + testing_size) * chunk_number):], axis=0)
            validation.to_csv(os.path.join(validation_path, state + '_' + experiment.filename), index=False)

def list_files_for_states(directory: str, states: List[str]) -> dict[str, List[str]]:
    files = {}
    for state in states:
        files_for_state = list_files(directory, state)
        assert len(files_for_state) > 0, f"No files found for state: {state}"
        files[state] = files_for_state
    return files

def load_data_for_learning(filenames: dict[str, List[str]], iteration) -> pd.DataFrame:
        data = {}
        states = filenames.keys()
        for i, state in enumerate(states):
            data[state] = load_data(filenames[state][iteration])[DATA_COLUMNS]
            data[state][state] = 1

        training_data = pd.concat([data[state] for state in states], axis=0)
        training_data.fillna(int(0), inplace=True)
        return training_data

def load_data_for_states(states: List[str], dir: str='testing'):
    # read all files at once
    filenames = list_files_for_states(dir, states)    
    datasets_num = len(filenames[states[0]]) # assuming that all states have the same number of datasets
    
    data_list = []
    for i in range(datasets_num):
        data_list.append(load_data_for_learning(filenames, i))

    training_data = pd.concat(data_list, axis=0)
    return training_data 

def apply_argmax(y):
  return np.argmax(y, axis=1)

def new_load_data(dir: str, for_svm=False) -> tuple[np.ndarray, np.ndarray]:
    training_ds = load_data_for_states(STATES, dir)
    training_ds = training_ds.sample(frac=1, random_state=42).reset_index(drop=True)

    x, y = training_ds.iloc[:, :-len(STATES)], training_ds.iloc[:, -len(STATES):]
    y = y.to_numpy(dtype=np.int32)
    if for_svm:
        y = apply_argmax(y)
        
    return x.to_numpy(), y

def clear_directories(dirs: List[str]):
    print(f"Clearing directories...{dirs}")
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    
from tabulate import tabulate

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
