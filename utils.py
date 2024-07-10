from typing import Generator, List, Tuple
import pandas as pd
import numpy as np
import os
import shutil
import torch


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

class Experiment:
    def __init__(self, path: str) -> None:
        self. filename = path.split('/')[-1]
        self.data = load_data(path).iloc[:,16:56]
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
        # regard margin and annotations 
        experiment[state_name] = pd.concat([experiment[state_name], experiment.data.iloc[state_begin:state_end]], axis=0)
        state_begin = state_end

def shuffleData(data: pd.DataFrame, chunk_size: int):
    chunks = [data[i: i+ chunk_size] for i in range(0, len(data), chunk_size)]
    np.random.shuffle(chunks)
    return pd.concat(chunks, axis=0)


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

        try:
            validation = pd.concat(chunks[int((training_size + testing_size) * chunk_number):], axis=0)
            validation.to_csv(os.path.join(validation_path, state + '_' + experiment.filename), index=False)
        except:
            print('Validation set won\'t be created...' )

# find all paths for training and testing
# import 4 dataframes from each state
# add 'boring', 'flow', 'frustration', 'neutral' columns and fill them with 0s and 1 
# merge all dataframes into one, split into 1-second chunks and shuffle it
# yield the output
def list_files_for_states(directory: str, states: List[str]) -> dict[str, List[str]]:
    files = {}
    for state in states:
        files[state] = list_files(directory, state)
    return files

def load_data_for_learning(filenames: dict[str, List[str]], iteration) -> pd.DataFrame:
        data = {}
        states = filenames.keys()
        for state in states:
            data[state] = load_data(filenames[state][iteration])

            data[state][state] = 1

        training_data = pd.concat([data[state] for state in states], axis=0)
        training_data.fillna(int(0), inplace=True)
        # training_data = shuffleData(training_data, SAMPLING_RATE)
        return training_data
    

def getTrainingRowNum(states: List[str], training_dir: str='training') -> int:
    # it should also take bach size into account, because we drop $(datasets % batch_size) rows, but for now "-5" will do
    training_filenames = list_files_for_states(training_dir, states)
    datasets_num = len(training_filenames[states[0]])

    sum = 0
    for i in range(datasets_num):
        sum += len(load_data_for_learning(training_filenames, i))
    return sum
  
def prepareXY(data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # x.shape = (batch_num, SAMPLE_RATE, 40)
    # y.shape = (batch_num, 4)

    data = data[: (len(data) // SAMPLING_RATE) * SAMPLING_RATE]
    tensor = torch.tensor(data.values, dtype=torch.float32)
    training = tensor.view(-1, SAMPLING_RATE, tensor.shape[1])

    x = training[:, :, :len(DATA_COLUMNS)]
    y = training[:, 0, len(DATA_COLUMNS):]
    return x, y

def lazy_load_training_data(states: List[str], training_dir: str='training'):
    training_filenames = list_files_for_states(training_dir, states)

    datasets_num = len(training_filenames[states[0]]) # assuming that all states have the same number of datasets
    
    for i in range(datasets_num):
        training_data = load_data_for_learning(training_filenames, i)
        training_data = shuffleData(training_data, SAMPLING_RATE)

        yield prepareXY(training_data)

def load_training_data(states: List[str], training_dir: str='training'):
    training_filenames = list_files_for_states(training_dir, states)
    datasets_num = len(training_filenames[states[0]])

    data_list = []
    for i in range(datasets_num):
        data_list.append(load_data_for_learning(training_filenames, i))

    training_data = pd.concat(data_list, axis=0)
    training_data = shuffleData(training_data, SAMPLING_RATE)
    return prepareXY(training_data)


def load_testing_data(states: List[str], testing_dir: str='testing'):
    # read all files at once
    testing_filenames = list_files_for_states(testing_dir, states)    
    datasets_num = len(testing_filenames[states[0]]) # assuming that all states have the same number of datasets
    
    data_list = []
    for i in range(datasets_num):
        data_list.append(load_data_for_learning(testing_filenames, i))

    training_data = pd.concat(data_list, axis=0)
    training_data = shuffleData(training_data, SAMPLING_RATE)

    # x = training_data.iloc[:, :len(DATA_COLUMNS)]
    # y = training_data.iloc[:, len(DATA_COLUMNS):]
    x, y = prepareXY(training_data)
    return x, y

def clear_directories(dirs: List[str]):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    