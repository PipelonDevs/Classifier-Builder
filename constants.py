#### PATHS ####
DATASET_PATH = '../EEG_Datasets/FinalBNMDataSets/'

TRANING_DATA_PATH = './training/'
TESTING_DATA_PATH = './testing/'
VALIDATION_DATA_PATH = './validation/'

EXPORTED_MODELS_PATH = './exported_models'

SURVEY_PREFIX = 'Ankieta_'
DATASET_PREFIX = 'Data_'
ANNOTAIONS_PREFIX = 'markers-survey-'

#### EXPERIMENT ####
SAMPLING_RATE = 25 # Hz
STATES = ['boredom', 'flow'] # boredom, neutral, flow, frustration
EXPERIMENT_DURATION = 39.5 * 60 
# DATA_COLUMNS = [
#     'alpha channel 1', 'alpha channel 2', 'alpha channel 3', 'alpha channel 4',
#     'alpha channel 5', 'alpha channel 6', 'alpha channel 7', 'alpha channel 8',
#     'beta low channel 1', 'beta low channel 2', 'beta low channel 3', 'beta low channel 4',
#     'beta low channel 5', 'beta low channel 6', 'beta low channel 7', 'beta low channel 8',
#     'beta mid channel 1', 'beta mid channel 2', 'beta mid channel 3', 'beta mid channel 4',
#     'beta mid channel 5', 'beta mid channel 6', 'beta mid channel 7', 'beta mid channel 8',
#     'beta high channel 1', 'beta high channel 2', 'beta high channel 3', 'beta high channel 4',
#     'beta high channel 5', 'beta high channel 6', 'beta high channel 7', 'beta high channel 8',
#     'gamma channel 1', 'gamma channel 2', 'gamma channel 3', 'gamma channel 4',
#     'gamma channel 5', 'gamma channel 6', 'gamma channel 7', 'gamma channel 8'
# ]
DATA_COLUMNS = [
  'alpha channel 1', 'alpha channel 2', 'alpha channel 3', 'alpha channel 4',
  'beta low channel 1', 'beta low channel 2', 'beta low channel 3', 'beta low channel 4',
  'beta mid channel 1', 'beta mid channel 2', 'beta mid channel 3', 'beta mid channel 4',
  'beta high channel 1', 'beta high channel 2', 'beta high channel 3', 'beta high channel 4',
  'gamma channel 1', 'gamma channel 2', 'gamma channel 3','gamma channel 4'
 ]
#### TRAINING ####
TRAINING_SIZE, TESTING_SIZE, VALIDATION_SIZE = 0.7, 0.2, 0.1
