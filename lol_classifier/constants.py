#### PATHS ####
DATASET_PATH = '/media/jakubner/WINDOWSD/jakubner/Pipelon/realtime-eeg-experiments/Mind-Collector/data/'

TRAINING_DATA_PATH = './training/'
TESTING_DATA_PATH = './testing/'
VALIDATION_DATA_PATH = './validation/'

EXPORTED_MODELS_PATH = './exported_models'

DATASET_POSTFIX = 'eeg.csv'

#### EXPERIMENT ####
SAMPLING_RATE = 25 # Hz
STATES = ['kill', 'death', 'neutral' ]
DATA_COLUMNS = [
#    'theta channel 1', 'theta channel 2', 'theta channel 3', 'theta channel 4',
#    'theta channel 5', 'theta channel 6', 'theta channel 7', 'theta channel 8',
    'alpha channel 1', 'alpha channel 2', 'alpha channel 3', 'alpha channel 4',
    # 'alpha channel 5', 'alpha channel 6', 'alpha channel 7', 'alpha channel 8',
    'beta low channel 1', 'beta low channel 2', 'beta low channel 3', 'beta low channel 4',
    # 'beta low channel 5', 'beta low channel 6', 'beta low channel 7', 'beta low channel 8',
    'beta mid channel 1', 'beta mid channel 2', 'beta mid channel 3', 'beta mid channel 4',
    # 'beta mid channel 5', 'beta mid channel 6', 'beta mid channel 7', 'beta mid channel 8',
    'beta high channel 1', 'beta high channel 2', 'beta high channel 3', 'beta high channel 4',
    # 'beta high channel 5', 'beta high channel 6', 'beta high channel 7', 'beta high channel 8',
    'gamma channel 1', 'gamma channel 2', 'gamma channel 3', 'gamma channel 4',
    # 'gamma channel 5', 'gamma channel 6', 'gamma channel 7', 'gamma channel 8'
]

#### TRAINING ####
TRAINING_SIZE, TESTING_SIZE, VALIDATION_SIZE = 0.8, 0.2, 0.0
