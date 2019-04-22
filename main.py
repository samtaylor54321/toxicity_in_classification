import pandas as pd
import time
from Model_Build.lstm import LSTMClassifier
from sklearn.model_selection import StratifiedKFold
from Model_Build.tokenise import get_weights_and_sequence_tokens


# User defined params
params = {
    'train_data_path': 'Data/train.csv',
    'results_path': 'Results',

    'embedding': 'glove_twitter',  # None for a trainable Keras embedding
    'debug_size': None,
    'max_sequence_length': 100,
    'models': ['lstm'],
    'lstm_units': 254,
    'batch_size': 1024,
    'max_epochs': 5,
    'n_cv_folds': 3,
    'random_seed': 0,
}

model_dict = {
    'lstm': LSTMClassifier,
}

df = pd.read_csv(params['train_data_path'], nrows=params['debug_size'])
X, y, _, word_index, sample_weight = \
    get_weights_and_sequence_tokens(df, params)

for model_name in params['models']:
    model = model_dict[model_name](params, word_index)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    model.cv(X, y, cv, sample_weight)
    model.train(X, y, sample_weights=sample_weight)
    model.save(params['results_path'])
