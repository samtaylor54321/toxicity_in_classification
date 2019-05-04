import logging
import pandas as pd
from datetime import datetime
from Model_Build.lstm import LSTMClassifier, BidirectionalLSTMClassifier
from Model_Build.lightgbm import LightGBMClassifier
from sklearn.model_selection import StratifiedKFold
from Model_Build.tokenise import get_weights, spacy_tokenise_and_lemmatize, \
    separate_target, sequence_tokens
from Vector_Build.feature_build import get_relational_features

run_timestamp = datetime.now().strftime('%Y%m%d_%H.%M.%S')

# User defined params
params = {
    'train_data_path': 'Data/train.csv',
    'results_path': 'Results',
    'use_premade_relational_features': True,
    'embedding': 'ft_common_crawl',  # See embeddings.py for dict of options
    'debug_size': None,
    'max_sequence_length': 100,
    'deep_models': [], #['lstm'],
    'tree_models': ['lightgbm'],
    'lstm_units': 127,
    'batch_size': 1024,
    'max_epochs': 5,
    'n_cv_folds': 3,
    'random_seed': 0,
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model_dict = {
    'lstm': LSTMClassifier,
    'bidi_lstm': BidirectionalLSTMClassifier,
    'lightgbm': LightGBMClassifier
}

df = pd.read_csv(params['train_data_path'], nrows=params['debug_size'])

# Pre-process
if params['use_premade_relational_features']:
    logging.info('Loading pre-made relational features')
    features = pd.read_csv('Data/additional_features.csv',
                           nrows=params['debug_size'])
else:
    features = get_relational_features(df)
sample_weight = get_weights(df)
df, y = separate_target(df)
if params['deep_models']:
    df = spacy_tokenise_and_lemmatize(df)
    X, word_index = sequence_tokens(df, params, train=True)
del df

# Train models
for model_name in params['deep_models']:
    model = model_dict[model_name](params, word_index)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    model.cv(X, y, cv, sample_weight)
    model.train(X, y, sample_weights=sample_weight)
    X_lstm_rep = model.intermediate_prediction('lstm_1', X)
    model.save(params['results_path'], run_timestamp)

for model_name in params['tree_models']:
    model = model_dict[model_name](params)
    model.cv(features, y, sample_weights=sample_weight)
    model.train(features, y, sample_weights=sample_weight)
    model.save(params['results_path'], run_timestamp)
    print(model.cv_results)

