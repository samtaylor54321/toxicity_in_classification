import gc
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from keras import backend as K
from Model_Build.lstm import LSTMClassifier, BidirectionalLSTMClassifier, \
    BidrectionalLSTMGlove
from Model_Build.lightgbm import LightGBMClassifier
from sklearn.model_selection import StratifiedKFold
from Model_Build.tokenise import get_weights, spacy_tokenise_and_lemmatize, \
    separate_target, sequence_tokens
from Vector_Build.feature_build import get_relational_features
run_timestamp = datetime.now().strftime('%Y%m%d_%H.%M.%S')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

# # --- Parameters -- # #

params = {
    'train_data_path': 'Data/train.csv',
    'test_data_path': 'Data/test.csv',
    'results_path': 'Results',
    'use_premade_relational_features': True,
    'debug_size': None,
    'max_sequence_length': 100,
    'models_to_train': [],  # Used to create a stack
    'pre_trained_models': ['bidi_lstm_glove', 'bidi_lstm'],  # Added to stack
    'tree_model': 'lightgbm',  # Trained on the stack's OOF preds
    'n_cv_folds': 4,
    'random_seed': 0,
}
save_dir = Path(params['results_path']) / run_timestamp
save_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(params['train_data_path'], nrows=params['debug_size'])
test_df = pd.read_csv(params['test_data_path'], nrows=params['debug_size'])

# # --- Model options --- # #

model_dict = {
    'lstm': LSTMClassifier,
    'bidi_lstm': BidirectionalLSTMClassifier,
    'bidi_lstm_glove': BidrectionalLSTMGlove,
    'lightgbm': LightGBMClassifier
}

# # --- Pre-processing --- # #

# Relational features for use in any tree models
if params['use_premade_relational_features']:
    logging.info('Loading pre-made relational features')
    features = pd.read_csv('Data/additional_features.csv',
                           nrows=params['debug_size'])
else:
    features = get_relational_features(df)

# Sample weight and label
sample_weight = get_weights(df)
df, y = separate_target(df)

# Convert sentences to sequences of tokens for nn models
df = spacy_tokenise_and_lemmatize(df)
test_df = spacy_tokenise_and_lemmatize(test_df)
X, word_index = sequence_tokens(df, params, train=True)
X_test, _ = sequence_tokens(test_df, params, train=False)
del df, test_df

# # --- Model training --- # #

# Train each model
for model_name in params['models_to_train']:
    model = model_dict[model_name](params, word_index)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    logging.info('Cross validating model %s', model_name)
    model.cv(X, y, X_test, cv, sample_weight)
    logging.info('Training model %s on full dataset', model_name)
    model.train(X, y, sample_weights=sample_weight)
    model.save(save_dir)
    K.clear_session()
    del model
    gc.collect()
del X

logging.info('Complete')

