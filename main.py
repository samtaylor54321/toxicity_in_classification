import gc
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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

# # --- Parameters -- # #

params = {
    'train_data_path': 'Data/train.csv',
    'test_data_path': 'Data/test.csv',
    'results_path': 'Results',
    'use_premade_relational_features': True,
    'embedding': 'ft_common_crawl',  # See embeddings.py for dict of options
    'debug_size': None,
    'max_sequence_length': 100,
    'models_to_train': [],  # Used to create a stack
    'pre_trained_models': ['lstm', 'bidi_lstm'],  # Added to stack
    'tree_model': 'lightgbm',  # Trained on the stack's OOF preds
    'lstm_units': 64,
    'batch_size': 256,  # 512 for LSTM, 256 for Bidirectional lstm
    'max_epochs': 10,
    'n_cv_folds': 4,
    'random_seed': 0,
}
df = pd.read_csv(params['train_data_path'], nrows=params['debug_size'])
test_df = pd.read_csv(params['test_data_path'], nrows=params['debug_size'])

# # --- Model options --- # #

model_dict = {
    'lstm': LSTMClassifier,
    'bidi_lstm': BidirectionalLSTMClassifier,
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

# Train each model, recording OOF predictions for use in a stacked model
model_preds, model_reps = [], []
for model_name in params['models_to_train']:
    model = model_dict[model_name](params, word_index)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    logging.info('Cross validating model %s', model_name)
    cv_pred, cv_rep, test_pred, test_rep = model.cv(X, y, X_test, cv, sample_weight)
    model_preds.append(cv_pred)
    model_reps.append(cv_rep)
    logging.info('Training model %s on full dataset', model_name)
    model.train(X, y, sample_weights=sample_weight)
    model.save(params['results_path'], run_timestamp)
    cv_rep.to_csv('Data/{}_train_data_representation.csv'.format(model.__name__),
                  index=False)
    cv_pred.to_csv('Data/{}_train_data_prediction.csv'.format(model.__name__),
                  index=False)
    test_rep.to_csv('Data/{}_test_data_representation.csv'.format(model.__name__),
                  index=False)
    test_pred.to_csv('Data/{}_test_data_prediction.csv'.format(model.__name__),
                  index=False)
    del model
    gc.collect()
del X

# If using any pretrained models, add their representations to the stack
for model_name in params['pre_trained_models']:
    logging.info('Loading pretrained representations from model %s', model_name)
    model = model_dict[model_name](params, None)
    cv_rep = pd.read_csv('Data/{}_train_data_representation.csv'
                         .format(model.__name__), nrows=params['debug_size'])
    cv_pred = pd.read_csv('Data/{}_train_data_prediction.csv'
                          .format(model.__name__), nrows=params['debug_size'])
    model_preds.append(cv_pred)
    model_reps.append(cv_rep)

# Train a tree model on the stack's predictions
logging.info('Combining model representations and relational features')
X_trees = pd.concat(model_preds + model_reps + [features], axis=1)
X_trees = X_trees.reindex(sorted(X_trees.columns), axis=1)
logging.info('Writing final training set')
X_trees.to_csv('Data/stacked_model_dataset.csv', index=False)

logging.info('Training final model')
model = model_dict[params['tree_model']](params)
model.cv(X_trees, y, sample_weights=sample_weight)
model.train(X_trees, y, sample_weights=sample_weight)
model.save(params['results_path'], run_timestamp)
logging.info(model.cv_results)
logging.info('Complete')

