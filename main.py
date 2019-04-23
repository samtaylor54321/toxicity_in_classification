from Model_Build.lstm import LSTMClassifier, BidirectionalLSTMClassifier
from sklearn.model_selection import StratifiedKFold
from Model_Build.tokenise import *


# User defined params
params = {
    'train_data_path': 'Data/train.csv',
    'results_path': 'Results',
    'embedding': 'ft_common_crawl',  # See embeddings.py for dict of options
    'debug_size': None,
    'max_sequence_length': 100,
    'models': ['bidi_lstm'],
    'lstm_units': 127,
    'batch_size': 1024,
    'max_epochs': 5,
    'n_cv_folds': 3,
    'random_seed': 0,
}

model_dict = {
    'lstm': LSTMClassifier,
    'bidi_lstm': BidirectionalLSTMClassifier
}

df = pd.read_csv(params['train_data_path'], nrows=params['debug_size'])

# Pre-process
sample_weight = get_weights(df)
df, y = separate_target(df)
df = spacy_tokenise_and_lemmatize(df)
X, word_index = sequence_tokens(df, params, train=True)
del df

# Train models
for model_name in params['models']:
    model = model_dict[model_name](params, word_index)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    model.cv(X, y, cv, sample_weight)
    model.train(X, y, sample_weights=sample_weight)
    model.save(params['results_path'])
