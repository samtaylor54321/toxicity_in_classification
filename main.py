import pandas as pd
import keras.backend as K
from Model_Build.lstm import LSTMClassifier
from sklearn.model_selection import StratifiedKFold


# User defined params
params = {
    'train_data_path': 'Data/train.csv',
    'pretokenised_path': 'Data/Tokenised_Test/tokenised_train.csv',
    'sequences_path': 'Data/Sequenced_Text/word2vec_train_custom_stopwords.csv',
    'embedding_path': 'Embedding_Build/Trained_Embeddings/'
                      'word2vec_embedding_weights_custom_stopwords.csv',
    'results_path': 'Results',

    'debug_size': 1000,
    'identity_weight': 1.2,
    'models': ['lstm'],
    'lstm_units': 254,
    'batch_size': 2048,
    'max_epochs': 5,
    'n_cv_folds': 3,
    'random_seed': 0,
}

model_dict = {
    'lstm': LSTMClassifier,
}

X = pd.read_csv(params['sequences_path'], nrows=params['debug_size'])
if 'Unnamed: 0' in X.columns:
    X.drop(columns=['Unnamed: 0'], inplace=True)
X = X.values

y = pd.read_csv(params['train_data_path'], nrows=params['debug_size'], usecols=['target'])
y = y['target']
y = (y >= .5).astype(int)

for model_name in params['models']:
    model = model_dict[model_name](params)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    model.cv(X, y, cv)
    model.train(X, y)
    model.save(params['results_path'])
