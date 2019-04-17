import gc
import pandas as pd
from lstm import LSTMClassifier
from sklearn.model_selection import StratifiedKFold


# User defined params
params = {
    'debug_size': None,
    'sequence_path':  '../Data/Sequenced_Text/word2vec_train.csv',
    'identity_data_path': '../Data/train.csv',
    'embedding_path': '../Embedding_Build/Trained_Embeddings/'
                      'word2vec_embedding_weights.csv',
    'results_path': '../Results',
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

X = pd.read_csv(params['sequence_path'], nrows=params['debug_size'])
if 'Unnamed: 0' in X.columns:
    X.drop(columns=['Unnamed: 0'], inplace=True)
X = X.values

train = pd.read_csv(params['identity_data_path'], nrows=params['debug_size'])
y = train.pop('target')
y = (y >= .5).astype(int)
del train
gc.collect()

for model_name in params['models']:
    model = model_dict[model_name](params)
    cv = StratifiedKFold(params['n_cv_folds'],
                         random_state=params['random_seed'])
    model.cv(X, y, cv)
    #model.train(X, y)
    model.save(params['results_path'])
