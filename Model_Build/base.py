import gc
import yaml
from numpy import mean, nan
from pathlib import Path
from datetime import datetime
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from .utils import *
from .embeddings import *

class BaseClassifier:
    """
    Base class of Keras classifiers
    """

    def __init__(self, params, word_index):
        """ Set generic parameters """
        self.run_config = params
        self.word_index = word_index
        self.embedding = params['embedding']
        self.identity_data_path = params['train_data_path']
        self.batch_size = params['batch_size']
        self.epochs = params['max_epochs']
        self.bias_identities = get_identities()
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H.%M.%S')
        self.comp_metric = None
        self.cv_comp_metrics = []
        self.result = {}
        self.cv_results = []
        self.model = None

    def get_n_unique_words(self):
        data = pd.read_csv(self.run_config['train_data_path'],
                           nrows=self.run_config['debug_size'])
        return len(pd.unique(data.values.ravel('K')))

    def load_identity_data(self):
        identity_data = pd.read_csv(self.identity_data_path,
                                    usecols=self.bias_identities,
                                    nrows=self.run_config['debug_size'])
        return identity_data.fillna(0).astype(bool)

    def embedding_as_keras_layer(self):
        """ Load specified embedding as a Keras layer """
        embedding_details = get_embedding_details(self.embedding)
        if self.embedding == 'word2vec':
            embedding_matrix = pd.read_csv(embedding_details['path'])
            return Embedding(*embedding_matrix.shape,
                             weights=[embedding_matrix],
                             trainable=False)
        elif self.embedding in ['ft_common_crawl', 'glove_twitter']:
            embedding_matrix = build_embedding_matrix(
                self.word_index,
                embedding_details['path'],
                embedding_details['dim']
            )
            return Embedding(*embedding_matrix.shape,
                             weights=[embedding_matrix],
                             trainable=False)
        else:
            return Embedding(self.get_n_unique_words(), 100)

    def create_model(self):
        pass

    def train(self, X, y, train_idx=None, val_idx=None, sample_weights=None):
        """ Define a training function with early stopping """
        if train_idx is not None:
            X_train = X[train_idx]
            X_val = X[val_idx]
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
            else:
                y_train = y[train_idx]
                y_val = y[val_idx]
            validation_data = [X_val, y_val]
            sample_weights = sample_weights[train_idx]
        else:
            X_train = X
            y_train = y
            validation_data = None
        del X, y
        gc.collect()

        self.model = self.create_model()
        early_stopping_loss = 'loss' if train_idx is None else 'val_loss'

        self.result = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            sample_weight=sample_weights,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_loss,
                    min_delta=0.001,
                    patience=3,
                    verbose=1
                )
            ]
        )

        if train_idx is not None:
            print('Computing bias metrics...')
            identity_data = self.load_identity_data()
            identity_data = identity_data.iloc[val_idx, :]

            y_pred = self.model.predict(X_val)
            bias_metrics_df = compute_bias_metrics_for_model(
                identity_data, self.bias_identities, y_val, y_pred
            )
            self.comp_metric = get_final_metric(
                bias_metrics_df, roc_auc_score(y_val, y_pred)
            )
            print('Bias metrics computed. Score = {:.4f}'
                  .format(self.comp_metric))

    def cv(self, X, y, cv=StratifiedKFold(3), sample_weights=None):
        """ Apply training function in CV fold """
        for fold_no, (train_idx, val_idx) in \
                enumerate(cv.split(X, np.round(y))):
            print('Fitting fold {} / {}\n'.format(fold_no + 1, cv.get_n_splits()))

            self.train(X, y, train_idx, val_idx, sample_weights)
            self.cv_comp_metrics.append(self.comp_metric)
            self.cv_results.append(self.result.history)

            K.clear_session()
            del self.model
            gc.collect()

        self.run_config['cv_comp_metrics'] = self.cv_comp_metrics
        self.run_config['cv_results'] = self.cv_results

    def save(self, path):
        if len(self.cv_comp_metrics) > 0:
            score = mean(self.cv_comp_metrics)
        else:
            score = self.comp_metric
        score = nan if score is None else score

        out_dir = Path(path)
        out_dir = out_dir / '{}_score_{:.4f}'.format(self.run_timestamp, score)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_out_path = out_dir / 'CONFIG_{}.yaml'.format(self.__name__)
        self.run_config['cv_comp_metrics'] = \
            [str(x) for x in self.run_config['cv_comp_metrics']]
        scores = {}
        for i, fold_scores in enumerate(self.run_config['cv_results']):
            scores['fold_' + str(i)] = \
                {metric: [str(score) for score in scores]
                for metric, scores in fold_scores.items()}
        self.run_config['cv_results'] = scores
        with results_out_path.open('w') as f:
             yaml.dump(self.run_config, f)

        model_out_path = out_dir / 'MODEL_{}.h5'.format(self.__name__)
        self.model.save(str(model_out_path))
