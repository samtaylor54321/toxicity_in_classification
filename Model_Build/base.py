import gc
import logging
import yaml
import pandas as pd
import numpy as np
from numpy import mean, nan
from pathlib import Path
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from .utils import get_identities, get_final_metric, \
    compute_bias_metrics_for_model
from .embeddings import get_embedding_details, build_embedding_matrix

class BaseKerasClassifier:
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
        self.comp_metric = None
        self.cv_comp_metrics = []
        self.result = {}
        self.cv_results = []
        self.model = None
        self.representation_layer = None
        self.embedding_matrix = None

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
            if self.embedding_matrix is None:
                self.embedding_matrix = build_embedding_matrix(
                    self.word_index,
                    embedding_details['path'],
                    embedding_details['dim']
                )
            return Embedding(*self.embedding_matrix.shape,
                             weights=[self.embedding_matrix],
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
                    patience=2,
                    verbose=1
                )
            ]
        )

        if train_idx is not None:
            logging.info('Computing bias metrics...')
            identity_data = self.load_identity_data()
            identity_data = identity_data.iloc[val_idx, :]

            y_pred = self.model.predict(X_val)
            bias_metrics_df = compute_bias_metrics_for_model(
                identity_data, self.bias_identities, y_val, y_pred
            )
            self.comp_metric = get_final_metric(
                bias_metrics_df, roc_auc_score(y_val, y_pred)
            )
            logging.info('Bias metrics computed. Score = {:.4f}'
                         .format(self.comp_metric))

    def cv(self, X, y, X_test, cv=StratifiedKFold(3), sample_weights=None):
        """ Apply training function in CV fold """
        oof_preds = pd.DataFrame(np.zeros(y.shape),
                                 columns=[self.__name__ + '_preds'])
        test_preds = np.zeros([X_test.shape[0], cv.get_n_splits()])
        oof_reps, val_idxs = [], []
        for fold_no, (train_idx, val_idx) in \
                enumerate(cv.split(X, np.round(y))):
            logging.info('Fitting fold {} / {}'
                         .format(fold_no + 1, cv.get_n_splits()))

            self.train(X, y, train_idx, val_idx, sample_weights)
            self.cv_comp_metrics.append(self.comp_metric)
            self.cv_results.append(self.result.history)

            logging.info('Making out-of-fold predictions')
            oof_preds.iloc[val_idx] = self.model.predict(X[val_idx])
            test_preds[:, fold_no] = self.model.predict(X_test).reshape(-1,)
            logging.info('Getting out-of-fold representations')
            oof_reps.append(self.intermediate_prediction(
                self.representation_layer,
                X[val_idx])
            )
            if fold_no == 0:
                test_reps = self.intermediate_prediction(
                    self.representation_layer,
                    X_test
                )
            else:
                test_reps += self.intermediate_prediction(
                    self.representation_layer,
                    X_test
                )
            val_idxs.append(val_idx)
            K.clear_session()
            del self.model
            gc.collect()

        self.run_config['cv_comp_metrics'] = self.cv_comp_metrics
        self.run_config['cv_results'] = self.cv_results
        oof_reps = pd.DataFrame(
            data=np.concatenate(oof_reps),
            index=np.concatenate(val_idxs),
        )
        oof_reps.columns = [self.__name__ + '_rep_dim_' + str(i)
                            for i in range(oof_reps.shape[1])]

        test_preds = pd.DataFrame(
            data=test_preds.mean(axis=1),
            columns=[self.__name__ + '_preds']
        )
        test_reps = pd.DataFrame(
            data=test_reps / cv.get_n_splits(),
            columns=[self.__name__ + '_rep_dim_' + str(i)
                     for i in range(test_reps.shape[1])]
        )
        return oof_preds, oof_reps, test_preds, test_reps

    def intermediate_prediction(self, prediction_layer, data):
        """ Predict on an intermediate layer """
        intermediate_layer_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(prediction_layer).output
        )
        return intermediate_layer_model.predict(data)

    def save(self, path, timestamp):
        if len(self.cv_comp_metrics) > 0:
            score = mean(self.cv_comp_metrics)
        else:
            score = self.comp_metric
        score = nan if score is None else score

        out_dir = Path(path)
        out_dir = out_dir / '{}'.format(timestamp)
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

        model_out_path = out_dir / 'MODEL_{}_score_{:.4f}.h5'\
            .format(self.__name__, score)
        self.model.save(str(model_out_path))
