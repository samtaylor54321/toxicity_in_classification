import logging
import pandas as pd
import numpy as np
import lightgbm as lgbm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from Model_Build.utils import get_identities, compute_bias_metrics_for_model, \
    get_final_metric


def custom_eval(y_true, y_pred):
    logging.info('Computing bias metrics...')
    identity_data = pd.read_csv('Data/train.csv',
                                usecols=get_identities())
    identity_data.fillna(0).astype(bool)
    bias_metrics_df = compute_bias_metrics_for_model(
        identity_data, get_identities(), y_true, y_pred
    )
    return get_final_metric(
        bias_metrics_df, roc_auc_score(y_true, y_pred)
    )


class LightGBMClassifier:

    def __init__(self, params):
        self.__name__ = 'lightgbm'
        self.params = params
        self.model = None
        self.cv_results = None
        self.optimum_boost_rounds = None
        self.lgbm_params = {
            'objective': 'binary',
            'metric': 'auc'
        }
        self.comp_metric = None

    def train(self, X, y, sample_weights):
        train_set = lgbm.Dataset(
            data=X,
            label=y,
            weight=sample_weights
        )
        self.model = lgbm.train(
            params=self.lgbm_params,
            train_set=train_set,
            num_boost_round=self.optimum_boost_rounds or 100
        )
        self.X_column_names = X.columns

    def cv(self, X, y, sample_weights):
        train_set = lgbm.Dataset(
            data=X,
            label=y,
            weight=sample_weights
        )
        self.cv_results = lgbm.cv(
            params=self.lgbm_params,
            train_set=train_set,
            nfold=self.params['n_cv_folds'],
            early_stopping_rounds=20,
            seed=self.params['random_seed'],
            verbose_eval=20
        )
        self.optimum_boost_rounds = np.argmax(self.cv_results['auc-mean'])

    def save(self, save_dir):
        if self.cv_results:
            score = np.max(self.cv_results['auc-mean'])
        else:
            score = None

        importance_path = save_dir / 'lightgbm_feature_importance.csv'
        feature_importance = pd.DataFrame({
            'Feature': self.X_column_names,
            'Importance': self.model.feature_importance()
        }).sort_values(by='Importance', ascending=False)
        feature_importance.to_csv(importance_path, index=False)

        model_out_path = out_dir / 'MODEL_{}_score_{:.4f}.txt'\
            .format(self.__name__, score)
        self.model.save_model(str(model_out_path))