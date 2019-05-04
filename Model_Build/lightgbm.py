import pandas as pd
import numpy as np
import lightgbm as lgbm
from pathlib import Path


class LightGBMClassifier:

    def __init__(self, params):
        self.params = params
        self.model = None
        self.cv_results = None
        self.lgbm_params = {
            'objective': 'binary',
            'metric': 'auc'
        }

    def train(self, X, y, sample_weights):
        train_set = lgbm.Dataset(
            data=X,
            label=y,
            weight=sample_weights
        )
        self.model = lgbm.train(
            params=self.lgbm_params,
            train_set=train_set
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
            seed=self.params['random_seed']
        )

    def save(self, path, timestamp):
        if self.cv_results:
            score = np.max(self.cv_results['auc-mean'])
        else:
            score = None
        out_dir = Path(path)
        out_dir = out_dir / '{}_lightgbm_score_{:.4f}'.format(timestamp, score)
        out_dir.mkdir(parents=True, exist_ok=True)

        importance_path = out_dir / 'feature_importance.csv'
        feature_importance = pd.DataFrame({
            'Feature': self.X_column_names,
            'Importance': self.model.feature_importance()
        }).sort_values(by='Importance', ascending=False)
        feature_importance.to_csv(importance_path, index=False)