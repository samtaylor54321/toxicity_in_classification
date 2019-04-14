import yaml
from numpy import mean, nan
from pathlib import Path
from .base import BaseClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM


class LSTMClassifier(BaseClassifier):

    def __init__(self, params):
        super().__init__(params)
        self.lstm_units = params['lstm_units']

    def create_model(self):
        model = Sequential()
        embedding = self.embedding_as_keras_layer()
        model.add(embedding)
        model.add(LSTM(units=self.lstm_units,
                       dropout=0.2,
                       recurrent_dropout=0.2))
        model.add(Dense(units=1,
                        activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='nadam',
            metrics=[self.auc]
        )
        return model

    def save(self, path):
        if len(self.cv_comp_metrics) > 0:
            score = mean(self.cv_comp_metrics)
        else:
            score = self.comp_metric
        score = nan if score is None else score

        out_dir = Path(path)
        out_dir = out_dir / '{}_score_{:.4f}'.format(self.run_timestamp, score)
        out_dir.mkdir(parents=True, exist_ok=True)

        results_out_path = out_dir / 'CONFIG_lstm.csv'
        with results_out_path.open('w') as f:
             yaml.dump(self.run_config, f)

        model_out_path = out_dir / 'MODEL_lstm.h5'
        self.model.save(str(model_out_path))
