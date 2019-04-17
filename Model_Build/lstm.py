import yaml
import pickle
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

    def save(self, dir):
        out_dir = Path(dir)
        out_dir = out_dir / self.run_timestamp
        out_dir.mkdir(parents=True, exist_ok=True)

        model_out_path = out_dir / 'MODEL_lstm.pkl'
        with open(model_out_path, 'wb') as model_out:
            pickle.dump(self.model, model_out)

        results_out_path = outdir / 'RESULTS_lstm.csv')
        with open(results_out_path, 'w') as results_out:
            yaml.dump(self.run_config, results_out)
