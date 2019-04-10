import os
import pickle
from .base import BaseClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM

class LSTMClassifier(BaseClassifier):

    def __init__(self, params):
        super().__init__(params)
        self.lstm_nodes = params['lstm_nodes']

    def create_model(self):
        model = Sequential()
        embedding = self.embedding_as_keras_layer()
        model.add(embedding)
        model.add(LSTM(units=self.lstm_nodes,
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

    def save(self):
        out_path = os.path.join('Trained_Models', 'lstm_' + self.run_timestamp + '.pkl')
        pickle.dump(self.model, open(out_path, 'wb'))

