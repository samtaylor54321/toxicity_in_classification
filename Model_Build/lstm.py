from .base import BaseClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM
from .utils import auc


class LSTMClassifier(BaseClassifier):

    def __init__(self, params):
        super().__init__(params)
        self.__name__ = 'lstm_classifier'
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
            metrics=[auc]
        )
        return model
