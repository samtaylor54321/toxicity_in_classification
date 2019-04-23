from .base import BaseClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from .utils import auc


class LSTMClassifier(BaseClassifier):

    def __init__(self, params, word_index):
        super().__init__(params, word_index)
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
            loss='binary-crossentropy',
            optimizer='nadam',
            metrics=[auc]
        )
        return model


class BidirectionalLSTMClassifier(BaseClassifier):

    def __init__(self, params, word_index):
        super().__init__(params, word_index)
        self.__name__ = 'bidirectional_lstm_classifier'
        self.lstm_units = params['lstm_units']


    def create_model(self):
        model = Sequential()
        embedding = self.embedding_as_keras_layer()
        model.add(embedding)
        model.add(Bidirectional(LSTM(units=self.lstm_units,
                                     dropout=0.2,
                                     recurrent_dropout=0.2)))
        model.add(Dense(units=1,
                        activation='sigmoid'))
        model.compile(
            loss='binary-crossentropy',
            optimizer='nadam',
            metrics=[auc]
        )
        return model