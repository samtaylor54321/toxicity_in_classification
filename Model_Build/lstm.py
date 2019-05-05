from .base import BaseKerasClassifier
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Bidirectional, SpatialDropout1D
from .utils import auc


class LSTMClassifier(BaseKerasClassifier):

    def __init__(self, params, word_index):
        super().__init__(params, word_index)
        self.__name__ = 'lstm_classifier'
        self.representation_layer = 'cu_dnnlstm_1'
        self.lstm_units = params['lstm_units']

    def create_model(self):
        model = Sequential()
        embedding = self.embedding_as_keras_layer()
        model.add(embedding)
        model.add(SpatialDropout1D(rate=0.2))
        model.add(CuDNNLSTM(units=self.lstm_units))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='nadam',
            metrics=[auc]
        )
        return model


class BidirectionalLSTMClassifier(BaseKerasClassifier):

    def __init__(self, params, word_index):
        super().__init__(params, word_index)
        self.__name__ = 'bidirectional_lstm_classifier'
        self.representation_layer = 'bidirectional_1'
        self.lstm_units = params['lstm_units']

    def create_model(self):
        model = Sequential()
        embedding = self.embedding_as_keras_layer()
        model.add(embedding)
        model.add(SpatialDropout1D(rate=0.2))
        model.add(Bidirectional(CuDNNLSTM(units=self.lstm_units)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='nadam',
            metrics=[auc]
        )
        return model