from .base import BaseKerasClassifier
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, \
    GlobalAveragePooling1D
from .utils import auc


class LSTMClassifier(BaseKerasClassifier):
    """ Baseline LSTM """

    def __init__(self, params, word_index):
        super().__init__(params, word_index)
        self.__name__ = 'lstm_classifier'
        self.representation_layer = 'cu_dnnlstm_1'

        # -- Define model parameters -- #
        self.embedding = 'ft_common_crawl'  # See embeddings.py for dict of options
        self.lstm_units = 64
        self.max_epochs = 10
        self.batch_size = 512

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
    """ 'Borrows' the graph used by the best Kaggle public kernels  """

    def __init__(self, params, word_index):
        super(BidirectionalLSTMClassifier, self).__init__(params, word_index)
        self.__name__ = 'bidirectional_lstm_classifier'
        self.representation_layer = 'concatenate_1'

        # -- Define model parameters -- #
        self.embedding = 'ft_common_crawl'  # See embeddings.py for dict of options
        self.lstm_units = 64
        self.dense_hidden_units = 64 * 4
        self.epochs = 10
        self.batch_size = 256

    def create_model(self):
        words = Input(shape=(self.run_config['max_sequence_length'],))
        x = self.embedding_as_keras_layer()(words)
        x = SpatialDropout1D(0.3)(x)
        x = Bidirectional(CuDNNLSTM(self.lstm_units, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(self.lstm_units, return_sequences=True))(x)

        hidden = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x),
        ])
        hidden = add([
            hidden,
            Dense(self.dense_hidden_units, activation='relu')(hidden)
        ])
        hidden = add([
            hidden,
            Dense(self.dense_hidden_units, activation='relu')(hidden)
        ])
        result = Dense(1, activation='sigmoid')(hidden)

        model = Model(inputs=words, outputs=result)
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=[auc]
        )
        return model


class BidrectionalLSTMGlove(BidirectionalLSTMClassifier):
    """ As above but with GloVe embedding """

    def __init__(self, params, word_index):
        super(BidrectionalLSTMGlove, self).__init__(params, word_index)
        self.__name__ = 'bidirectional_lstm_classifier_glove'
        self.embedding = 'glove_common_crawl'
