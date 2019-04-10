import os
import pickle
import tensorflow as tf
import keras.backend as K
from datetime import datetime
from keras.layers import embeddings
from keras.callbacks import EarlyStopping
from .utils import *

class BaseClassifier:
    """
    Base class of Keras classifiers
    """

    def __init__(self, params):
        self.embedding_path = params['embedding_path']
        self.identity_data_path = params['identity_data_path']
        self.bias_identities = get_identities()
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H.%M.%S')
        self.comp_metric = None
        self.result = {}

    def embedding_as_keras_layer(self):
        """ Load specified embedding as a Keras layer """
        embedding = pickle.load(open(self.embedding_path, 'rb'))
        if isinstance(embedding, embeddings.Embedding):
            return embedding
        else:
            raise TypeError('Embedding at {} can\'t be converted to keras layer'.
                            format(self.embedding_path))

    def create_model(self):
        pass

    def train(self, X_train, y_train, batch_size, max_epochs, validation_data=None):
        """ Define a training function with early stopping """
        self.model = self.create_model()
        self.result = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.001,
                    patience=3,
                    verbose=1
                )
            ]
        )

        if validation_data:
            train = pd.read_csv(self.identity_data_path)
            train[self.bias_identities].fillna(0, inplace=True)
            train.loc[:, self.bias_identities] = train.loc[:, self.bias_identities].astype(bool)

            y_pred = self.model.predict(validation_data[0])
            bias_metrics_df = compute_bias_metrics_for_model(train, self.bias_identities,
                                                             validation_data[1], y_pred)
            self.comp_metric = get_final_metric(bias_metrics_df, roc_auc_score(validation_data[1], y_pred))

    def save(self):
        pass

    @staticmethod
    def auc(y_true, y_pred):
        """ Tensor-based ROC-AUC metric for use as loss function """
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc