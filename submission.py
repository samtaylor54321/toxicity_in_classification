import gc
import pickle
import string
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Paths to things we'll need
test_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'
model_path = '../input/lstm-commoncrawl-classifier/MODEL_lstm_classifier.h5'

params = {
    'max_sequence_length': 100,
    'embedding': None
}


def main(data_path, model_path):
    # Load and tokenise
    test = pd.read_csv(data_path)
    X = get_weights_and_sequence_tokens(test, params, train=False)

    # Run model
    print('Loading model')
    model = load_model(model_path, custom_objects={'auc': auc, 'custom_loss': custom_loss})
    print('Predicting')
    y_pred = model.predict(X)

    # Submit
    print('Saving submission')
    submission = pd.DataFrame({'id': test.index,
                               'prediction': y_pred.reshape(-1, )})
    submission.to_csv('submission.csv', index=False)
    print('Finished')


def auc(y_true, y_pred):
    """ Tensor-based ROC-AUC metric for use as loss function """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true, (-1, 1)),
                               y_pred)


def spacy_tokenise_and_lemmatize(df):
    # Identify punctuation and stopwords to ignore
    punctuations = string.punctuation
    custom_stop_words = False

    if not custom_stop_words:
        stopwords = list(STOP_WORDS)
    else:
        stopwords_to_allow = [
            'really', 'should', 'must', 'much', 'ourselves', 'you', 'see',
            'nobody', 'him', 'everywhere', 'side', 'they', 'herself',
            'i', 'are', 'hers', 'your', 'but', 'its', 'every',
            'he', 'her', 'get', 'noone', 'whatever', 'very',
            'some', 'yourself', 'into', 'us', 'ours', 'off',
            'we', 'himself', 'themselves', 'our', 'she',
            'yours', 'anyone', 'me', 'go', 'same', 'those', 'my', 'too', 'myself', 'them',
            'all', 'his', 'against', 'others', 'please'
        ]
        stopwords = [word for word in list(STOP_WORDS)
                     if word not in stopwords_to_allow]

    # Get a spaCy model for English
    parser = English()

    # Tokenization function
    def spacy_tokenizer(sentence):
        mytokens = parser(sentence)
        mytokens = [word.lemma_.lower().strip()
                    if word.lemma_ != '-PRON-' else word.lower_
                    for word in mytokens]
        mytokens = [word for word in mytokens
                    if word not in stopwords
                    and word not in punctuations]
        return ' '.join([i for i in mytokens])

    # Overall processing function
    df.set_index('id', inplace=True)
    df['comment_text'] = df['comment_text'].apply(spacy_tokenizer)
    return df


def get_weights_and_sequence_tokens(df, params, train=True):
    """
    Inspired by:
    https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/
    """

    print('Tokenising train set')
    df = spacy_tokenise_and_lemmatize(df)
    if train:
        test = pd.read_csv('Data/test.csv')
        print('Tokenising test set')
        test = spacy_tokenise_and_lemmatize(test)
        tokenizer = Tokenizer()
        print('Creating keras tokeniser and word index')
        tokenizer.fit_on_texts(list(df['comment_text'])
                               + list(test['comment_text']))
        word_index = tokenizer.word_index
        pickle.dump(tokenizer, open('Model_Build/Trained_Models/keras_tokeniser.pkl', 'wb'))
        del test
    else:
        print('Loading pretrained tokenizer')
        tokenizer = pickle.load(open('../input/keras-tokeniser/keras_tokeniser.pkl', 'rb'))
        word_index = tokenizer.word_index

    print('Sequencing and padding tokenised text')
    if params['embedding'] == 'word2vec':
        w2v = pickle.load(
            open('Model_Build/Trained_Models/'
                 'word2vec_model_custom_stopwords.pkl', 'rb')
        )
        sequences = []
        for row in df['comment_text'].str.split(' ').tolist():
            sequences.append([w2v.wv.vocab[word].index
                              if word in w2v.wv.vocab else 0
                              for word in row])
        sequences = pad_sequences(sequences, maxlen=100)
        X = pd.DataFrame(sequences).values
    else:
        X = tokenizer.texts_to_sequences(list(df['comment_text']))
        X = pad_sequences(X, maxlen=params['max_sequence_length'])

    del tokenizer, df
    gc.collect()

    return X


if __name__ == '__main__':
    main(test_path, model_path)
