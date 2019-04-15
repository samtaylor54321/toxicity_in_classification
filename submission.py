import pickle
import string
import pandas as pd
from pathlib import Path
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras.backend as K
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# Paths to things we'll need
path = Path('.')
test_path = path / 'Data' / 'test.csv'
sequencer = path / 'Model_Build' / 'Trained_Models' / 'word2vec_model.pkl'
model = path / 'Results' / '20190413_09.43.22_score_0.9191' / 'MODEL_lstm.h5'


def auc(y_true, y_pred):
    """ Tensor-based ROC-AUC metric for use as loss function """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def tokenise(path):
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
    def process(df_path):
        df = pd.read_csv(df_path)
        df.set_index('id', inplace=True)
        df['comment_text'] = df['comment_text'].apply(spacy_tokenizer)
        return df

    return process(path)


# Load and tokenise
test = tokenise(test_path)
test['comment_text'].fillna('emptyword', inplace=True)

# Convert text to sequenced word indices
sequencer = pickle.load(sequencer.open('rb'))
sequences = []
for row in test['comment_text'].str.split(' ').tolist():
    sequences.append([sequencer.wv.vocab[word].index
                      if word in sequencer.wv.vocab else 0
                      for word in row])
sequences = pad_sequences(sequences, maxlen=100)
sequences = pd.DataFrame(sequences).values

# Run model
model = load_model(str(model), custom_objects={'auc': auc})
y_pred = model.predict(sequences)

# Submit
submission = pd.DataFrame({'id': test['id'],
                           'prediction': y_pred})
submission.to_csv('submission.csv', index=False)
