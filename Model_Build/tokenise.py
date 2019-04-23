import gc
import string
import pickle
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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


def get_weights(df):
    """
    Inspired by:
    https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/
    """
    identity_columns = ['asian', 'atheist',
                        'bisexual', 'black', 'buddhist', 'christian', 'female',
                        'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
                        'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
                        'muslim', 'other_disability', 'other_gender',
                        'other_race_or_ethnicity', 'other_religion',
                        'other_sexual_orientation', 'physical_disability',
                        'psychiatric_or_mental_illness', 'transgender', 'white']

    print('Calculating sample weights')
    # Overall
    weights = np.ones((len(df),))
    # Subgroup
    weights += (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # loss_weight = 1.0 / weights.mean()
    return weights


def separate_target(df):
    y = (df['target'].values >= 0.5).astype(np.int)
    return df.drop(columns=['target']), y


def sequence_tokens(df, params, train=True):
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
        tokenizer = pickle.load(open('Model_Build/Trained_Models/keras_tokeniser.pkl', 'wb'))
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

    return X, word_index