import gc
import pickle
import string
import logging
import pandas as pd
import numpy as np
import lightgbm as lgbm
from glob import glob
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Paths to things we'll need
model_dir = '../input/final-model/MODEL_lightgbm_score_0.9573.txt'
test_path = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'

params = {
    'max_sequence_length': 100,
    'embedding': None,
    'pretrained_models': ['lstm', 'bidilstm']
}


def main(data_path, model_dir, params):
    # Load and tokenise
    test = pd.read_csv(data_path)
    test = spacy_tokenise_and_lemmatize(test)
    test = spacy_tokenise_and_lemmatize(test)
    X_test, _ = sequence_tokens(test, params, train=False)
    del test

    # Load pretrained models
    logging.info('Loading pretrained models')
    models = [load_model(path) for path in glob(model_dir)]
    scores = [path[-10:-4] for path in glob(model_dir)]

    # Assign weights to a models prediction based on CV score


    # Predict on test set
    preds = np.concatenate([model.predict(X_test) for model in models])
    y_pred = np.zeros([preds.shape[0], ])
    for i, score in enumerate(scores):
        y_pred +=



    # Load test representations from pretrained models
    model_preds = [
        pd.read_csv('../input/common-crawl-lstm-preds/lstm_classifier_test_data_prediction.csv'),
        pd.read_csv('../input/common-crawl-bidilstm-preds/bidirectional_lstm_classifier_test_data_prediction.csv')
    ]
    model_reps = [
        pd.read_csv('../input/common-crawl-lstm-rep/lstm_classifier_test_data_representation.csv'),
        pd.read_csv('../input/common-crawl-bidilstm-rep/bidirectional_lstm_classifier_test_data_representation.csv')
    ]
    X_test = pd.concat(model_preds + model_reps + [features], axis=1)
    X_test = X_test.reindex(sorted(X_test.columns), axis=1)

    # Run model

    model = lgbm.Booster(model_file=model_path)
    logging.info('Predicting')
    y_pred = model.predict(X_test)

    # Submit
    logging.info('Saving submission')
    submission = pd.DataFrame({'id': test['id'],
                               'prediction': y_pred.reshape(-1, )})
    submission.to_csv('submission.csv', index=False)
    logging.info('Finished')


def spacy_tokenise_and_lemmatize(df):
    """
    Remove punctuation as per Spacy and a custom stopword list.
    Lemmatize.
    Set everythin lower case.
    """
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


def sequence_tokens(df, params, train=True):
    """
    Convert the sentences to sequences of integers corresponding to words.
    Save the fitted indexer/tokeniser for use in submission
    """
    if train:
        test = pd.read_csv('Data/test.csv', nrows=params['debug_size'])
        logging.info('Tokenising test set')
        test = spacy_tokenise_and_lemmatize(test)
        tokenizer = Tokenizer()
        logging.info('Creating keras tokeniser and word index')
        tokenizer.fit_on_texts(list(df['comment_text'])
                               + list(test['comment_text']))
        word_index = tokenizer.word_index
        pickle.dump(tokenizer, open('Model_Build/Trained_Models/keras_tokeniser.pkl', 'wb'))
        del test
    else:
        logging.info('Loading pretrained tokenizer')
        with open('../input/tokenisers/keras_tokeniser.pkl', 'rb') as f:
            try:
                tokenizer = pickle.load(f)
            except FileNotFoundError as e:
                print('Can\'t find prefitted tokeniser. May need to upload'
                      'to Kaggle')
                raise e
        word_index = tokenizer.word_index

    logging.info('Sequencing and padding tokenised text')
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

def get_relational_features(df):
    """
    Feature engineering for tree based models
    """
    logging.info('Adding relational features')
    path = Path('..')
    ud_data_path = path / 'input' / 'urban-dictionary-words-dataset' / 'urbandict-word-def.csv'
    banned_data_path = path / 'input' / 'bad-words' / 'swearWords.csv'
    # out_df_path = path / 'Data' / 'additional_features.csv'

    vector_comments = vectorise_input_text(df['comment_text'])
    ud_counts = \
        get_urban_dictionary_features(vector_comments, ud_data_path)
    bw_counts = \
        get_banned_word_list(vector_comments, banned_data_path)
    punc_counts = count_punctuation(df['comment_text'])
    del vector_comments

    feature_set = pd.concat([ud_counts,
                             bw_counts,
                             punc_counts],
                            axis=1)
    #logging.info('Writing relational features to %s', out_df_path)
    #feature_set.to_csv(out_df_path, index=False)
    #logging.info('Relational features complete')
    return feature_set


def vectorise_input_text(comments):
    logging.info('Setting comments to lower case')
    comments = comments.str.lower()
    logging.info('Vectorising comments')
    return comments.str.split(' ', expand=True, n=100)


def get_urban_dictionary_features(comments, ud_data_path):
    """
    Takes a dataframe and appends features relating to the Urban
    Dictionary dataset
    """

    logging.info('Loading Urban Dictionary data')
    ud_data = pd.read_csv(ud_data_path,
                          usecols=['word'],
                          error_bad_lines=False,
                          warn_bad_lines=False)
    ud_data['word'] = ud_data['word'].str.lower()

    # Ensure Urban Dictionary words are unique to save compute
    ud_data.drop_duplicates(inplace=True)

    logging.info('Counting UD corpus occurrences per comment')
    counts = pd.DataFrame(
        comments.isin(ud_data['word']).sum(),
        columns=['n_occurences_in_ud_data']
    )
    logging.info('Complete. Average occurrences per comment: {:.4f}'
                 .format(counts['n_occurences_in_ud_data'].mean()))
    return counts


def get_banned_word_list(comments, banned_word_data_path):
    """
    Takes a dataframe and appends features relating to the banned
    word list
    """

    logging.info('Loading banned words data')
    banned_words = pd.read_csv(banned_word_data_path,
                               header=None)
    banned_words = banned_words[0].str.lower()

    # Ensure Urban Dictionary words are unique to save compute
    banned_words.drop_duplicates(inplace=True)

    logging.info('Counting occurrences of banned words per comment')
    counts = pd.DataFrame(
        comments.isin(banned_words).sum(),
        columns=['n_swearwords']
    )
    logging.info('Complete. Average occurrences per comment: {:.4f}'
                 .format(counts['n_swearwords'].mean()))
    return counts


def count_punctuation(comments):
    """  From Sam's R script  """
    punctuation = {
        'exclamation_mark': '\!',
        'question_mark': '\?',
        'semicolon': '\;',
        'ampersand': '\&',
        'comma': '\,',
        'full_stop': '\.'
    }
    cols = {}
    for name, char in punctuation.items():
        cols[name + '_count'] = comments.str.count(char).tolist()
    return pd.DataFrame(cols)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    main(test_path, model_dir, params)
