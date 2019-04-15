import os
import string
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

INPUT_PATH = os.path.join(os.pardir, 'Data')
OUT_PATH = os.path.join(os.pardir, 'Data', 'Tokenised_Text')
TRAIN_FILE = 'train.csv'

train_path = os.path.join(INPUT_PATH, TRAIN_FILE)
train_out_path = os.path.join(OUT_PATH, 'tokenised_train_custom_stopwords.csv')


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


if __name__ == '__main__':
    processed_data = tokenise(train_path)
    processed_data.to_csv(train_out_path)