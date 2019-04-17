import os
import spacy
import string
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

INPUT_PATH = os.path.join(os.pardir, 'Data')
OUT_PATH = os.path.join(os.pardir, 'Data', 'Tokenised_Text')
TRAIN_FILE = 'train.csv'

train_path = os.path.join(INPUT_PATH, TRAIN_FILE)
train_out_path = os.path.join(OUT_PATH, 'tokenised_train.csv')


def tokenise(path, out_path=None, submission=False):
    # Identify punctuation and stopwords to ignore
    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

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

    if submission:
        return process(path)
    else:
        processed_data = process(path)
        processed_data.to_csv(train_out_path)

if __name__ == '__main__':
    tokenise(train_path, train_out_path)
