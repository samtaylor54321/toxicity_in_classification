import logging
import pandas as pd
from pathlib import Path


def get_relational_features(df):
    """
    Feature engineering for tree based models
    """
    logging.info('Adding relational features')
    path = Path('..')
    ud_data_path = path / 'Data' / 'urbandict-word-def.csv'
    banned_data_path = path / 'Data' / 'swearWords.csv'
    out_df_path = path / 'Data' / 'additional_features.csv'

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
    logging.info('Writing relational features to %s', out_df_path)
    feature_set.to_csv(out_df_path, index=False)
    logging.info('Relational features complete')
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
        cols[name + '_count'] = comments.str.count(char).to_list()
    return pd.DataFrame(cols)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    df = pd.read_csv('../Data/train.csv')
    get_relational_features(df)
