import logging
import pandas as pd
from pathlib import Path


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    path = Path('..')
    ud_data_path = path / 'Data' / 'urbandict-word-def.csv'
    train_df_path = path / 'Data' / 'train.csv'
    out_df_path = path / 'Data' / 'train_with_features.csv'

    logging.info('Loading main dataset')
    df = pd.read_csv(train_df_path)
    df['num_words_in_ud_corpus'] = \
        get_urban_dictionary_features(df['comment_text'], ud_data_path)

    logging.info('Writing augmented dataset')
    df.to_csv(out_df_path, index=False)
    logging.info('Complete')


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

    logging.info('Setting comments to lower case')
    comments = comments.str.lower()
    logging.info('Vectorising comments')
    comments = comments.str.split(' ', expand=True)

    logging.info('Counting UD corpus occurrences per comment')
    counts = comments.isin(ud_data['word']).sum()
    logging.info('Complete. Average occurrences per comment: {:.4f}'
                 .format(counts.mean()))

    return counts


if __name__ == '__main__':
    main()
