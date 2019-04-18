import gc
import numpy as np


def get_embedding_details(name):
    embedding_library = {
        'word2vec': {
            'path': 'Embedding_Build/Trained_Embeddings/'
                    'word2vec_embedding_weights_custom_stopwords.csv',
            'dim': 100
        },
        'ft_common_crawl': {
            'path': 'Embedding_Build/Pretrained_Embeddings/'
                    'crawl-300d-2M.vec/crawl-300d-2M.vec',
            'dim': 300
        },
        'glove_twitter': {
            'path': 'Embedding_Build/Pretrained_Embeddings/'
                    'glove.twitter.27B/glove.twitter.27B.200d.txt',
            'dim': 200
        }
    }
    if name is None:
        return None
    else:
        try:
            embedding = embedding_library[name]
        except KeyError as e:
            print('Unrecognised embedding name {}. Options are:\n'
                  '\t{}'.format(name, list(embedding_library.keys())))
            raise e
        return embedding

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    print('Reading in embedding matrix')
    with open(path, 'r', encoding='latin-1') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_embedding_matrix(word_index, path, embedding_dim):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    print('Building embedding matrix')
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
        except:
            embedding_matrix[i] = embedding_index['unknown']

    del embedding_index
    gc.collect()
    return embedding_matrix