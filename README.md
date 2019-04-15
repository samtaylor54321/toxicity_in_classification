# kaggle_jigsaw
Repo for Kaggle Jigsaw Competition

https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

## Scores

Metric = competition's custom bias metric

| Model | Embedding | Comment | Local CV score | Kaggle leaderboard score |
| --- | --- | --- | --- | --- |
| Single LSTM | Custom word2vec | Default stopwords | 0.9191 |  |
| Single LSTM | Custom word2vec | Custom stopwords | 0.9194 |  |

## Embeddings

Word embeddings to try

- [fastText's Common Crawl embedding](https://fasttext.cc/docs/en/english-vectors.htm)
- [GloVe Twitter embedding](https://nlp.stanford.edu/projects/glove/)
- Train a custom word2vec embedding
- Let keras train one during main model training

