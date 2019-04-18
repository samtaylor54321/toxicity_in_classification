# Kaggle Jigsaw Bias in Toxicity Classification challenge

Repo for Kaggle Jigsaw Competition

https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

## External Datasets/Sources - Must be declared if used

Reddit Corpus

https://www.reddit.com/r/datasets/comments/65o7py/updated_reddit_comment_dataset_as_torrents/

Youtube Corpus

https://www.kaggle.com/datasnaek/youtube

Swearword Corpus

http://www.bannedwordlist.com/ 

Emoji Sentiment Ranking

http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html 

Urban Dictionary Corpus

https://www.kaggle.com/therohk/urban-dictionary-words-dataset 

Academic Research

Hate Speech Classifier 

https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665/14843  (Paper)

https://github.com/t-davidson/hate-speech-and-offensive-language (Github)

API

Perspective API

https://www.perspectiveapi.com/#/


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

