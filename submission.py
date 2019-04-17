import pandas as pd
from pathlib import Path
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from Vector_Build.tokenise import tokenise

# Paths to things we'll need
path = ('.')
test_path = path / 'Data' / 'test.csv'
sequencer = path / 'Model_Build' / 'Trained_Models' / 'word2vec_model.pkl'
model = path / 'Results' / '20190412_18.18.48_score_nan' / 'MODEL_lstm.h5'

# Load and tokenise
test = pd.read_csv(test_path)
test = tokenise(test)
test['comment_text'].fillna('emptyword', inplace=True)

# Convert text to sequenced word indices
sequencer = pickle.load(sequencer)
sequences = []
for row in test['comment_text'].str.split(' ').tolist():
    sequences.append([sequencer.vocab[word].index for word in row])
sequences = pad_sequences(sequences, maxlen=100)
sequences = pd.DataFrame(sequences).values

# Run model
model = load_model(model)
y_pred = model.predict(sequences)

# Submit
submission = pd.DataFrame({'id': test['id'],
                           'prediction': y_pred})
submission.to_csv('submission.csv', index=False)
